//! The session: a loaded graph plus its state, and the few verbs that act
//! on it. Deliberately narrow, because this is the surface the Ruby
//! extension binds (docs/plan.md section 4).
//!
//! A session owns parameters and counters; it does not own data. Every step
//! is given its batch (docs/plan.md section 5A.2): the caller decides what
//! the model sees, and nothing here reads a file or calls back into Ruby.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use mlx_rs::transforms::{eval, value_and_grad_with_argnums};
use mlx_rs::Array;
use serde::Deserialize;

use crate::graph::{Graph, GraphConfig};
use crate::interp;

/// A tensor as it crosses the boundary: flat data plus its shape. The only
/// tensor shape that travels, in either direction, and always as a copy.
#[derive(Deserialize)]
pub struct Tensor {
    pub shape: Vec<i32>,
    pub data: Vec<f32>,
}

impl Tensor {
    fn to_array(&self) -> Array {
        Array::from_slice(&self.data, &self.shape)
    }
}

/// The same tensor, with its data as native-endian f32 bytes rather than
/// JSON numbers.
///
/// Measurement drove this (docs/plan.md section 5A.2.1): serializing a
/// batch as JSON cost two thirds of a step at 512 rows, while the call
/// boundary itself was noise. The shape stays JSON - it is a handful of
/// integers, and readable - and only the payload goes packed.
pub struct PackedTensor {
    pub shape: Vec<i32>,
    pub bytes: Vec<u8>,
}

impl PackedTensor {
    fn to_tensor(&self, name: &str) -> Result<Tensor> {
        anyhow::ensure!(
            self.bytes.len() % 4 == 0,
            "input {name:?}: {} bytes is not a whole number of f32 values",
            self.bytes.len()
        );
        let data = self
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Ok(Tensor {
            shape: self.shape.clone(),
            data,
        })
    }
}

/// One step's inputs, by graph input name.
pub type Batch = BTreeMap<String, Tensor>;

/// The same, packed. Converted to a [`Batch`] on arrival.
pub type PackedBatch = BTreeMap<String, PackedTensor>;

/// Unpacks a batch, naming the input if the bytes do not divide.
pub fn unpack(packed: &PackedBatch) -> Result<Batch> {
    packed
        .iter()
        .map(|(name, t)| Ok((name.clone(), t.to_tensor(name)?)))
        .collect()
}

/// The initial parameters, by path.
#[derive(Deserialize)]
struct Weights {
    params: BTreeMap<String, Tensor>,
}

pub struct Session {
    graph: Graph,
    params: Vec<Array>,
    step: usize,
    last_loss: f32,
    lr: f32,
}

impl Session {
    /// Loads a graph and its initial parameters. `graph_json` is the
    /// canonical JSON of a GraphConfig; the model named "spike", or the only
    /// model, is taken. Data comes later, one batch per step.
    pub fn open(graph_json: &str, weights_json: &str) -> Result<Self> {
        let config: GraphConfig = serde_json::from_str(graph_json).context("parsing the graph")?;
        let mut models = config.models;
        let graph = models
            .remove("spike")
            .or_else(|| models.pop_first().map(|(_, g)| g))
            .context("the graph has no models")?;

        let weights: Weights =
            serde_json::from_str(weights_json).context("parsing the initial parameters")?;
        let params = graph
            .parameters
            .iter()
            .map(|p| {
                let t = weights
                    .params
                    .get(&p.path)
                    .with_context(|| format!("missing parameter {:?}", p.path))?;
                anyhow::ensure!(
                    t.shape == p.shape,
                    "parameter {:?}: given shape {:?} is not the declared {:?}",
                    p.path,
                    t.shape,
                    p.shape
                );
                Ok(t.to_array())
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            graph,
            params,
            step: 0,
            last_loss: f32::NAN,
            lr: 0.1,
        })
    }

    pub fn step(&self) -> usize {
        self.step
    }

    pub fn loss(&self) -> f32 {
        self.last_loss
    }

    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// One knob, for now. Effect begins with the next step.
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    pub fn parameter_paths(&self) -> Vec<String> {
        self.graph.parameters.iter().map(|p| p.path.clone()).collect()
    }

    pub fn input_names(&self) -> Vec<String> {
        self.graph.inputs.iter().map(|i| i.name.clone()).collect()
    }

    /// One step on `batch`: forward, backward, SGD update. Long-running and
    /// free of any Ruby, so the extension calls it with the GVL released.
    pub fn run_step(&mut self, batch: &Batch) -> Result<f32> {
        let inputs = self.bind(batch)?;
        self.update(&inputs)
    }

    /// `n` steps, each on its own batch. The batches are given up front, so
    /// the engine never asks anyone for data mid-span.
    pub fn run_steps(&mut self, batches: &[Batch]) -> Result<f32> {
        anyhow::ensure!(!batches.is_empty(), "a span needs at least one batch");
        for batch in batches {
            let inputs = self.bind(batch)?;
            self.update(&inputs)?;
        }
        Ok(self.last_loss)
    }

    /// The loss and one gradient per parameter for `batch`, without updating
    /// anything.
    pub fn loss_and_grads(&self, batch: &Batch) -> Result<(Array, Vec<Array>)> {
        let inputs = self.bind(batch)?;
        self.differentiate(&inputs)
    }

    /// Gradients as copies, by parameter path, in declaration order.
    pub fn gradients(&self, batch: &Batch) -> Result<Vec<(String, Tensor)>> {
        let (_, grads) = self.loss_and_grads(batch)?;
        self.graph
            .parameters
            .iter()
            .zip(grads)
            .map(|(spec, grad)| Ok((spec.path.clone(), to_tensor(&grad)?)))
            .collect()
    }

    /// A copy of one parameter, by path. Copies, not handles: nothing that
    /// lives on the device escapes this crate.
    pub fn fetch(&self, path: &str) -> Result<Tensor> {
        let index = self
            .graph
            .parameters
            .iter()
            .position(|p| p.path == path)
            .with_context(|| format!("no parameter named {path:?}"))?;
        to_tensor(&self.params[index])
    }

    /// Binds a batch to the graph's inputs, refusing anything the graph did
    /// not declare and anything it declared but the batch omits. Shapes are
    /// checked against the declaration, where a null dimension is symbolic
    /// and may differ from step to step (docs/plan.md section 6.2).
    fn bind(&self, batch: &Batch) -> Result<BTreeMap<String, Array>> {
        for name in batch.keys() {
            anyhow::ensure!(
                self.graph.inputs.iter().any(|i| &i.name == name),
                "the graph has no input named {name:?}"
            );
        }
        self.graph
            .inputs
            .iter()
            .map(|spec| {
                let t = batch
                    .get(&spec.name)
                    .with_context(|| format!("the batch is missing input {:?}", spec.name))?;
                anyhow::ensure!(
                    t.shape.len() == spec.shape.len(),
                    "input {:?}: rank {} does not match the declared {}",
                    spec.name,
                    t.shape.len(),
                    spec.shape.len()
                );
                for (axis, (given, declared)) in t.shape.iter().zip(&spec.shape).enumerate() {
                    if let Some(declared) = declared {
                        anyhow::ensure!(
                            given == declared,
                            "input {:?}: dimension {axis} is {given}, declared {declared}",
                            spec.name
                        );
                    }
                }
                let expected: usize = t.shape.iter().map(|d| *d as usize).product();
                anyhow::ensure!(
                    t.data.len() == expected,
                    "input {:?}: {} values for shape {:?}",
                    spec.name,
                    t.data.len(),
                    t.shape
                );
                Ok((spec.name.clone(), t.to_array()))
            })
            .collect()
    }

    fn differentiate(&self, inputs: &BTreeMap<String, Array>) -> Result<(Array, Vec<Array>)> {
        let argnums: Vec<i32> = (0..self.params.len() as i32).collect();
        let graph = &self.graph;
        let fun = |ps: &[Array]| -> std::result::Result<Vec<Array>, mlx_rs::error::Exception> {
            interp::evaluate(graph, ps, inputs)
        };
        let mut vg = value_and_grad_with_argnums(fun, &argnums[..]);
        let (mut values, grads) = vg(&self.params)?;
        anyhow::ensure!(
            values.len() == 1,
            "the graph must output exactly one value (the loss), got {}",
            values.len()
        );
        let loss = values.remove(0);
        eval(std::iter::once(&loss).chain(grads.iter()))?;
        Ok((loss, grads))
    }

    fn update(&mut self, inputs: &BTreeMap<String, Array>) -> Result<f32> {
        let (loss, grads) = self.differentiate(inputs)?;
        let lr = Array::from_f32(self.lr);
        self.params = self
            .params
            .iter()
            .zip(&grads)
            .map(|(p, g)| Ok(p.subtract(g.multiply(&lr)?)?))
            .collect::<Result<Vec<_>>>()?;
        eval(self.params.iter())?;
        self.last_loss = loss.item::<f32>();
        self.step += 1;
        Ok(self.last_loss)
    }
}

/// A device array as a copy on the host. Contiguous first: a gradient can
/// come back strided (through a transpose, say), and reading it out needs
/// contiguous memory.
fn to_tensor(array: &Array) -> Result<Tensor> {
    let array = array.contiguous()?;
    eval(std::iter::once(&array))?;
    Ok(Tensor {
        shape: array.shape().to_vec(),
        data: array.as_slice::<f32>().to_vec(),
    })
}
