//! The session: a loaded graph plus its state, and the few verbs that act
//! on it. Deliberately narrow, because this is the surface the Ruby
//! extension binds (docs/plan.md section 4).

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use mlx_rs::transforms::{eval, value_and_grad_with_argnums};
use mlx_rs::Array;
use serde::Deserialize;

use crate::graph::{Graph, GraphConfig};
use crate::interp;

/// A tensor as it arrives from outside: flat data plus its shape. The only
/// tensor shape that crosses the boundary, in either direction.
#[derive(Deserialize)]
pub struct Tensor {
    pub shape: Vec<i32>,
    pub data: Vec<f32>,
}

#[derive(Deserialize)]
struct Bindings {
    inputs: BTreeMap<String, Tensor>,
    params: BTreeMap<String, Tensor>,
}

pub struct Session {
    graph: Graph,
    params: Vec<Array>,
    inputs: BTreeMap<String, Array>,
    step: usize,
    last_loss: f32,
    lr: f32,
}

impl Session {
    /// Loads a graph and its bindings. `graph_json` is the canonical JSON of
    /// a GraphConfig; the model named "spike", or the only model, is taken.
    pub fn open(graph_json: &str, bindings_json: &str) -> Result<Self> {
        let config: GraphConfig = serde_json::from_str(graph_json).context("parsing the graph")?;
        let mut models = config.models;
        let graph = models
            .remove("spike")
            .or_else(|| models.pop_first().map(|(_, g)| g))
            .context("the graph has no models")?;

        let bindings: Bindings =
            serde_json::from_str(bindings_json).context("parsing the bindings")?;
        let params = graph
            .parameters
            .iter()
            .map(|p| {
                let t = bindings
                    .params
                    .get(&p.path)
                    .with_context(|| format!("bindings are missing parameter {:?}", p.path))?;
                anyhow::ensure!(
                    t.shape == p.shape,
                    "parameter {:?}: bound shape {:?} is not the declared {:?}",
                    p.path,
                    t.shape,
                    p.shape
                );
                Ok(Array::from_slice(&t.data, &t.shape))
            })
            .collect::<Result<Vec<_>>>()?;
        let inputs = bindings
            .inputs
            .iter()
            .map(|(name, t)| (name.clone(), Array::from_slice(&t.data, &t.shape)))
            .collect();

        Ok(Self {
            graph,
            params,
            inputs,
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

    /// Runs `n` steps of plain SGD and returns the last loss. Long-running
    /// and free of any Ruby: the extension calls this with the GVL released.
    pub fn run_steps(&mut self, n: usize) -> Result<f32> {
        let lr = Array::from_f32(self.lr);
        for _ in 0..n {
            let (loss, grads) = self.loss_and_grads()?;
            self.params = self
                .params
                .iter()
                .zip(&grads)
                .map(|(p, g)| Ok(p.subtract(g.multiply(&lr)?)?))
                .collect::<Result<Vec<_>>>()?;
            eval(self.params.iter())?;
            self.last_loss = loss.item::<f32>();
            self.step += 1;
        }
        Ok(self.last_loss)
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
        let array = self.params[index].contiguous()?;
        eval(std::iter::once(&array))?;
        Ok(Tensor {
            shape: array.shape().to_vec(),
            data: array.as_slice::<f32>().to_vec(),
        })
    }

    /// The loss and one gradient per parameter, for the currently bound
    /// inputs.
    pub fn loss_and_grads(&self) -> Result<(Array, Vec<Array>)> {
        let argnums: Vec<i32> = (0..self.params.len() as i32).collect();
        let graph = &self.graph;
        let inputs = &self.inputs;
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

    /// Gradients as copies, by parameter path, in declaration order.
    pub fn gradients(&self) -> Result<Vec<(String, Tensor)>> {
        let (_, grads) = self.loss_and_grads()?;
        self.graph
            .parameters
            .iter()
            .zip(grads)
            .map(|(spec, grad)| {
                let grad = grad.contiguous()?;
                eval(std::iter::once(&grad))?;
                Ok((
                    spec.path.clone(),
                    Tensor {
                        shape: grad.shape().to_vec(),
                        data: grad.as_slice::<f32>().to_vec(),
                    },
                ))
            })
            .collect()
    }
}
