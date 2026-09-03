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
use mlx_rs::{Array, Dtype};
use serde::Deserialize;

use crate::checkpoint;
use crate::graph::{Graph, GraphConfig};
use crate::interp;
use crate::optimizer::{Config as OptimizerConfig, Optimizer};

/// A tensor as it crosses the boundary: a dtype, a shape, and the values.
/// Always a copy, never a handle.
///
/// The dtype travels because a graph can declare an i32 input - which is
/// what an embedding reads - and a boundary that assumed f32 could not
/// carry one (docs/plan.md section 5A.2).
pub struct Tensor {
    pub dtype: Dtype,
    pub shape: Vec<i32>,
    pub values: Values,
}

/// The payload, in the only two forms the boundary carries.
pub enum Values {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

impl Values {
    pub fn len(&self) -> usize {
        match self {
            Values::F32(v) => v.len(),
            Values::I32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Tensor {
    fn to_array(&self) -> Array {
        match &self.values {
            Values::F32(data) => Array::from_slice(data, &self.shape),
            Values::I32(data) => Array::from_slice(data, &self.shape),
        }
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
    /// "f32" or "i32", as the graph declares dtypes.
    pub dtype: String,
    pub shape: Vec<i32>,
    pub bytes: Vec<u8>,
}

impl PackedTensor {
    fn to_tensor(&self, name: &str) -> Result<Tensor> {
        anyhow::ensure!(
            self.bytes.len() % 4 == 0,
            "input {name:?}: {} bytes is not a whole number of 4-byte values",
            self.bytes.len()
        );
        let words = self.bytes.chunks_exact(4).map(|b| [b[0], b[1], b[2], b[3]]);
        let (dtype, values) = match self.dtype.as_str() {
            "f32" => (
                Dtype::Float32,
                Values::F32(words.map(f32::from_ne_bytes).collect()),
            ),
            "i32" => (
                Dtype::Int32,
                Values::I32(words.map(i32::from_ne_bytes).collect()),
            ),
            other => anyhow::bail!("input {name:?}: dtype {other:?} does not cross the boundary"),
        };
        Ok(Tensor {
            dtype,
            shape: self.shape.clone(),
            values,
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

/// One initial parameter as JSON carries it. Parameters are f32 (the
/// engine differentiates them), so this needs no dtype; a batch does, and
/// travels packed instead (see [`PackedTensor`]).
#[derive(Deserialize)]
struct InitialTensor {
    shape: Vec<i32>,
    data: Vec<f32>,
}

/// The initial parameters, by path.
#[derive(Deserialize)]
struct Weights {
    params: BTreeMap<String, InitialTensor>,
}

/// One model as the session holds it: its graph, and where its parameters
/// sit in the run's flat parameter vector.
struct Model {
    name: String,
    graph: Graph,
    /// Range into `params`, in the order the config declared.
    slice: std::ops::Range<usize>,
}

pub struct Session {
    models: Vec<Model>,
    /// The graph that reaches the loss. When a config declares none, the
    /// single model's own output is the loss.
    objective: Option<Graph>,
    /// Every parameter of every model, model by model in name order. This
    /// order is the contract the Ruby side also follows.
    params: Vec<Array>,
    /// Qualified paths ("student.head.weight"), parallel to `params`.
    paths: Vec<String>,
    /// Positions in `params` that autodiff differentiates.
    argnums: Vec<i32>,
    /// The update rule and its slots. Half of what a checkpoint restores.
    optimizer: Optimizer,
    /// The digest of the GraphConfig this session runs, so a checkpoint can
    /// say which description its state belongs to.
    config_digest: String,
    /// The RNG, held as state rather than left to a global. Every step
    /// splits it, so the sequence of draws is a function of the seed and
    /// the step count, and a resumed run draws what a continuous one would
    /// (docs/plan.md section 11.1).
    rng: Array,
    seed: u64,
    step: usize,
    last_loss: f32,
}

impl Session {
    /// Loads a GraphConfig and its initial parameters. Parameters are given
    /// by qualified path ("student.head.weight"), which is also the order
    /// the engine keeps them in. Data comes later, one batch per step.
    pub fn open(graph_json: &str, weights_json: &str) -> Result<Self> {
        Self::open_with(graph_json, weights_json, OptimizerConfig::Sgd { lr: 0.1 })
    }

    /// The same, with the update rule named.
    pub fn open_with(
        graph_json: &str,
        weights_json: &str,
        optimizer: OptimizerConfig,
    ) -> Result<Self> {
        // The digest of exactly the bytes the caller handed over, which is
        // what the Ruby side computed and what a checkpoint records.
        let config_digest = {
            use sha2::{Digest, Sha256};
            format!("{:x}", Sha256::digest(graph_json.as_bytes()))
        };
        let config: GraphConfig = serde_json::from_str(graph_json).context("parsing the graph")?;
        anyhow::ensure!(!config.models.is_empty(), "the graph has no models");
        let weights: Weights =
            serde_json::from_str(weights_json).context("parsing the initial parameters")?;

        let mut models = Vec::new();
        let mut params = Vec::new();
        let mut paths = Vec::new();
        let mut argnums = Vec::new();
        // BTreeMap iterates in name order, which is the declared order.
        for (name, graph) in config.models {
            let trained = config.train.contains(&name);
            let start = params.len();
            for spec in &graph.parameters {
                let path = format!("{name}.{}", spec.path);
                let t = weights
                    .params
                    .get(&path)
                    .with_context(|| format!("missing parameter {path:?}"))?;
                anyhow::ensure!(
                    t.shape == spec.shape,
                    "parameter {path:?}: given shape {:?} is not the declared {:?}",
                    t.shape,
                    spec.shape
                );
                if trained && spec.trainable {
                    argnums.push(params.len() as i32);
                }
                params.push(Array::from_slice(&t.data, &t.shape));
                paths.push(path);
            }
            let slice = start..params.len();
            models.push(Model { name, graph, slice });
        }
        anyhow::ensure!(
            !argnums.is_empty(),
            "nothing to train: no model in {:?} has trainable parameters",
            config.train
        );

        let optimizer = Optimizer::new(optimizer, &params, &argnums)?;
        let seed = 0;
        Ok(Self {
            rng: mlx_rs::random::key(seed)?,
            seed,
            models,
            objective: config.objective,
            params,
            paths,
            argnums,
            optimizer,
            config_digest,
            step: 0,
            last_loss: f32::NAN,
        })
    }

    pub fn step(&self) -> usize {
        self.step
    }

    pub fn loss(&self) -> f32 {
        self.last_loss
    }

    pub fn lr(&self) -> f32 {
        self.optimizer.config().lr()
    }

    /// A knob: effect begins with the next step.
    pub fn set_lr(&mut self, lr: f32) {
        self.optimizer.config_mut().set_lr(lr);
    }

    /// What update rule this session runs, as data.
    pub fn optimizer_config(&self) -> &OptimizerConfig {
        self.optimizer.config()
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Restarts the RNG. A knob like any other: after this the draws are a
    /// function of the new seed alone.
    pub fn set_seed(&mut self, seed: u64) -> Result<()> {
        self.seed = seed;
        self.rng = mlx_rs::random::key(seed)?;
        Ok(())
    }

    /// One step on `batch`: forward, backward, optimizer update. Long-
    /// running and free of any Ruby, so the extension calls it with the GVL
    /// released.
    pub fn run_step(&mut self, batch: &Batch) -> Result<f32> {
        let fields = self.bind(batch)?;
        self.update(&fields)
    }

    /// One step per batch. The batches are given up front, so the engine
    /// never asks anyone for data mid-span.
    pub fn run_steps(&mut self, batches: &[Batch]) -> Result<f32> {
        anyhow::ensure!(!batches.is_empty(), "a span needs at least one batch");
        for batch in batches {
            let fields = self.bind(batch)?;
            self.update(&fields)?;
        }
        Ok(self.last_loss)
    }

    /// The loss and one gradient per differentiated parameter for `batch`,
    /// without updating anything.
    pub fn loss_and_grads(&self, batch: &Batch) -> Result<(Array, Vec<Array>)> {
        let fields = self.bind(batch)?;
        self.differentiate(&fields)
    }

    /// Gradients as copies, by qualified parameter path. Only differentiated
    /// parameters appear: a frozen model's have none.
    pub fn gradients(&self, batch: &Batch) -> Result<Vec<(String, Tensor)>> {
        let (_, grads) = self.loss_and_grads(batch)?;
        self.argnums
            .iter()
            .zip(grads)
            .map(|(&i, grad)| Ok((self.paths[i as usize].clone(), to_tensor(&grad)?)))
            .collect()
    }

    /// A copy of one parameter, by qualified path. Copies, not handles:
    /// nothing that lives on the device escapes this crate.
    pub fn fetch(&self, path: &str) -> Result<Tensor> {
        let index = self
            .paths
            .iter()
            .position(|p| p == path)
            .with_context(|| format!("no parameter named {path:?}"))?;
        to_tensor(&self.params[index])
    }

    /// Qualified parameter paths, in the order the engine keeps them.
    pub fn parameter_paths(&self) -> Vec<String> {
        self.paths.clone()
    }

    /// Every batch field the run reads, across the models and the objective.
    pub fn input_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .models
            .iter()
            .map(|m| &m.graph)
            .chain(self.objective.iter())
            .flat_map(|g| g.inputs.iter().filter_map(|i| i.batch_field()))
            .map(str::to_string)
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// Writes the run's state: parameters, optimizer slots, counters, and
    /// what they belong to. Atomic (docs/plan.md section 11.2).
    pub fn save(&self, dir: &str) -> Result<String> {
        let (m, v) = self.optimizer.slots();
        let state = checkpoint::State {
            config_digest: &self.config_digest,
            step: self.step,
            optimizer: self.optimizer.config(),
            optimizer_steps: self.optimizer.steps_taken(),
            parameters: self
                .paths
                .iter()
                .cloned()
                .zip(self.params.iter())
                .collect(),
            argnums: &self.argnums,
            slots: (m, v),
            rng: &self.rng,
            seed: self.seed,
        };
        Ok(checkpoint::write(dir, state)?.display().to_string())
    }

    /// Restores state written by [`Session::save`], refusing anything that
    /// does not belong to this session: another description, another
    /// optimizer, a parameter of another shape, a missing slot.
    ///
    /// Nothing is committed until everything has been read, checked and
    /// evaluated. A checkpoint that turns out to be wrong halfway leaves
    /// this session exactly as it was, which is what StepError promises
    /// (docs/plan.md section 5A.4).
    pub fn restore(&mut self, dir: &str) -> Result<()> {
        let loaded = checkpoint::read(dir)?;
        let manifest = &loaded.manifest;
        anyhow::ensure!(
            manifest.config_digest == self.config_digest,
            "this checkpoint belongs to another graph (digest {}, not {})",
            &manifest.config_digest[..12.min(manifest.config_digest.len())],
            &self.config_digest[..12]
        );
        anyhow::ensure!(
            &manifest.optimizer == self.optimizer.config(),
            "this checkpoint was written by a different optimizer ({:?})",
            manifest.optimizer
        );
        anyhow::ensure!(
            manifest.parameters.len() == self.paths.len(),
            "this checkpoint has {} parameters, this session has {}",
            manifest.parameters.len(),
            self.paths.len()
        );

        let mut params = Vec::with_capacity(self.params.len());
        for (i, path) in self.paths.iter().enumerate() {
            let entry = &manifest.parameters[i];
            anyhow::ensure!(
                &entry.path == path,
                "parameter {i} is {:?} here and {:?} in the checkpoint",
                path,
                entry.path
            );
            let array = loaded
                .parameters
                .get(path)
                .with_context(|| format!("the checkpoint has no parameter {path:?}"))?;
            anyhow::ensure!(
                array.shape() == self.params[i].shape(),
                "parameter {path:?}: checkpoint shape {:?} is not {:?}",
                array.shape(),
                self.params[i].shape()
            );
            anyhow::ensure!(
                array.dtype() == self.params[i].dtype(),
                "parameter {path:?}: checkpoint dtype {:?} is not {:?}",
                array.dtype(),
                self.params[i].dtype()
            );
            params.push(array.clone());
        }

        // The optimizer's slots, all of them or none. An AdamW session
        // restored without moments used to index past the end of an empty
        // vector on its next step.
        let (mut m, mut v) = (Vec::new(), Vec::new());
        if self.optimizer.wants_slots() {
            anyhow::ensure!(
                !loaded.slots.is_empty(),
                "this checkpoint has no optimizer state, and {} needs it",
                self.optimizer.config().name()
            );
            for &i in &self.argnums {
                let path = &self.paths[i as usize];
                let (slot_m, slot_v) = loaded.slots.get(path).with_context(|| {
                    format!("the checkpoint has no optimizer state for {path:?}")
                })?;
                anyhow::ensure!(
                    slot_m.shape() == self.params[i as usize].shape()
                        && slot_v.shape() == self.params[i as usize].shape(),
                    "optimizer state for {path:?} has the wrong shape"
                );
                m.push(slot_m.clone());
                v.push(slot_v.clone());
            }
        } else {
            anyhow::ensure!(
                loaded.slots.is_empty(),
                "this checkpoint carries optimizer state, and {} has none",
                self.optimizer.config().name()
            );
        }

        let rng = loaded.rng.context("the checkpoint has no RNG state")?;

        // Everything is here and consistent; make it real before touching
        // this session, so a failure below cannot leave it half restored.
        eval(params.iter().chain(m.iter()).chain(v.iter()).chain(std::iter::once(&rng)))?;

        self.params = params;
        self.optimizer.restore(m, v, manifest.optimizer_steps);
        self.rng = rng;
        self.seed = manifest.seed;
        self.step = manifest.step;
        self.last_loss = f32::NAN;
        Ok(())
    }

    /// Turns a batch into the fields the run reads, refusing anything no
    /// graph declared and anything a graph declared but the batch omits.
    /// Shapes are checked against the declaration, where a null dimension is
    /// symbolic and may differ from step to step (docs/plan.md 6.2).
    fn bind(&self, batch: &Batch) -> Result<BTreeMap<String, Array>> {
        let wanted = self.input_names();
        for name in batch.keys() {
            anyhow::ensure!(
                wanted.iter().any(|w| w == name),
                "no input named {name:?} is read here (this run reads {wanted:?})"
            );
        }

        let mut fields = BTreeMap::new();
        for graph in self.models.iter().map(|m| &m.graph).chain(self.objective.iter()) {
            for spec in &graph.inputs {
                let Some(field) = spec.batch_field() else {
                    continue;
                };
                if fields.contains_key(field) {
                    continue;
                }
                let t = batch
                    .get(field)
                    .with_context(|| format!("the batch is missing input {field:?}"))?;
                anyhow::ensure!(
                    t.shape.len() == spec.shape.len(),
                    "input {field:?}: rank {} does not match the declared {}",
                    t.shape.len(),
                    spec.shape.len()
                );
                for (axis, (given, declared)) in t.shape.iter().zip(&spec.shape).enumerate() {
                    if let Some(declared) = declared {
                        anyhow::ensure!(
                            given == declared,
                            "input {field:?}: dimension {axis} is {given}, declared {declared}"
                        );
                    }
                }
                let declared = dtype_named(&spec.dtype)
                    .with_context(|| format!("input {field:?}: unknown dtype in the graph"))?;
                anyhow::ensure!(
                    t.dtype == declared,
                    "input {field:?}: given {:?}, declared {}",
                    t.dtype,
                    spec.dtype
                );
                let expected: usize = t.shape.iter().map(|d| *d as usize).product();
                anyhow::ensure!(
                    t.values.len() == expected,
                    "input {field:?}: {} values for shape {:?}",
                    t.values.len(),
                    t.shape
                );
                fields.insert(field.to_string(), t.to_array());
            }
        }
        Ok(fields)
    }

    /// Runs every model, then the objective over their outputs, and returns
    /// the loss. A model output feeding several places is computed once
    /// (docs/plan.md 5A.3).
    fn forward(
        models: &[Model],
        objective: Option<&Graph>,
        params: &[Array],
        fields: &BTreeMap<String, Array>,
        rng: &Array,
    ) -> std::result::Result<Array, mlx_rs::error::Exception> {
        let fail = |what: String| mlx_rs::error::Exception::custom(what);
        let mut outputs: BTreeMap<(String, String), Array> = BTreeMap::new();

        // One key per graph, split from the step's, so a model and the
        // objective never draw the same numbers.
        let mut key = rng.clone();
        for model in models {
            let inputs = Self::resolve(&model.graph, fields, &outputs, &model.name)?;
            let (next, mine) = mlx_rs::random::split(&key, 2)?;
            key = next;
            let produced =
                interp::evaluate(&model.graph, &params[model.slice.clone()], &inputs, Some(&mine))?;
            for (name, value) in produced {
                outputs.insert((model.name.clone(), name), value);
            }
        }

        let Some(objective) = objective else {
            // No objective: a single model's only output is the loss.
            let mut values = outputs.into_values();
            let loss = values
                .next()
                .ok_or_else(|| fail("the model produced no output".into()))?;
            return Ok(loss);
        };

        let inputs = Self::resolve(objective, fields, &outputs, "objective")?;
        let produced = interp::evaluate(objective, &[], &inputs, Some(&key))?;
        let mut values = produced.into_values();
        let loss = values
            .next()
            .ok_or_else(|| fail("the objective produced no output".into()))?;
        Ok(loss)
    }

    /// One graph's inputs, each read from its declared source.
    fn resolve(
        graph: &Graph,
        fields: &BTreeMap<String, Array>,
        outputs: &BTreeMap<(String, String), Array>,
        where_: &str,
    ) -> std::result::Result<BTreeMap<String, Array>, mlx_rs::error::Exception> {
        graph
            .inputs
            .iter()
            .map(|spec| {
                let value = if let Some(field) = spec.batch_field() {
                    fields.get(field).cloned().ok_or_else(|| {
                        mlx_rs::error::Exception::custom(format!(
                            "{where_}: no batch field {field:?}"
                        ))
                    })?
                } else {
                    let (model, output) = spec.model_output().ok_or_else(|| {
                        mlx_rs::error::Exception::custom(format!(
                            "{where_}: input {:?} has no source",
                            spec.name
                        ))
                    })?;
                    outputs
                        .get(&(model.to_string(), output.to_string()))
                        .cloned()
                        .ok_or_else(|| {
                            mlx_rs::error::Exception::custom(format!(
                                "{where_}: {model}.{output} has not been produced; \
                                 a model must be declared before what reads it"
                            ))
                        })?
                };
                Ok((spec.name.clone(), value))
            })
            .collect()
    }

    fn differentiate(&self, fields: &BTreeMap<String, Array>) -> Result<(Array, Vec<Array>)> {
        let models = &self.models;
        let objective = self.objective.as_ref();
        let rng = &self.rng;
        let fun = |ps: &[Array]| -> std::result::Result<Vec<Array>, mlx_rs::error::Exception> {
            Self::forward(models, objective, ps, fields, rng).map(|loss| vec![loss])
        };
        let mut vg = value_and_grad_with_argnums(fun, &self.argnums[..]);
        let (mut values, grads) = vg(&self.params)?;
        let loss = values.remove(0);
        eval(std::iter::once(&loss).chain(grads.iter()))?;
        Ok((loss, grads))
    }

    /// One optimizer step on the differentiated parameters. A frozen
    /// model's are left exactly as they were.
    /// One step, as a transaction: the next parameters, slots and key are
    /// built and evaluated first, and only then does this session become
    /// them. A step that fails leaves everything as it was, which is what
    /// StepError promises.
    fn update(&mut self, fields: &BTreeMap<String, Array>) -> Result<f32> {
        let (loss, grads) = self.differentiate(fields)?;
        let mut params = self.params.clone();
        let next_optimizer = self.optimizer.next(&mut params, &self.argnums, &grads)?;
        let (rng, _) = mlx_rs::random::split(&self.rng, 2)?;

        let (m, v) = next_optimizer.slots();
        eval(
            params
                .iter()
                .chain(m.iter())
                .chain(v.iter())
                .chain(std::iter::once(&rng))
                .chain(std::iter::once(&loss)),
        )?;

        self.params = params;
        self.optimizer = next_optimizer;
        self.rng = rng;
        self.last_loss = loss.item::<f32>();
        self.step += 1;
        Ok(self.last_loss)
    }
}

/// The dtypes a graph may name, as MLX knows them.
fn dtype_named(name: &str) -> Option<Dtype> {
    match name {
        "f32" => Some(Dtype::Float32),
        "bf16" => Some(Dtype::Bfloat16),
        "i32" => Some(Dtype::Int32),
        "bool" => Some(Dtype::Bool),
        _ => None,
    }
}

/// A device array as a copy on the host. Contiguous first: a gradient can
/// come back strided (through a transpose, say), and reading it out needs
/// contiguous memory.
fn to_tensor(array: &Array) -> Result<Tensor> {
    let array = array.contiguous()?;
    eval(std::iter::once(&array))?;
    let dtype = array.dtype();
    let values = match dtype {
        Dtype::Int32 => Values::I32(array.as_slice::<i32>().to_vec()),
        // Parameters and gradients are f32; anything else that reaches here
        // is converted rather than reinterpreted.
        _ => Values::F32(array.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec()),
    };
    Ok(Tensor {
        dtype,
        shape: array.shape().to_vec(),
        values,
    })
}
