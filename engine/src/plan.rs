//! What a run is going to do, decided once when it opens.
//!
//! A GraphConfig says which models exist, which are trained, and how the
//! objective reads them. None of that changes while a run goes, so it is
//! settled here: the models in order, where each one's parameters sit in
//! the flat vector, what the batch must supply, and which positions
//! autodiff differentiates.
//!
//! Separating it from the state (`TrainState`) and from the running
//! (`Executor`) is what lets each be read on its own.

use std::collections::BTreeMap;
use std::ops::Range;

use anyhow::{Context, Result};
use mlx_rs::Array;
use serde::Deserialize;

use crate::graph::{Graph, GraphConfig};
use crate::tensor::Tensor;

/// One model as the plan holds it: its graph, and where its parameters sit
/// in the run's flat parameter vector.
pub struct Model {
    pub name: String,
    pub graph: Graph,
    /// Range into the parameter vector, in the order the config declared.
    pub slice: Range<usize>,
}

/// One initial parameter as JSON carries it. Parameters are f32 (the
/// engine differentiates them), so this needs no dtype; a batch does, and
/// travels packed instead.
#[derive(Deserialize)]
struct InitialTensor {
    shape: Vec<i32>,
    data: Vec<f32>,
}

#[derive(Deserialize)]
struct Weights {
    params: BTreeMap<String, InitialTensor>,
}

/// The shape of a run: everything settled before the first step.
pub struct Plan {
    pub models: Vec<Model>,
    /// The graph that reaches the loss. When a config declares none, the
    /// single model's own output is the loss.
    pub objective: Option<Graph>,
    /// Qualified paths ("student.head.weight"), parallel to the parameter
    /// vector. This order is the contract checkpoints follow.
    pub paths: Vec<String>,
    /// Positions the config declared trainable: the bounds within which
    /// freezing moves.
    pub candidates: Vec<i32>,
    /// The digest of exactly the bytes the caller handed over, which is
    /// what the Ruby side computed and what a checkpoint records.
    pub config_digest: String,
}

impl Plan {
    /// Reads a GraphConfig and its initial parameters, returning the plan
    /// and the parameters in the order the plan names them.
    pub fn open(graph_json: &str, weights_json: &str) -> Result<(Self, Vec<Array>)> {
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
        let mut candidates = Vec::new();
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
                let expected: usize = t.shape.iter().map(|d| *d as usize).product();
                anyhow::ensure!(
                    t.data.len() == expected,
                    "parameter {path:?}: {} values for shape {:?}",
                    t.data.len(),
                    t.shape
                );
                if trained && spec.trainable {
                    candidates.push(params.len() as i32);
                }
                params.push(Array::from_slice(&t.data, &t.shape));
                paths.push(path);
            }
            let slice = start..params.len();
            models.push(Model { name, graph, slice });
        }
        anyhow::ensure!(
            !candidates.is_empty(),
            "nothing to train: no model in {:?} has trainable parameters",
            config.train
        );

        Ok((
            Self {
                models,
                objective: config.objective,
                paths,
                candidates,
                config_digest,
            },
            params,
        ))
    }

    /// Every graph a run evaluates, models first.
    pub fn graphs(&self) -> impl Iterator<Item = &Graph> {
        self.models.iter().map(|m| &m.graph).chain(self.objective.iter())
    }

    /// Every batch field the run reads.
    pub fn input_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .graphs()
            .flat_map(|g| g.inputs.iter().filter_map(|i| i.batch_field()))
            .map(str::to_string)
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// Every name a tap could ask for.
    pub fn node_names(&self) -> Vec<String> {
        self.graphs()
            .flat_map(|g| g.nodes.iter().filter_map(|n| n.name.clone()))
            .collect()
    }

    pub fn index_of(&self, path: &str) -> Option<usize> {
        self.paths.iter().position(|p| p == path)
    }

    /// The qualified paths at the given positions, in that order.
    pub fn paths_of(&self, positions: &[i32]) -> Vec<String> {
        positions
            .iter()
            .map(|&i| self.paths[i as usize].clone())
            .collect()
    }

    /// Every parameter a model declared trainable, whether or not it is
    /// frozen right now: the set freezing and unfreezing move within.
    pub fn candidate_paths(&self) -> Vec<String> {
        self.paths_of(&self.candidates)
    }

    /// Turns a batch into the fields the run reads, refusing anything no
    /// graph declared and anything a graph declared but the batch omits.
    /// A null dimension is symbolic and may differ from step to step
    /// (docs/plan.md 6.2).
    pub fn bind(&self, batch: &BTreeMap<String, Tensor>) -> Result<BTreeMap<String, Array>> {
        let wanted = self.input_names();
        for name in batch.keys() {
            anyhow::ensure!(
                wanted.iter().any(|w| w == name),
                "no input named {name:?} is read here (this run reads {wanted:?})"
            );
        }

        let mut fields = BTreeMap::new();
        for graph in self.graphs() {
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
                let declared = crate::tensor::dtype_named(&spec.dtype)
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
}

/// A freeze pattern: a path, or a prefix ending in `*`.
///
/// Deliberately small. A path names one parameter, `student.*` names a
/// model's, `student.layers.3.*` names a block's; anything more expressive
/// would be a query language nobody asked for.
pub struct Pattern {
    prefix: String,
    exact: bool,
    pub matched_any: bool,
}

impl Pattern {
    pub fn parse(pattern: &str) -> Result<Self> {
        anyhow::ensure!(!pattern.is_empty(), "a freeze pattern must not be empty");
        Ok(match pattern.strip_suffix('*') {
            Some(prefix) => Self {
                prefix: prefix.to_string(),
                exact: false,
                matched_any: false,
            },
            None => Self {
                prefix: pattern.to_string(),
                exact: true,
                matched_any: false,
            },
        })
    }

    pub fn matches(&mut self, path: &str) -> bool {
        let hit = if self.exact {
            path == self.prefix
        } else {
            path.starts_with(&self.prefix)
        };
        self.matched_any |= hit;
        hit
    }
}
