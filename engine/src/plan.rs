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
    /// Those bytes themselves. A checkpoint carries them, so that what it
    /// holds can be read without the run that wrote it: a digest names a
    /// description, it does not reconstruct one (docs/plan.md 11.2).
    pub graph_json: String,
    pub semantics_version: u32,
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
                graph_json: graph_json.to_string(),
                semantics_version: config.semantics_version,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;
    use crate::tensor::Values;
    use mlx_rs::Dtype;
    use serde_json::{json, Value};

    fn open(config: &str, weights: &str) -> Result<(Plan, Vec<Array>)> {
        Plan::open(config, weights)
    }

    /// The message from an open that should have failed. `unwrap_err`
    /// would want Debug on a Plan, which is a lot of Array to print.
    fn refusal(config: &str, weights: &str) -> String {
        match open(config, weights) {
            Ok(_) => panic!("this should not have opened"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn parameters_are_ordered_by_model_name_then_declaration() {
        let (config, weights) = fixtures::teacher_and_student();
        let (plan, params) = open(&config, &weights).unwrap();
        // "student" sorts before "teacher", which is the order a
        // checkpoint's parameter list has to follow.
        assert_eq!(plan.paths, vec!["student.scale", "teacher.scale"]);
        assert_eq!(params.len(), 2);
        assert_eq!(plan.models[0].slice, 0..1);
        assert_eq!(plan.models[1].slice, 1..2);
    }

    #[test]
    fn only_a_trained_models_parameters_are_candidates() {
        let (config, weights) = fixtures::teacher_and_student();
        let (plan, _) = open(&config, &weights).unwrap();
        assert_eq!(plan.candidates, vec![0]);
        assert_eq!(plan.candidate_paths(), vec!["student.scale"]);
    }

    #[test]
    fn the_digest_is_of_the_bytes_handed_over() {
        let (config, weights) = fixtures::scaled_mean();
        let (plan, _) = open(&config, &weights).unwrap();
        let same = open(&config, &weights).unwrap().0;
        assert_eq!(plan.config_digest, same.config_digest);
        assert_eq!(plan.config_digest.len(), 64);

        // Whitespace is part of the bytes, not of the meaning; the Ruby
        // side digests the canonical form, and the engine digests whatever
        // it was actually given. They agree because they see the same bytes.
        let spaced = format!("{config} ");
        let other = open(&spaced, &weights).unwrap().0;
        assert_ne!(plan.config_digest, other.config_digest);
    }

    #[test]
    fn a_missing_parameter_is_named() {
        let (config, _) = fixtures::scaled_mean();
        let e = refusal(&config, r#"{"params":{}}"#);
        assert!(e.contains("m.w"), "{e}");
    }

    #[test]
    fn a_parameter_of_the_wrong_shape_is_refused() {
        let (config, _) = fixtures::scaled_mean();
        let weights = json!({"params": {"m.w": {"shape": [3], "data": [1.0, 2.0, 3.0]}}});
        let e = refusal(&config, &weights.to_string());
        assert!(e.contains("is not the declared"), "{e}");
    }

    #[test]
    fn a_parameter_with_too_few_values_is_refused_rather_than_read_past() {
        let (config, _) = fixtures::scaled_mean();
        let weights = json!({"params": {"m.w": {"shape": [2], "data": [1.0]}}});
        let e = refusal(&config, &weights.to_string());
        assert!(e.contains("1 values for shape"), "{e}");
    }

    #[test]
    fn a_run_with_nothing_to_train_is_refused() {
        let (config, weights) = fixtures::teacher_and_student();
        let frozen = config.replace(r#""train":["student"]"#, r#""train":[]"#);
        assert_ne!(frozen, config, "the fixture's train list moved");
        let e = refusal(&frozen, &weights);
        assert!(e.contains("nothing to train"), "{e}");
    }

    #[test]
    fn a_config_with_no_models_is_refused() {
        let config = fixtures::config(json!({}), Value::Null, json!([]));
        let e = refusal(&config, r#"{"params":{}}"#);
        assert!(e.contains("no models"), "{e}");
    }

    #[test]
    fn input_names_are_sorted_and_deduplicated_across_graphs() {
        let (config, weights) = fixtures::teacher_and_student();
        let (plan, _) = open(&config, &weights).unwrap();
        // Both models read the same field; the objective reads neither.
        assert_eq!(plan.input_names(), vec!["x"]);
    }

    #[test]
    fn node_names_are_what_a_tap_may_ask_for() {
        let (config, weights) = fixtures::scaled_mean();
        let (plan, _) = open(&config, &weights).unwrap();
        assert_eq!(plan.node_names(), vec!["scaled"]);
    }

    fn tensor(shape: Vec<i32>, values: Values) -> Tensor {
        let dtype = match values {
            Values::F32(_) => Dtype::Float32,
            Values::I32(_) => Dtype::Int32,
        };
        Tensor {
            dtype,
            shape,
            values,
        }
    }

    fn plan_for_bind() -> Plan {
        let (config, weights) = fixtures::scaled_mean();
        open(&config, &weights).unwrap().0
    }

    #[test]
    fn a_symbolic_dimension_may_differ_from_batch_to_batch() {
        let plan = plan_for_bind();
        for rows in [1, 5] {
            let batch: BTreeMap<String, Tensor> = [(
                "x".to_string(),
                tensor(vec![rows, 2], Values::F32(vec![0.0; rows as usize * 2])),
            )]
            .into_iter()
            .collect();
            assert!(plan.bind(&batch).is_ok(), "{rows} rows should bind");
        }
    }

    #[test]
    fn a_fixed_dimension_may_not() {
        let plan = plan_for_bind();
        let batch = [(
            "x".to_string(),
            tensor(vec![2, 3], Values::F32(vec![0.0; 6])),
        )]
        .into_iter()
        .collect();
        let e = plan.bind(&batch).unwrap_err();
        assert!(e.to_string().contains("dimension 1 is 3"), "{e}");
    }

    #[test]
    fn a_batch_of_the_wrong_rank_is_refused() {
        let plan = plan_for_bind();
        let batch = [("x".to_string(), tensor(vec![4], Values::F32(vec![0.0; 4])))]
            .into_iter()
            .collect();
        let e = plan.bind(&batch).unwrap_err();
        assert!(e.to_string().contains("rank 1"), "{e}");
    }

    #[test]
    fn a_batch_of_the_wrong_dtype_is_refused() {
        let plan = plan_for_bind();
        let batch = [("x".to_string(), tensor(vec![2, 2], Values::I32(vec![0; 4])))]
            .into_iter()
            .collect();
        let e = plan.bind(&batch).unwrap_err();
        assert!(e.to_string().contains("declared f32"), "{e}");
    }

    #[test]
    fn a_batch_whose_values_do_not_fill_its_shape_is_refused() {
        let plan = plan_for_bind();
        let batch = [(
            "x".to_string(),
            tensor(vec![2, 2], Values::F32(vec![0.0; 3])),
        )]
        .into_iter()
        .collect();
        let e = plan.bind(&batch).unwrap_err();
        assert!(e.to_string().contains("3 values for shape"), "{e}");
    }

    #[test]
    fn a_field_no_graph_reads_is_refused_rather_than_ignored() {
        let plan = plan_for_bind();
        let batch = [
            ("x".to_string(), tensor(vec![1, 2], Values::F32(vec![0.0; 2]))),
            ("y".to_string(), tensor(vec![1], Values::F32(vec![0.0]))),
        ]
        .into_iter()
        .collect();
        let e = plan.bind(&batch).unwrap_err();
        assert!(e.to_string().contains("no input named \"y\""), "{e}");
    }

    #[test]
    fn a_field_a_graph_reads_but_the_batch_omits_is_refused() {
        let plan = plan_for_bind();
        let e = plan.bind(&BTreeMap::new()).unwrap_err();
        assert!(e.to_string().contains("missing input \"x\""), "{e}");
    }

    #[test]
    fn a_pattern_is_a_path_or_a_prefix() {
        let mut exact = Pattern::parse("a.b").unwrap();
        assert!(exact.matches("a.b"));
        assert!(!exact.matches("a.bc"));
        assert!(!exact.matches("a"));
        assert!(exact.matched_any);

        let mut prefix = Pattern::parse("a.*").unwrap();
        assert!(prefix.matches("a.b"));
        assert!(prefix.matches("a.b.c"));
        assert!(!prefix.matches("ab.c"));

        let mut nothing = Pattern::parse("z").unwrap();
        assert!(!nothing.matches("a"));
        assert!(!nothing.matched_any);

        assert!(Pattern::parse("").is_err());
    }
}
