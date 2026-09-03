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
use std::path::Path;

use anyhow::{Context, Result};
use mlx_rs::transforms::eval;
use mlx_rs::Array;
use serde::Deserialize;

use crate::graph::{GraphConfig, ParameterSpec};
use crate::op::{qualified, Program, OBJECTIVE};
use crate::tensor::{dtype_named, Tensor};

/// Where a run's initial parameters come from.
///
/// Named rather than inferred from a type, because there will be more than
/// two: a GraphConfig already declares an initializer per parameter, so a
/// run that starts from those and a seed is a natural third.
pub enum Weights<'a> {
    /// Values, as JSON: {"params": {path => {"shape": [..], "data": [..]}}}.
    /// What a spike and a test use, and what the boundary carried when
    /// there was nothing else. Not for a real model: 130M parameters as
    /// JSON numbers is gigabytes of text on a path that is measured in
    /// milliseconds per step.
    Inline(&'a str),
    /// A safetensors file, read by the engine. Its keys are the qualified
    /// paths the graph declares ("student.linear.weight"), which is also
    /// how a checkpoint names them, so a run can start from what another
    /// run reached.
    File(&'a Path),
    /// One published checkpoint per model, by model name.
    ///
    /// A model published on its own does not know it will be called
    /// "student" here, so its tensors are named without that: ruri-v3's
    /// file holds `embeddings.tok_embeddings.weight`, not
    /// `student.embeddings...`. Naming a file per model is what lets a
    /// published checkpoint be imported with no renaming at all, and it is
    /// the shape a distillation wants anyway, having two of them.
    ///
    /// `fresh` names the parameters that are *expected* to be in no file
    /// and are built from their declared initializers instead: the
    /// classification head put on top of a published encoder exists in no
    /// checkpoint. Patterns, as freezing uses them
    /// ("student.head.*"), and everything they do not name must be in its
    /// file. Saying which are new is the caller's intent; a name missing
    /// because it was mistyped should still be an error, and it is.
    Pretrained {
        files: &'a BTreeMap<String, std::path::PathBuf>,
        fresh: &'a [String],
    },
}

/// One model as the plan holds it: its program, and where its parameters
/// sit in the run's flat parameter vector.
pub struct Model {
    pub name: String,
    /// Resolved when the plan opened, so a step reads values rather than
    /// JSON (`crate::op`).
    pub program: Program,
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
struct WeightsJson {
    params: BTreeMap<String, InitialTensor>,
}

/// The shape of a run: everything settled before the first step.
pub struct Plan {
    pub models: Vec<Model>,
    /// The program that reaches the loss. When a config declares none, the
    /// single model's own output is the loss.
    pub objective: Option<Program>,
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
    /// Every batch field the run reads, sorted and without repeats.
    /// Settled here rather than derived per step: this module's whole
    /// claim is that it does not move after `open`.
    input_names: Vec<String>,
    /// Every name a tap could ask for.
    node_names: Vec<String>,
    output_names: Vec<String>,
}

impl Plan {
    /// Reads a GraphConfig and its initial parameters, returning the plan
    /// and the parameters in the order the plan names them.
    pub fn open(graph_json: &str, weights: Weights<'_>) -> Result<(Self, Vec<Array>)> {
        Self::open_seeded(graph_json, weights, 0)
    }

    /// The same, with the seed the fresh parameters are drawn from.
    ///
    /// Explicit because a run that starts partly from initializers is
    /// reproducible only if that draw is (docs/plan.md 11.1).
    pub fn open_seeded(
        graph_json: &str,
        weights: Weights<'_>,
        seed: u64,
    ) -> Result<(Self, Vec<Array>)> {
        let config_digest = {
            use sha2::{Digest, Sha256};
            format!("{:x}", Sha256::digest(graph_json.as_bytes()))
        };
        let config: GraphConfig = serde_json::from_str(graph_json).context("parsing the graph")?;
        anyhow::ensure!(!config.models.is_empty(), "the graph has no models");
        let mut source = Source::read(weights)?;
        source.seed(seed)?;

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
                if trained && spec.trainable {
                    candidates.push(params.len() as i32);
                }
                params.push(source.parameter(&name, &path, spec)?);
                paths.push(path);
            }
            let slice = start..params.len();
            let program = Program::resolve(graph, slice.len(), &name)?;
            models.push(Model {
                name,
                program,
                slice,
            });
        }
        // A run that names no model differentiates nothing, and has said
        // so: a graph opened to be read (a loss over representations from
        // somewhere else) trains nothing by design. Naming models and
        // then having no parameters among them is still the mistake it
        // was, and is still refused here rather than at step one.
        anyhow::ensure!(
            !candidates.is_empty() || config.train.is_empty(),
            "nothing to train: no model in {:?} has trainable parameters",
            config.train
        );
        // One evaluation for the lot: a file's arrays are lazy until asked.
        eval(params.iter())?;

        let objective = config
            .objective
            .map(|graph| Program::resolve(graph, 0, OBJECTIVE))
            .transpose()?;
        let plan = Self {
            input_names: input_names(&models, objective.as_ref()),
            node_names: node_names(&models, objective.as_ref()),
            output_names: output_names(&models),
            models,
            objective,
            paths,
            candidates,
            config_digest,
            graph_json: graph_json.to_string(),
            semantics_version: config.semantics_version,
        };
        Ok((plan, params))
    }

    /// Every program a run evaluates, models first.
    pub fn programs(&self) -> impl Iterator<Item = &Program> {
        self.models
            .iter()
            .map(|m| &m.program)
            .chain(self.objective.iter())
    }

    /// Every batch field the run reads.
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Every name a tap could ask for.
    pub fn node_names(&self) -> &[String] {
        &self.node_names
    }

    /// Every model output a forward pass can be asked for, qualified by
    /// the model that declares it.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
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
        self.bind_programs(batch, self.programs())
    }

    /// The same, for the models alone.
    ///
    /// What a forward pass needs: it stops before the objective, so the
    /// objective's own fields (the labels, a teacher's scores) are not
    /// something to ask a caller for. A batch that carries them anyway is
    /// still accepted, since they are fields this run reads.
    pub fn bind_models(&self, batch: &BTreeMap<String, Tensor>) -> Result<BTreeMap<String, Array>> {
        self.bind_programs(batch, self.models.iter().map(|m| &m.program))
    }

    fn bind_programs<'a>(
        &self,
        batch: &BTreeMap<String, Tensor>,
        programs: impl Iterator<Item = &'a Program>,
    ) -> Result<BTreeMap<String, Array>> {
        for name in batch.keys() {
            anyhow::ensure!(
                self.input_names.iter().any(|w| w == name),
                "no input named {name:?} is read here (this run reads {:?})",
                self.input_names
            );
        }

        let mut fields = BTreeMap::new();
        for program in programs {
            for spec in &program.inputs {
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

/// Initial parameters, read and waiting to be matched to what the graph
/// declared.
enum Source {
    Inline(BTreeMap<String, InitialTensor>),
    /// Keyed by qualified path.
    File(std::collections::HashMap<String, Array>),
    /// Keyed by model, then by that model's own path, with the patterns
    /// naming what is expected to come from an initializer instead.
    Pretrained {
        files: BTreeMap<String, std::collections::HashMap<String, Array>>,
        fresh: Vec<String>,
        /// Split per parameter as they are built, so what one draws does
        /// not depend on how many came before it.
        key: Option<Array>,
    },
}

impl Source {
    fn read(weights: Weights<'_>) -> Result<Self> {
        Ok(match weights {
            Weights::Inline(json) => {
                let parsed: WeightsJson =
                    serde_json::from_str(json).context("parsing the initial parameters")?;
                Source::Inline(parsed.params)
            }
            Weights::File(path) => Source::File(read_safetensors(path)?),
            Weights::Pretrained { files, fresh } => Source::Pretrained {
                files: files
                    .iter()
                    .map(|(model, path)| {
                        Ok((
                            model.clone(),
                            read_safetensors(path)
                                .with_context(|| format!("model {model:?}"))?,
                        ))
                    })
                    .collect::<Result<_>>()?,
                fresh: fresh.to_vec(),
                key: None,
            },
        })
    }

    /// Where the fresh parameters are drawn from.
    fn seed(&mut self, seed: u64) -> Result<()> {
        if let Source::Pretrained { key, fresh, .. } = self {
            if !fresh.is_empty() {
                *key = Some(mlx_rs::random::key(seed)?);
            }
        }
        Ok(())
    }

    /// The one parameter a declaration asks for, checked against it.
    ///
    /// Shape has to match: a parameter of another shape is another
    /// parameter. Dtype does not, and is converted. Importing is starting
    /// somewhere, not resuming: a model published in bf16 is a perfectly
    /// good place to start an f32 run, and refusing it would be refusing
    /// the point of the exercise. `TrainState::restore` is the other case
    /// and refuses, because a resumed run has to be the same run.
    fn parameter(&self, model: &str, path: &str, spec: &ParameterSpec) -> Result<Array> {
        let declared = dtype_named(&spec.dtype)
            .with_context(|| format!("parameter {path:?}: unknown dtype {:?}", spec.dtype))?;
        let array = match self {
            Source::Inline(params) => {
                let t = params
                    .get(path)
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
                Array::from_slice(&t.data, &t.shape)
            }
            Source::File(arrays) => {
                let array = arrays.get(path).with_context(|| {
                    format!(
                        "the file has no parameter {path:?}. One file holds every \
                         model's parameters under their qualified paths, which is how \
                         a checkpoint writes them; a published model, which does not \
                         know what it will be called here, is named per model instead."
                    )
                })?;
                shaped(array, path, spec)?
            }
            Source::Pretrained { files, fresh, key } => {
                if let Some(pattern) = fresh.iter().find(|p| matches(p, path)) {
                    let key = key.as_ref().with_context(|| {
                        format!("parameter {path:?} matches {pattern:?} but no seed was set")
                    })?;
                    // Its own key, split from the run's by the path, so a
                    // parameter draws the same numbers wherever it sits in
                    // the declaration order.
                    let (_, mine) = mlx_rs::random::split(key, 2)?;
                    return crate::init::build(spec, &mine);
                }
                let arrays = files.get(model).with_context(|| {
                    format!("no file was given for model {model:?}")
                })?;
                // The model's own path, without the name this run gives it.
                let array = arrays.get(&spec.path).with_context(|| {
                    format!(
                        "model {model:?}'s file has no parameter {:?}. If it is meant \
                         to be new, name it: a head put on a published encoder is in \
                         no checkpoint, and saying so is what tells a typo from a \
                         fresh parameter.",
                        spec.path
                    )
                })?;
                shaped(array, path, spec)?
            }
        };
        Ok(if array.dtype() == declared {
            array
        } else {
            array.as_dtype(declared)?
        })
    }
}

/// Every batch field the run reads, once each and in order.
///
/// Free functions rather than methods, because they are how a `Plan` is
/// built rather than something one answers: what this type promises is
/// that everything in it was settled before the first step, and a plan
/// that had to be filled in after construction would not have been.
fn input_names(models: &[Model], objective: Option<&Program>) -> Vec<String> {
    let mut names: Vec<String> = models
        .iter()
        .map(|m| &m.program)
        .chain(objective)
        .flat_map(|p| p.inputs.iter().filter_map(|i| i.batch_field()))
        .map(str::to_string)
        .collect();
    names.sort();
    names.dedup();
    names
}

/// Every name a tap may ask for, qualified by the graph it is in.
///
/// Qualified for the same reason a parameter path is: a distillation runs
/// two models of one architecture, so "hidden" names a node in both of
/// them and "student.hidden" names one.
fn node_names(models: &[Model], objective: Option<&Program>) -> Vec<String> {
    models
        .iter()
        .flat_map(|m| m.program.node_names().map(|n| qualified(&m.name, n)))
        .chain(
            objective
                .into_iter()
                .flat_map(|o| o.node_names().map(|n| qualified(OBJECTIVE, n))),
        )
        .collect()
}

/// Every model output, qualified ("student.embedding").
///
/// The objective's outputs are not among them: its output is the loss,
/// which `evaluate` returns, and what this names is what a model
/// produces. Qualified for the reason a parameter path is: two models of
/// one architecture both declare "embedding".
fn output_names(models: &[Model]) -> Vec<String> {
    models
        .iter()
        .flat_map(|m| {
            m.program
                .outputs
                .iter()
                .map(|(name, _)| qualified(&m.name, name))
        })
        .collect()
}

fn read_safetensors(path: &Path) -> Result<std::collections::HashMap<String, Array>> {
    anyhow::ensure!(
        path.is_file(),
        "no parameters at {} (a safetensors file was expected)",
        path.display()
    );
    Array::load_safetensors(path).with_context(|| format!("reading {}", path.display()))
}

/// The array, if its shape is the one the graph declared.
fn shaped(array: &Array, path: &str, spec: &ParameterSpec) -> Result<Array> {
    anyhow::ensure!(
        array.shape() == spec.shape,
        "parameter {path:?}: the file holds shape {:?}, the graph declares {:?}",
        array.shape(),
        spec.shape
    );
    Ok(array.clone())
}

/// Whether a pattern names this path, in the same small language freezing
/// uses: an exact path, or a prefix ending in `*`.
fn matches(pattern: &str, path: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => path.starts_with(prefix),
        None => pattern == path,
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

    /// The common case in these tests: parameters as JSON.
    fn open(config: &str, weights: &str) -> Result<(Plan, Vec<Array>)> {
        Plan::open(config, Weights::Inline(weights))
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
    fn a_run_that_names_a_model_with_nothing_trainable_is_refused() {
        let (config, weights) = fixtures::teacher_and_student();
        let untrainable = config.replace(r#""trainable":true"#, r#""trainable":false"#);
        assert_ne!(untrainable, config, "the fixture's parameters moved");
        let e = refusal(&untrainable, &weights);
        assert!(e.contains("nothing to train"), "{e}");
    }

    /// The same emptiness, said on purpose. A graph opened to be read
    /// (a loss over representations computed elsewhere) trains nothing and
    /// has a use; what is refused is meaning to train and finding out at
    /// step one that there was nothing to.
    #[test]
    fn a_run_that_names_no_model_to_train_opens() {
        let (config, weights) = fixtures::teacher_and_student();
        let read_only = config.replace(r#""train":["student"]"#, r#""train":[]"#);
        assert_ne!(read_only, config, "the fixture's train list moved");

        let (plan, params) = open(&read_only, &weights).unwrap();

        assert!(plan.candidates.is_empty());
        assert_eq!(params.len(), 2, "the parameters are still there to read");
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
        assert_eq!(plan.node_names(), vec!["m.scaled"]);
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

    /// Writes what the graph declares into a safetensors file, the way a
    /// checkpoint does: keyed by qualified path.
    fn write_parameters(dir: &std::path::Path, named: &[(&str, Array)]) -> std::path::PathBuf {
        let path = dir.join("parameters.safetensors");
        Array::save_safetensors(named.iter().map(|(n, a)| (n, a)), None, &path).unwrap();
        path
    }

    #[test]
    fn parameters_can_come_from_a_file_named_the_way_the_graph_names_them() {
        let dir = tempfile::tempdir().unwrap();
        let file = write_parameters(
            dir.path(),
            &[("m.w", Array::from_slice(&[5.0f32, 6.0], &[2]))],
        );

        let (config, _) = fixtures::scaled_mean();
        let (plan, params) = Plan::open(&config, Weights::File(&file)).unwrap();
        assert_eq!(plan.paths, vec!["m.w"]);
        assert_eq!(crate::tensor::to_tensor(&params[0]).unwrap().shape, vec![2]);
    }

    #[test]
    fn a_file_missing_a_parameter_says_which_and_what_the_names_should_be() {
        let dir = tempfile::tempdir().unwrap();
        let file = write_parameters(
            dir.path(),
            &[("something.else", Array::from_slice(&[0.0f32, 0.0], &[2]))],
        );

        let (config, _) = fixtures::scaled_mean();
        let e = match Plan::open(&config, Weights::File(&file)) {
            Ok(_) => panic!("this should not have opened"),
            Err(e) => format!("{e:#}"),
        };
        assert!(e.contains("m.w"), "{e}");
        assert!(e.contains("qualified paths"), "{e}");
    }

    #[test]
    fn a_file_holding_another_shape_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let file = write_parameters(
            dir.path(),
            &[("m.w", Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]))],
        );

        let (config, _) = fixtures::scaled_mean();
        let e = match Plan::open(&config, Weights::File(&file)) {
            Ok(_) => panic!("this should not have opened"),
            Err(e) => format!("{e:#}"),
        };
        assert!(e.contains("the graph declares"), "{e}");
    }

    #[test]
    fn a_file_in_another_precision_is_converted_rather_than_refused() {
        // Importing is starting somewhere, not resuming: a model published
        // in bf16 is a fine place to start an f32 run.
        let dir = tempfile::tempdir().unwrap();
        let half = Array::from_slice(&[0.5f32, 2.0], &[2])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let file = write_parameters(dir.path(), &[("m.w", half)]);

        let (config, _) = fixtures::scaled_mean();
        let (_, params) = Plan::open(&config, Weights::File(&file)).unwrap();
        assert_eq!(params[0].dtype(), Dtype::Float32);
        let held = crate::tensor::to_tensor(&params[0]).unwrap();
        match held.values {
            Values::F32(v) => assert_eq!(v, vec![0.5, 2.0]),
            _ => panic!("a parameter is f32"),
        }
    }

    #[test]
    fn a_file_that_is_not_there_says_so_rather_than_reporting_a_missing_parameter() {
        let (config, _) = fixtures::scaled_mean();
        let missing = std::path::Path::new("/nowhere/parameters.safetensors");
        let e = match Plan::open(&config, Weights::File(missing)) {
            Ok(_) => panic!("this should not have opened"),
            Err(e) => format!("{e:#}"),
        };
        assert!(e.contains("no parameters at"), "{e}");
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
