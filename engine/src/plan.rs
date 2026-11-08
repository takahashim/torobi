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
use crate::mlxc::quantized::{dequantize, quantize, Mode, Quantization, Quantized};
use crate::mlxc::transforms::eval;
use crate::mlxc::{Array, Dtype};
use serde::Deserialize;

use crate::graph::{GraphConfig, InputSpec, ParameterSpec, Ref};
use crate::op::{qualified, Op, Program, OBJECTIVE};
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
    ///
    /// A weight the file holds quantized (a packed u32 `.weight` with a
    /// `.scales` beside it, as mlx-lm publishes them) and the graph
    /// declares as numbers is read as the numbers it stands for
    /// (docs/qlora.md section 3.4 (A')). A published 4-bit checkpoint
    /// quantizes its embedding too, and Torobi holds embeddings in bf16.
    ///
    /// The other way round as well: a weight the graph declares packed
    /// (three parameters read by one `quantized_matmul`) and the file
    /// holds as numbers is quantized as it is read, once per weight,
    /// to what the matmul says (section 3.4 (B)). That is how a bf16
    /// checkpoint opens as a quantized model.
    Pretrained {
        files: &'a BTreeMap<String, std::path::PathBuf>,
        fresh: &'a [String],
    },
}

/// How a packed u32 parameter is read back as the numbers it stands for:
/// the quantization the `quantized_matmul` that consumes it names, and
/// where that matmul's scales and biases sit in the parameter vector.
/// The three arrays themselves say none of this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Packed {
    pub quantization: Quantization,
    pub scales: usize,
    pub biases: Option<usize>,
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
    /// What each parameter is declared as, parallel to `paths`: what the
    /// parameters are made from when the plan is materialized.
    specs: Vec<ParameterSpec>,
    /// The packed quantized weights, by position, with what reads each
    /// one back as numbers (`TrainState::fetch`).
    pub packed: BTreeMap<usize, Packed>,
}

impl Plan {
    /// Reads a GraphConfig and its initial parameters, returning the plan
    /// and the parameters in the order the plan names them.
    ///
    /// Seedless, so only for the tests: a run states its seed, because a
    /// parameter built from its declaration is drawn from where the first
    /// step will draw from.
    #[cfg(test)]
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
        let plan = Self::resolve(graph_json)?;
        let mut source = Source::read(weights)?;
        source.seed(seed)?;
        // `materialize` consumes the source: a file's arrays are lazy
        // until asked, and the evaluation below is what reads them. By
        // then nothing but the parameters themselves may hold them, or a
        // weight the file holds in one form and the run in another (a
        // bf16 embedding published quantized, docs/qlora.md 3.4) would be
        // held twice until the source went.
        let params = source.materialize(&plan)?;
        // One evaluation for the lot.
        eval(params.iter())?;
        Ok((plan, params))
    }

    /// The description, resolved and checked, and nothing made: no file is
    /// read and MLX is not reached. What the parameters will be is decided
    /// here (their paths, which are trainable, what each is declared as);
    /// making them is `Source::materialize`'s.
    pub fn resolve(graph_json: &str) -> Result<Self> {
        let config_digest = crate::graph::digest(graph_json);
        let config: GraphConfig = serde_json::from_str(graph_json).context("parsing the graph")?;
        anyhow::ensure!(!config.models.is_empty(), "the graph has no models");

        let mut models = Vec::new();
        let mut paths = Vec::new();
        let mut specs = Vec::new();
        let mut candidates = Vec::new();
        let mut packed = BTreeMap::new();
        for (name, mut graph) in config.models {
            let trained = config.train.contains(&name);
            let start = paths.len();
            for spec in std::mem::take(&mut graph.parameters) {
                if trained && spec.trainable {
                    candidates.push(paths.len() as i32);
                }
                paths.push(format!("{name}.{}", spec.path));
                specs.push(spec);
            }
            let slice = start..paths.len();
            let program = Program::resolve(graph, slice.len(), &name)?;
            packed.extend(packed_weights(&program, start));
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

        let objective = config
            .objective
            .map(|graph| Program::resolve(graph, 0, OBJECTIVE))
            .transpose()?;
        Ok(Self {
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
            specs,
            packed,
        })
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

    /// The model and the output a qualified name refers to.
    ///
    /// **Found rather than parsed.** The names are made here, by joining
    /// with a dot, and an output's own name may hold one: a two-tower
    /// model declares "queries.embedding", so "m.queries.embedding"
    /// cannot be taken apart by looking at it. Splitting at the first dot
    /// is right only while no model name holds one, which is a promise
    /// the caller makes and this side should not depend on.
    pub fn output_named(&self, name: &str) -> Option<(&str, &str)> {
        self.models.iter().find_map(|model| {
            model
                .program
                .outputs
                .iter()
                .find(|(output, _)| qualified(&model.name, output) == name)
                .map(|(output, _)| (model.name.as_str(), output.as_str()))
        })
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
                let given = batch
                    .get(field)
                    .with_context(|| format!("the batch is missing input {field:?}"))?;
                fields.insert(field.to_string(), bind_one(field, spec, given)?);
            }
        }
        Ok(fields)
    }
}

/// One field, checked against what the graph declared and converted.
///
/// Five things have to hold, and each of them is a way a batch can be
/// wrong that nothing downstream would say plainly: the rank, every
/// concrete dimension, the dtype, and that the payload is as long as the
/// shape says. A null dimension is symbolic and may differ from step to
/// step (docs/plan.md 6.2), which is the one thing here that is allowed
/// to vary.
fn bind_one(field: &str, spec: &InputSpec, given: &Tensor) -> Result<Array> {
    anyhow::ensure!(
        given.shape.len() == spec.shape.len(),
        "input {field:?}: rank {} does not match the declared {}",
        given.shape.len(),
        spec.shape.len()
    );
    for (axis, (given, declared)) in given.shape.iter().zip(&spec.shape).enumerate() {
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
        given.dtype() == declared,
        "input {field:?}: given {:?}, declared {}",
        given.dtype(),
        spec.dtype
    );
    let expected: usize = given.shape.iter().map(|d| *d as usize).product();
    anyhow::ensure!(
        given.values.len() == expected,
        "input {field:?}: {} values for shape {:?}",
        given.values.len(),
        given.shape
    );
    given.to_array()
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
        fresh: Vec<Pattern>,
        /// Split per parameter as they are built, so what one draws does
        /// not depend on how many came before it.
        key: Option<Array>,
        /// Weights the file held as numbers and the graph holds packed,
        /// quantized once each and handed out to the three parameters
        /// that share them, by qualified module path ("m.layers.0.q_proj").
        /// Goes with the source, before anything is evaluated.
        quantized: std::collections::HashMap<String, Quantized>,
    },
}

/// Which of a packed weight's three parameters a declaration is.
#[derive(Clone, Copy)]
enum Piece {
    Weight,
    Scales,
    Biases,
}

impl Piece {
    /// The piece the parameter at `index` is, with where its packed
    /// weight sits, when it is one of a `quantized_matmul`'s three.
    fn of(plan: &Plan, index: usize) -> Option<(usize, Piece)> {
        if plan.packed.contains_key(&index) {
            return Some((index, Piece::Weight));
        }
        let path = &plan.paths[index];
        let (module, piece) = path
            .strip_suffix(".scales")
            .map(|module| (module, Piece::Scales))
            .or_else(|| path.strip_suffix(".biases").map(|module| (module, Piece::Biases)))?;
        let weight = plan.index_of(&format!("{module}.weight"))?;
        let packed = plan.packed.get(&weight)?;
        let mine = match piece {
            Piece::Scales => packed.scales == index,
            Piece::Biases => packed.biases == Some(index),
            Piece::Weight => unreachable!(),
        };
        mine.then_some((weight, piece))
    }

    fn take(self, quantized: &Quantized, path: &str) -> Result<Array> {
        Ok(match self {
            Piece::Weight => quantized.weight.clone(),
            Piece::Scales => quantized.scales.clone(),
            Piece::Biases => quantized
                .biases
                .clone()
                .with_context(|| format!("parameter {path:?}: this quantization has no biases"))?,
        })
    }
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
                fresh: fresh.iter().map(|p| Pattern::parse(p)).collect::<Result<_>>()?,
                key: None,
                quantized: std::collections::HashMap::new(),
            },
        })
    }

    /// Every parameter the plan declares, from this source, in the order
    /// the plan names them.
    ///
    /// Consumes the source. A file's arrays are handed over to the
    /// parameters as they are matched, and whatever the plan did not ask
    /// for goes with the source, before anything is evaluated.
    fn materialize(mut self, plan: &Plan) -> Result<Vec<Array>> {
        let mut params = Vec::with_capacity(plan.paths.len());
        for model in &plan.models {
            for i in model.slice.clone() {
                params.push(self.parameter(&model.name, plan, i)?);
            }
        }
        Ok(params)
    }

    /// Where the fresh parameters are drawn from.
    fn seed(&mut self, seed: u64) -> Result<()> {
        if let Source::Pretrained { key, fresh, .. } = self {
            if !fresh.is_empty() {
                *key = Some(crate::mlxc::random::key(seed)?);
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
    fn parameter(&mut self, model: &str, plan: &Plan, index: usize) -> Result<Array> {
        let (path, spec) = (plan.paths[index].as_str(), &plan.specs[index]);
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
                Array::from_slice(&t.data, &t.shape)?
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
            Source::Pretrained { files, fresh, key, quantized } => {
                if let Some(pattern) = fresh.iter().find(|p| p.matches(path)) {
                    let key = key.as_ref().with_context(|| {
                        format!("parameter {path:?} matches {pattern:?} but no seed was set")
                    })?;
                    // Its own key, split from the run's by the path, so a
                    // parameter draws the same numbers wherever it sits in
                    // the declaration order.
                    let (_, mine) = crate::mlxc::random::split(key, 2)?;
                    return crate::init::build(spec, &mine);
                }
                let arrays = files.get_mut(model).with_context(|| {
                    format!("no file was given for model {model:?}")
                })?;
                if let Some(piece) = quantized_on_read(arrays, quantized, plan, index)? {
                    return Ok(piece);
                }
                // The model's own path, without the name this run gives it.
                // Taken out of the map rather than read: from here on the
                // parameter is the only thing that holds it.
                let array = arrays.remove(&spec.path).with_context(|| {
                    format!(
                        "model {model:?}'s file has no parameter {:?}. If it is meant \
                         to be new, name it: a head put on a published encoder is in \
                         no checkpoint, and saying so is what tells a typo from a \
                         fresh parameter.",
                        spec.path
                    )
                })?;
                if let Some(numbers) = dequantized(arrays, path, spec, &array, declared)? {
                    return Ok(numbers);
                }
                shaped(&array, path, spec)?
            }
        };
        Ok(if array.dtype() == declared {
            array
        } else {
            array.as_dtype(declared)?
        })
    }
}

/// The packed weights a program reads, by their position in the parameter
/// vector (`start` is where the model's parameters begin), each with the
/// `quantized_matmul` that reads it. A weight read by two matmuls is read
/// the same way by both, since one set of scales fits one quantization.
fn packed_weights(program: &Program, start: usize) -> BTreeMap<usize, Packed> {
    let parameter_at = |reference: &Ref| match reference {
        Ref::Node(id) => match program.nodes[*id].op {
            Op::Parameter(index) => Some(start + index),
            _ => None,
        },
        Ref::Input(_) => None,
    };
    program
        .nodes
        .iter()
        .filter_map(|node| match node.op {
            Op::QuantizedMatmul(quantization) => {
                let weight = parameter_at(&node.inputs[1])?;
                let scales = parameter_at(&node.inputs[2])?;
                let biases = node.inputs.get(3).and_then(parameter_at);
                Some((weight, Packed { quantization, scales, biases }))
            }
            _ => None,
        })
        .collect()
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
/// A weight the graph holds packed and the file holds as numbers,
/// quantized as it is read (docs/qlora.md section 3.4 (B)): how a bf16
/// checkpoint opens as a quantized model.
///
/// The parameter at `index` is one of a `quantized_matmul`'s three
/// (`Piece::of`), the file has the module's `.weight` and no `.scales`,
/// so what it holds is numbers. They are quantized once, to what the
/// matmul says, and the three declarations are handed their pieces from
/// the one result. The file's array goes out of its map on the way, and
/// the result goes with the source, so after evaluation the packed form
/// is the only one in memory.
///
/// `None` when this is not such a parameter, or the file holds the
/// weight packed already ((A) applies: the shapes have to match as
/// declared). An error when the numbers are not the shape the packed
/// declaration unpacks to, saying both.
fn quantized_on_read(
    arrays: &mut std::collections::HashMap<String, Array>,
    quantized: &mut std::collections::HashMap<String, Quantized>,
    plan: &Plan,
    index: usize,
) -> Result<Option<Array>> {
    let Some((weight, piece)) = Piece::of(plan, index) else {
        return Ok(None);
    };
    let path = &plan.paths[index];
    let module = plan.paths[weight].strip_suffix(".weight").unwrap_or(&plan.paths[weight]);
    if let Some(done) = quantized.get(module) {
        return piece.take(done, path).map(Some);
    }
    let own = plan.specs[weight].path.strip_suffix(".weight").unwrap_or(&plan.specs[weight].path);
    let packed = plan.packed[&weight];
    let numbers = match arrays.get(&format!("{own}.weight")) {
        Some(array) if array.dtype() != Dtype::Uint32 && !arrays.contains_key(&format!("{own}.scales")) => {
            arrays.remove(&format!("{own}.weight")).expect("just seen")
        }
        _ => return Ok(None),
    };
    let declared = &plan.specs[weight].shape;
    let unpacked = match declared.as_slice() {
        [d_out, cols] => vec![*d_out, cols * 32 / packed.quantization.bits],
        _ => declared.clone(),
    };
    anyhow::ensure!(
        numbers.shape() == unpacked.as_slice(),
        "parameter {:?}: the file holds {own}.weight as {:?} {:?}, which is neither the packed \
         {declared:?} u32 the graph declares nor the {unpacked:?} it would be quantized from \
         ({} bits in groups of {})",
        plan.paths[weight],
        numbers.dtype(),
        numbers.shape(),
        packed.quantization.bits,
        packed.quantization.group_size
    );
    let done = quantize(&numbers, packed.quantization)
        .with_context(|| format!("parameter {:?}: quantizing what the file holds", plan.paths[weight]))?;
    let mine = piece.take(&done, path)?;
    quantized.insert(module.to_string(), done);
    Ok(Some(mine))
}

/// A weight the file holds quantized and the graph declares as numbers,
/// read as the numbers it stands for (docs/qlora.md section 3.4 (A')).
///
/// The file says how it was quantized by its shapes alone: a packed
/// `[d_out, d_in × bits / 32]` against the declared `[d_out, d_in]` gives
/// the bits, and `[d_out, d_in / group_size]` scales give the group. The
/// mode is affine, which is the one mlx-lm publishes and the one that
/// carries biases. `None` when the file holds the weight as numbers, so
/// the usual shape check applies; an error when it is packed in a way
/// the shapes cannot account for.
///
/// The scales and biases are taken out of the file's map as the weight
/// was: once the numbers are built, nothing else should hold them.
fn dequantized(
    arrays: &mut std::collections::HashMap<String, Array>,
    path: &str,
    spec: &ParameterSpec,
    weight: &Array,
    declared: Dtype,
) -> Result<Option<Array>> {
    if weight.dtype() != Dtype::Uint32 || declared == Dtype::Uint32 {
        return Ok(None);
    }
    let Some(module) = spec.path.strip_suffix(".weight") else {
        return Ok(None);
    };
    let scales = arrays.remove(&format!("{module}.scales")).with_context(|| {
        format!(
            "parameter {path:?}: the file holds a packed u32 weight {:?} and no {module}.scales \
             to read it with; the graph declares {:?} {}",
            weight.shape(),
            spec.shape,
            spec.dtype
        )
    })?;
    let biases = arrays.remove(&format!("{module}.biases"));
    let said = || {
        format!(
            "parameter {path:?}: the file holds a packed u32 weight {:?} with scales {:?}, which \
             is not {:?} {} quantized at any group size and bits this engine reads (groups of 32, \
             64 or 128 at 2 to 8 bits)",
            weight.shape(),
            scales.shape(),
            spec.shape,
            spec.dtype
        )
    };
    let (&[d_out, d_in], &[packed_rows, packed], &[scale_rows, groups]) =
        (spec.shape.as_slice(), weight.shape(), scales.shape())
    else {
        anyhow::bail!(said());
    };
    anyhow::ensure!(
        packed_rows == d_out && scale_rows == d_out && packed > 0 && groups > 0,
        said()
    );
    anyhow::ensure!((packed * 32) % d_in == 0 && d_in % groups == 0, said());
    let bits = packed * 32 / d_in;
    let group_size = d_in / groups;
    anyhow::ensure!(
        [2, 3, 4, 5, 6, 8].contains(&bits) && [32, 64, 128].contains(&group_size),
        said()
    );
    let quantization = Quantization { group_size, bits, mode: Mode::Affine };
    let numbers = dequantize(weight, &scales, biases.as_ref(), quantization, Some(declared))
        .with_context(|| format!("parameter {path:?}: dequantizing what the file holds"))?;
    Ok(Some(numbers))
}

fn shaped(array: &Array, path: &str, spec: &ParameterSpec) -> Result<Array> {
    anyhow::ensure!(
        array.shape() == spec.shape,
        "parameter {path:?}: the file holds shape {:?}, the graph declares {:?}",
        array.shape(),
        spec.shape
    );
    Ok(array.clone())
}

/// A parameter pattern: a path, or a prefix ending in `*`. The one small
/// language both freezing and a pretrained run's fresh parameters are named
/// in.
///
/// Deliberately small. A path names one parameter, `student.*` names a
/// model's, `student.layers.3.*` names a block's; anything more expressive
/// would be a query language nobody asked for.
pub struct Pattern {
    text: String,
    prefix: String,
    exact: bool,
}

impl Pattern {
    pub fn parse(pattern: &str) -> Result<Self> {
        anyhow::ensure!(!pattern.is_empty(), "a parameter pattern must not be empty");
        let (prefix, exact) = match pattern.strip_suffix('*') {
            Some(prefix) => (prefix, false),
            None => (pattern, true),
        };
        Ok(Self {
            text: pattern.to_string(),
            prefix: prefix.to_string(),
            exact,
        })
    }

    pub fn matches(&self, path: &str) -> bool {
        if self.exact {
            path == self.prefix
        } else {
            path.starts_with(&self.prefix)
        }
    }
}

impl std::fmt::Debug for Pattern {
    /// As it was written, which is how an error should quote it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;
    use crate::tensor::Values;
    use crate::mlxc::Dtype;
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
        Tensor { shape, values }
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
            &[("m.w", Array::from_slice(&[5.0f32, 6.0], &[2]).unwrap())],
        );

        let (config, _) = fixtures::scaled_mean();
        let (plan, params) = Plan::open(&config, Weights::File(&file)).unwrap();
        assert_eq!(plan.paths, vec!["m.w"]);
        assert_eq!(crate::tensor::to_tensor(&params[0]).unwrap().shape, vec![2]);
    }

    fn pretrained(dir: &std::path::Path, embedding_packed: bool) -> Result<(Plan, Vec<Array>)> {
        let files = BTreeMap::from([("m".to_string(), fixtures::quantized::file(dir, embedding_packed))]);
        Plan::open(&fixtures::quantized::config(), Weights::Pretrained { files: &files, fresh: &[] })
    }

    fn f32s(array: &Array) -> Vec<f32> {
        array.contiguous().unwrap().as_slice::<f32>().unwrap().to_vec()
    }

    /// A published 4-bit file holds its embedding packed, and the graph
    /// holds its embedding in numbers (docs/qlora.md 3.4 (A')): the file's
    /// three arrays become the one the graph declared, in its dtype and
    /// shape, within a step of the grid of what they were made from. The
    /// linear the graph does hold packed is read as the file has it.
    #[test]
    fn a_weight_the_file_holds_quantized_is_read_as_the_numbers_it_stands_for() {
        use fixtures::quantized::{weight, D_IN, D_OUT, EMB_ROWS};
        let dir = tempfile::tempdir().unwrap();
        let (plan, params) = pretrained(dir.path(), true).unwrap();

        let emb = plan.index_of("m.emb.weight").unwrap();
        assert_eq!(params[emb].dtype(), Dtype::Float32);
        assert_eq!(params[emb].shape(), &[EMB_ROWS as i32, D_IN as i32]);
        let (got, want) = (f32s(&params[emb]), weight(EMB_ROWS));
        let mut moved = false;
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 2.0 / 15.0, "{g} came back for {w}");
            moved |= (g - w).abs() > 1e-6;
        }
        assert!(moved, "the numbers went through 4 bits");

        let w = plan.index_of("m.proj.weight").unwrap();
        assert_eq!(params[w].dtype(), Dtype::Uint32);
        assert_eq!(params[w].shape(), &[D_OUT as i32, (D_IN * 4 / 32) as i32]);
        assert_eq!(
            plan.packed.get(&w),
            Some(&Packed {
                quantization: fixtures::quantized::QUANTIZATION,
                scales: plan.index_of("m.proj.scales").unwrap(),
                biases: Some(plan.index_of("m.proj.biases").unwrap()),
            })
        );
        assert_eq!(plan.packed.len(), 1, "the embedding is numbers now, not a packed weight");
    }

    /// The same model from a file that holds the embedding as numbers
    /// reads the same way: (A') applies only where the file is packed.
    #[test]
    fn a_weight_the_file_holds_as_numbers_is_read_as_before() {
        let dir = tempfile::tempdir().unwrap();
        let (plan, params) = pretrained(dir.path(), false).unwrap();
        let emb = plan.index_of("m.emb.weight").unwrap();
        assert_eq!(f32s(&params[emb]), fixtures::quantized::weight(fixtures::quantized::EMB_ROWS));
    }

    /// Packed with nothing to read it by, or packed in a way the declared
    /// shape cannot account for: refused with both shapes, before anything
    /// is dequantized.
    #[test]
    fn a_packed_weight_the_shapes_cannot_account_for_is_refused() {
        use fixtures::quantized::{file_with, D_IN, EMB_ROWS};
        let packed = |cols: i32| Array::zeros::<u32>(&[EMB_ROWS as i32, cols]).unwrap();
        let scales = |groups: i32| Array::zeros::<f32>(&[EMB_ROWS as i32, groups]).unwrap();
        let open = |emb: Vec<(&str, Array)>| {
            let dir = tempfile::tempdir().unwrap();
            let files = BTreeMap::from([("m".to_string(), file_with(dir.path(), emb))]);
            match Plan::open(&fixtures::quantized::config(), Weights::Pretrained { files: &files, fresh: &[] }) {
                Ok(_) => panic!("this file should not have opened"),
                Err(e) => format!("{e:#}"),
            }
        };

        let no_scales = open(vec![("emb.weight", packed((D_IN * 4 / 32) as i32))]);
        assert!(no_scales.contains("no emb.scales"), "{no_scales}");
        assert!(no_scales.contains("[4, 64] f32"), "{no_scales}");

        let odd_width = open(vec![("emb.weight", packed(5)), ("emb.scales", scales(2))]);
        assert!(odd_width.contains("is not [4, 64] f32 quantized"), "{odd_width}");
        assert!(odd_width.contains("[4, 5]"), "{odd_width}");

        let odd_groups = open(vec![("emb.weight", packed(8)), ("emb.scales", scales(3))]);
        assert!(odd_groups.contains("is not [4, 64] f32 quantized"), "{odd_groups}");
    }

    /// A bf16 checkpoint opens as a quantized model: the weight the graph
    /// holds packed is quantized as it is read, and what comes out is bit
    /// for bit what MLX's `quantize` gives on the same array, since it is
    /// the same call (docs/qlora.md 3.4 (B)). The three parameters that
    /// share it are handed the pieces of one quantization.
    #[test]
    fn a_weight_the_file_holds_as_numbers_is_quantized_as_it_is_read() {
        use crate::mlxc::quantized::quantize;
        use fixtures::quantized::{D_IN, D_OUT, QUANTIZATION};
        let dir = tempfile::tempdir().unwrap();
        let (file, written) = fixtures::quantized::file_bf16(dir.path());
        let files = BTreeMap::from([("m".to_string(), file)]);
        let (plan, params) =
            Plan::open(&fixtures::quantized::config(), Weights::Pretrained { files: &files, fresh: &[] }).unwrap();

        let want = quantize(&written, QUANTIZATION).unwrap();
        let at = |path: &str| &params[plan.index_of(path).unwrap()];
        let weight = at("m.proj.weight");
        assert_eq!(weight.dtype(), Dtype::Uint32);
        assert_eq!(weight.shape(), &[D_OUT as i32, (D_IN * 4 / 32) as i32]);
        assert_eq!(
            weight.contiguous().unwrap().as_slice::<u32>().unwrap(),
            want.weight.contiguous().unwrap().as_slice::<u32>().unwrap(),
            "the packed words are MLX's own"
        );
        for (path, wanted) in [("m.proj.scales", &want.scales), ("m.proj.biases", want.biases.as_ref().unwrap())] {
            let got = at(path);
            // Declared f32 here, so the bf16 scales the file's dtype gave
            // are widened; the numbers are the same.
            assert_eq!(got.dtype(), Dtype::Float32, "{path}");
            assert_eq!(f32s(got), f32s(&wanted.as_dtype(Dtype::Float32).unwrap()), "{path}");
        }
        // `head` and `emb` are read as before.
        assert_eq!(f32s(at("m.head.weight")), vec![1.0; D_OUT]);
    }

    /// What quantizing on read is for: once a bf16 file is open as a
    /// quantized model, what is held is the packed form, not the numbers
    /// it came from. The file's array leaves its map as it is read and
    /// the source goes before evaluation, so the bf16 copy is freed with
    /// the lazy graph that consumed it. Measured as what MLX holds after
    /// opening against what the bf16 numbers alone would take: a weight
    /// of a megabyte, so the allocator's rounding is noise against it.
    #[test]
    fn opening_a_bf16_file_as_a_quantized_model_does_not_keep_the_numbers() {
        use crate::mlxc::memory;
        let (d_out, d_in) = (512usize, 1024usize);
        let dir = tempfile::tempdir().unwrap();
        let (file, written) = fixtures::quantized::file_bf16_sized(dir.path(), d_out, d_in);
        drop(written);
        let files = BTreeMap::from([("m".to_string(), file)]);

        let before = memory::active().unwrap();
        let (plan, params) =
            Plan::open(&fixtures::quantized::config_sized(d_out, d_in), Weights::Pretrained { files: &files, fresh: &[] })
                .unwrap();
        let held = memory::active().unwrap().saturating_sub(before);

        let numbers = d_out * d_in * 2;
        let packed = d_out * d_in / 2 + 2 * d_out * (d_in / 32) * 4;
        assert!(
            held < numbers,
            "{held} bytes held after opening; the bf16 numbers alone are {numbers}, the packed form {packed}"
        );
        assert_eq!(params[plan.index_of("m.proj.weight").unwrap()].dtype(), Dtype::Uint32);
    }

    /// Numbers of a shape the packed declaration does not unpack to are
    /// a weight quantized some other way, or another weight: refused,
    /// with the declared shape, the shape it would be quantized from,
    /// and what the file holds.
    #[test]
    fn numbers_that_do_not_unpack_to_the_declared_shape_are_refused() {
        let open = |proj: Array| {
            let dir = tempfile::tempdir().unwrap();
            let files = BTreeMap::from([("m".to_string(), fixtures::quantized::file_with_proj(dir.path(), proj))]);
            match Plan::open(&fixtures::quantized::config(), Weights::Pretrained { files: &files, fresh: &[] }) {
                Ok(_) => panic!("this file should not have opened"),
                Err(e) => format!("{e:#}"),
            }
        };
        let narrow = open(Array::zeros::<f32>(&[8, 60]).unwrap());
        assert!(narrow.contains("neither the packed [8, 8] u32"), "{narrow}");
        assert!(narrow.contains("nor the [8, 64] it would be quantized from"), "{narrow}");
        assert!(narrow.contains("Float32 [8, 60]"), "{narrow}");

        // The packed shape in a float dtype is not a packed weight.
        let float_words = open(Array::zeros::<f32>(&[8, 8]).unwrap());
        assert!(float_words.contains("Float32 [8, 8]"), "{float_words}");
    }

    #[test]
    fn a_file_missing_a_parameter_says_which_and_what_the_names_should_be() {
        let dir = tempfile::tempdir().unwrap();
        let file = write_parameters(
            dir.path(),
            &[("something.else", Array::from_slice(&[0.0f32, 0.0], &[2]).unwrap())],
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
            &[("m.w", Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap())],
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
        let half = Array::from_slice(&[0.5f32, 2.0], &[2]).unwrap()
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
        let exact = Pattern::parse("a.b").unwrap();
        assert!(exact.matches("a.b"));
        assert!(!exact.matches("a.bc"));
        assert!(!exact.matches("a"));

        let prefix = Pattern::parse("a.*").unwrap();
        assert!(prefix.matches("a.b"));
        assert!(prefix.matches("a.b.c"));
        assert!(!prefix.matches("ab.c"));
        assert_eq!(format!("{prefix:?}"), r#""a.*""#);

        assert!(Pattern::parse("").is_err());
    }

    /// A plan is the description alone: resolved and checked with no
    /// weights to read and nothing made, which is what lets a graph be
    /// checked before any file or device is involved.
    #[test]
    fn a_description_resolves_without_its_weights() {
        let (config, _) = fixtures::scaled_mean();
        let plan = Plan::resolve(&config).unwrap();
        assert_eq!(plan.paths, vec!["m.w".to_string()]);
        assert_eq!(plan.candidates, vec![0]);
        assert_eq!(plan.specs[0].shape, vec![2]);
        assert_eq!(plan.config_digest, crate::graph::digest(&config));
    }

}
