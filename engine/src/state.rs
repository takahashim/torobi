//! What a run accumulates: parameters, optimizer slots, the RNG, and the
//! counters. Everything that a checkpoint has to carry, and nothing that a
//! GraphConfig already says.
//!
//! The plan (`crate::plan`) is fixed when a run opens; this is the part
//! that moves. It moves in one place, [`TrainState::advance`], which is
//! written as a transaction: the next state is built and evaluated in full
//! before any of it is assigned, so a step that fails leaves the run
//! exactly as it was (docs/plan.md section 5A.4).

use std::path::Path;

use anyhow::{Context, Result};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::checkpoint;
use crate::optimizer::{Config as OptimizerConfig, Optimizer};
use crate::plan::{Pattern, Plan};
use crate::tensor::{to_tensor, Tensor};

/// A checkpoint that has been read and found to belong to this run.
///
/// Exists only after every check has passed, which is what makes
/// [`TrainState::restore`] a commit rather than a sequence of hopes.
struct Restored {
    params: Vec<Array>,
    m: Vec<Array>,
    v: Vec<Array>,
    rng: Array,
    seed: u64,
    step: usize,
    optimizer_steps: u64,
    /// What the caller recorded alongside, handed back unchanged.
    run: String,
}

/// One pass's view of the state: what the executor reads, and nothing
/// more.
///
/// These three used to be public fields, which meant a caller assembled
/// them itself and could put them out of step with each other. The
/// invariant they are under (the optimizer's slots are parallel to
/// `argnums`, and `params` is indexed by it) is this module's to keep.
pub struct Pass<'a> {
    pub params: &'a [Array],
    pub argnums: &'a [i32],
    pub rng: &'a Array,
}

pub struct TrainState {
    /// Every parameter of every model, model by model in name order. This
    /// order is `Plan::paths`, and the contract the Ruby side follows.
    params: Vec<Array>,
    /// Positions in `params` that autodiff differentiates now. A subset of
    /// `Plan::candidates`; moves when the window freezes or unfreezes.
    argnums: Vec<i32>,
    /// The update rule and its slots. Half of what a checkpoint restores.
    optimizer: Optimizer,
    /// The RNG, held as state rather than left to a global. Every step
    /// splits it, so the sequence of draws is a function of the seed and
    /// the step count, and a resumed run draws what a continuous one would
    /// (docs/plan.md section 11.1).
    rng: Array,
    seed: u64,
    step: usize,
    last_loss: f32,
    /// Gradients waiting for a step, and the losses they came from.
    ///
    /// A batch that does not fit is trained as several that do, summing
    /// their gradients and updating once. Held here rather than by the
    /// caller because they are device arrays parallel to `argnums`, which
    /// is this module's invariant to keep: a caller holding them could
    /// freeze a parameter between two of them and hand back a vector that
    /// no longer lines up.
    pending: Option<Pending>,
}

/// What has been accumulated since the last step.
struct Pending {
    /// One per differentiated parameter, in `argnums` order.
    grads: Vec<Array>,
    /// The losses that produced them, so a step can report their mean.
    losses: Vec<f32>,
}

impl TrainState {
    /// The state a plan starts in: the given parameters, everything the
    /// plan declared trainable differentiated, and fresh optimizer slots.
    pub fn new(
        plan: &Plan,
        params: Vec<Array>,
        optimizer: OptimizerConfig,
        seed: u64,
    ) -> Result<Self> {
        let argnums = plan.candidates.clone();
        let optimizer = Optimizer::new(optimizer, &params, &argnums)?;
        Ok(Self {
            params,
            argnums,
            optimizer,
            rng: mlx_rs::random::key(seed)?,
            seed,
            step: 0,
            last_loss: f32::NAN,
            pending: None,
        })
    }

    /// What a pass reads. The only way out of here for the three of them.
    pub fn pass(&self) -> Pass<'_> {
        Pass {
            params: &self.params,
            argnums: &self.argnums,
            rng: &self.rng,
        }
    }

    /// Which parameters autodiff differentiates now, as positions.
    pub fn argnums(&self) -> &[i32] {
        &self.argnums
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

    /// What update rule this run uses, as data.
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

    /// Freezes or unfreezes parameters whose path matches `pattern`, and
    /// returns those that moved.
    ///
    /// Not a scalar knob (docs/plan.md section 5A.1): changing it changes
    /// what autodiff differentiates, so the optimizer's slots have to
    /// follow (kept for what stays, dropped for what freezes, started at
    /// zero for what thaws). Which is why this lives here rather than in a
    /// setter.
    pub fn set_frozen(&mut self, plan: &Plan, pattern: &str, frozen: bool) -> Result<Vec<String>> {
        let mut matcher = Pattern::parse(pattern)?;
        let mut wanted: Vec<i32> = Vec::new();
        let mut moved = Vec::new();
        for &i in &plan.candidates {
            let path = &plan.paths[i as usize];
            let matches = matcher.matches(path);
            let currently = self.argnums.contains(&i);
            let next = if matches { !frozen } else { currently };
            if next != currently {
                moved.push(path.clone());
            }
            if next {
                wanted.push(i);
            }
        }
        anyhow::ensure!(
            !moved.is_empty() || matcher.matched_any,
            "no parameter matches {pattern:?} (this run has {:?})",
            plan.candidate_paths()
        );
        if moved.is_empty() {
            return Ok(moved);
        }
        // The waiting gradients are one per differentiated parameter, in
        // this order. Moving the set would leave them meaning something
        // else, and a caller that froze in the middle of a batch has lost
        // track of which half is which.
        anyhow::ensure!(
            self.pending.is_none(),
            "{} parts are accumulated and freezing changes what a gradient is for. \
             Apply them or discard them first",
            self.accumulated()
        );
        anyhow::ensure!(
            !wanted.is_empty(),
            "freezing {pattern:?} would leave nothing to train"
        );

        self.optimizer.refit(&self.argnums, &wanted, &self.params)?;
        self.argnums = wanted;
        Ok(moved)
    }

    /// Writes one parameter, by qualified path, from a copy. The window's
    /// B capability (docs/plan.md section 8.3).
    pub fn put(&mut self, plan: &Plan, path: &str, tensor: &Tensor) -> Result<()> {
        let index = plan
            .index_of(path)
            .with_context(|| format!("no parameter named {path:?}"))?;
        let array = tensor.to_array();
        anyhow::ensure!(
            array.shape() == self.params[index].shape(),
            "parameter {path:?}: given shape {:?} is not {:?}",
            array.shape(),
            self.params[index].shape()
        );
        anyhow::ensure!(
            array.dtype() == self.params[index].dtype(),
            "parameter {path:?}: given {:?}, holds {:?}",
            array.dtype(),
            self.params[index].dtype()
        );
        eval(std::iter::once(&array))?;
        self.params[index] = array;
        Ok(())
    }

    /// A copy of one parameter, by qualified path. Copies, not handles:
    /// nothing that lives on the device escapes this crate.
    pub fn fetch(&self, plan: &Plan, path: &str) -> Result<Tensor> {
        let index = plan
            .index_of(path)
            .with_context(|| format!("no parameter named {path:?}"))?;
        to_tensor(&self.params[index])
    }

    /// One optimizer step from a loss and its gradients, as a transaction:
    /// the next parameters, slots and key are built and evaluated first,
    /// and only then does this state become them. A step that fails leaves
    /// everything as it was, which is what StepError promises.
    ///
    /// **A step whose loss is not finite is not taken.** Its gradients are
    /// not finite either, so taking it would put NaN into every parameter
    /// and nothing after it could recover; the only way back would be a
    /// checkpoint, which is why the plan listed a rollback knob. Not going
    /// there is cheaper than coming back, and free: the loss is already
    /// evaluated by the time this is called.
    ///
    /// The counters still move. A step was attempted, its batch was
    /// consumed, and the RNG was drawn from during the forward, so the draw
    /// belongs to the step count exactly as it would have; a resumed run
    /// has to see the same sequence. What does not move is the parameters
    /// and the optimizer's slots, so a policy that lowers the rate and
    /// carries on has something clean to carry on from.
    pub fn advance(&mut self, loss: &Array, grads: &[Array]) -> Result<f32> {
        let value = loss.item::<f32>();
        let (rng, _) = mlx_rs::random::split(&self.rng, 2)?;

        if !value.is_finite() {
            eval(std::iter::once(&rng))?;
            self.rng = rng;
            self.last_loss = value;
            self.step += 1;
            return Ok(value);
        }

        let mut params = self.params.clone();
        let next_optimizer = self.optimizer.next(&mut params, &self.argnums, grads)?;

        let (m, v) = next_optimizer.slots();
        eval(
            params
                .iter()
                .chain(m.iter())
                .chain(v.iter())
                .chain(std::iter::once(&rng)),
        )?;

        self.params = params;
        self.optimizer = next_optimizer;
        self.rng = rng;
        self.last_loss = value;
        self.step += 1;
        Ok(value)
    }

    /// Adds one batch's gradients to what is waiting, and reports its
    /// loss. No step is taken and no counter moves.
    ///
    /// What this is for: a batch too large to hold is trained as several
    /// that fit. The gradients of a sum are the sum of the gradients, so
    /// accumulating and then stepping reaches where one step over the
    /// whole batch would have. Whether the parts are a mean or a sum of
    /// each other is the caller's arithmetic (a loss that is a mean over
    /// its rows wants the parts weighted), and this does not guess.
    ///
    /// A part whose loss is not finite is not added, for the reason
    /// `advance` does not take such a step: its gradients are not finite
    /// either, and one of them would poison the sum. The loss is still
    /// reported, so a caller sees it happen.
    ///
    /// The RNG does not move. A draw belongs to a step, and this is a
    /// fraction of one; the step that applies these makes the draw.
    pub fn accumulate(&mut self, loss: &Array, grads: &[Array]) -> Result<f32> {
        let value = loss.item::<f32>();
        anyhow::ensure!(
            grads.len() == self.argnums.len(),
            "these gradients are for {} parameters and {} are differentiated",
            grads.len(),
            self.argnums.len()
        );
        if !value.is_finite() {
            return Ok(value);
        }

        let pending = match self.pending.take() {
            None => Pending {
                grads: grads.to_vec(),
                losses: vec![value],
            },
            Some(mut held) => {
                held.grads = held
                    .grads
                    .iter()
                    .zip(grads)
                    .map(|(held, new)| held.add(new))
                    .collect::<std::result::Result<_, _>>()?;
                held.losses.push(value);
                held
            }
        };
        // Evaluated here rather than left lazy: a run that accumulates ten
        // parts should hold ten sums, not a graph ten deep.
        eval(pending.grads.iter())?;
        self.pending = Some(pending);
        Ok(value)
    }

    /// Takes the step the accumulated gradients ask for, and reports the
    /// mean of the losses they came from.
    ///
    /// Refuses when nothing is waiting: a step from no gradients is not a
    /// step of zero, it is a caller that has lost track of where it is.
    pub fn apply(&mut self) -> Result<f32> {
        let Some(pending) = self.pending.take() else {
            anyhow::bail!("nothing has been accumulated, so there is no step to take");
        };
        let mean = pending.losses.iter().sum::<f32>() / pending.losses.len() as f32;
        let loss = Array::from_f32(mean);
        self.advance(&loss, &pending.grads)
    }

    /// How many parts are waiting for a step.
    pub fn accumulated(&self) -> usize {
        self.pending.as_ref().map_or(0, |p| p.losses.len())
    }

    /// Throws away what was accumulated, for a caller abandoning a batch
    /// part way through. Returns how many parts went.
    pub fn discard(&mut self) -> usize {
        self.pending.take().map_or(0, |p| p.losses.len())
    }

    /// Writes the run's state: parameters, optimizer slots, counters, the
    /// description they belong to, and whatever the caller wants recorded
    /// alongside. Atomic (docs/plan.md section 11.2).
    ///
    /// `run` is the caller's half of the record. Epoch, batch position,
    /// sampler state and dataset identity are not the engine's to know
    /// (it is handed a batch, it does not fetch one), so they travel as
    /// JSON that is written verbatim and read back verbatim.
    pub fn save(&self, plan: &Plan, dir: &str, run: &str) -> Result<String> {
        // A checkpoint is a whole run record, and gradients waiting for a
        // step are not in it: restoring one would silently be a run that
        // had dropped half a batch. Refusing says so while the caller can
        // still choose.
        anyhow::ensure!(
            self.pending.is_none(),
            "{} parts are accumulated and a checkpoint does not hold them. \
             Apply them or discard them first",
            self.accumulated()
        );
        let run: serde_json::Value = if run.trim().is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(run).context("the run metadata is not JSON")?
        };
        let (m, v) = self.optimizer.slots();
        let state = checkpoint::State {
            config_digest: &plan.config_digest,
            graph_json: &plan.graph_json,
            semantics_version: plan.semantics_version,
            step: self.step,
            optimizer: self.optimizer.config(),
            optimizer_steps: self.optimizer.steps_taken(),
            parameters: plan.paths.iter().cloned().zip(self.params.iter()).collect(),
            argnums: &self.argnums,
            slots: (m, v),
            rng: &self.rng,
            seed: self.seed,
            run,
        };
        Ok(checkpoint::write(dir, state)?.display().to_string())
    }

    /// Writes one model's parameters to `dir` as an HF-compatible fp32
    /// safetensors checkpoint. The GraphConfig model name is stripped from
    /// each qualified path, so a model declared with `encoder_prefix:`
    /// keeps that prefix and the file's keys match the published layout.
    ///
    /// Returns the (old, new) path pairs so the caller can write metadata
    /// that names what was saved.
    pub fn export_model(
        &self,
        plan: &Plan,
        model: &str,
        dir: &str,
    ) -> Result<Vec<(String, String)>> {
        let model = plan
            .models
            .iter()
            .find(|m| m.name == model)
            .with_context(|| {
                format!(
                    "no model named {model:?} (this run has {:?})",
                    plan.models.iter().map(|m| &m.name).collect::<Vec<_>>()
                )
            })?;
        let dir = Path::new(dir);
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating {}", dir.display()))?;

        let prefix = format!("{}.", model.name);
        let mut renamed = Vec::new();
        let mut owned: Vec<(String, Array)> = Vec::new();
        for (path, array) in plan.paths[model.slice.clone()]
            .iter()
            .zip(&self.params[model.slice.clone()])
        {
            let new_path = path.strip_prefix(&prefix).unwrap_or(path).to_string();
            renamed.push((path.clone(), new_path.clone()));
            let array = if array.dtype() == Dtype::Float32 {
                array.clone()
            } else {
                array.as_dtype(Dtype::Float32)?
            };
            owned.push((new_path, array));
        }
        eval(owned.iter().map(|(_, a)| a))?;
        Array::save_safetensors(
            owned.iter().map(|(path, array)| (path.as_str(), array)),
            None,
            dir.join("model.safetensors"),
        )
        .context("writing model.safetensors")?;
        Ok(renamed)
    }

    /// Restores state written by [`TrainState::save`], refusing anything
    /// that does not belong to this run: another description, another
    /// optimizer, a parameter of another shape, a missing slot.
    ///
    /// Returns what the caller recorded in `run`, so that whoever owns the
    /// data can put its sampler back where the checkpoint left it.
    ///
    /// Reading and accepting is [`TrainState::accept`]; this is the commit.
    /// They are separate so that "nothing moves until everything is
    /// checked" is a fact about the types rather than a claim in a comment:
    /// a [`Restored`] only exists if every check passed.
    pub fn restore(&mut self, plan: &Plan, dir: &str) -> Result<String> {
        let accepted = self.accept(plan, dir)?;

        // Everything is here and consistent; make it real before touching
        // this state, so a failure below cannot leave it half restored.
        eval(
            accepted
                .params
                .iter()
                .chain(accepted.m.iter())
                .chain(accepted.v.iter())
                .chain(std::iter::once(&accepted.rng)),
        )?;

        self.params = accepted.params;
        self.optimizer
            .restore(accepted.m, accepted.v, accepted.optimizer_steps);
        self.rng = accepted.rng;
        self.seed = accepted.seed;
        self.step = accepted.step;
        self.last_loss = f32::NAN;
        // Gradients for parameters that are no longer there. Going back is
        // the answer to a batch that went wrong, so they are dropped
        // rather than refused.
        self.pending = None;
        Ok(accepted.run)
    }

    /// Reads a checkpoint and checks all of it against this run. Touches
    /// nothing.
    fn accept(&self, plan: &Plan, dir: &str) -> Result<Restored> {
        let loaded = checkpoint::read(dir)?;
        let manifest = &loaded.manifest;
        anyhow::ensure!(
            manifest.config_digest == plan.config_digest,
            "this checkpoint belongs to another graph (digest {}, not {})",
            &manifest.config_digest[..12.min(manifest.config_digest.len())],
            &plan.config_digest[..12]
        );
        anyhow::ensure!(
            &manifest.optimizer == self.optimizer.config(),
            "this checkpoint was written by a different optimizer ({:?})",
            manifest.optimizer
        );
        anyhow::ensure!(
            manifest.parameters.len() == plan.paths.len(),
            "this checkpoint has {} parameters, this session has {}",
            manifest.parameters.len(),
            plan.paths.len()
        );

        let mut params = Vec::with_capacity(self.params.len());
        for (i, path) in plan.paths.iter().enumerate() {
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

        let (m, v) = self.accept_slots(plan, &loaded)?;
        Ok(Restored {
            params,
            m,
            v,
            rng: loaded.rng.context("the checkpoint has no RNG state")?,
            seed: manifest.seed,
            step: manifest.step,
            optimizer_steps: manifest.optimizer_steps,
            run: serde_json::to_string(&manifest.run)?,
        })
    }

    /// The optimizer's slots, all of them or none.
    ///
    /// An AdamW session restored without moments used to index past the
    /// end of an empty vector on its next step, which is why this refuses
    /// rather than filling in.
    fn accept_slots(
        &self,
        plan: &Plan,
        loaded: &checkpoint::Loaded,
    ) -> Result<(Vec<Array>, Vec<Array>)> {
        if !self.optimizer.wants_slots() {
            anyhow::ensure!(
                loaded.slots.is_empty(),
                "this checkpoint carries optimizer state, and {} has none",
                self.optimizer.config().name()
            );
            return Ok((Vec::new(), Vec::new()));
        }

        anyhow::ensure!(
            !loaded.slots.is_empty(),
            "this checkpoint has no optimizer state, and {} needs it",
            self.optimizer.config().name()
        );
        let (mut m, mut v) = (Vec::new(), Vec::new());
        for &i in &self.argnums {
            let path = &plan.paths[i as usize];
            let (slot_m, slot_v) = loaded
                .slots
                .get(path)
                .with_context(|| format!("the checkpoint has no optimizer state for {path:?}"))?;
            anyhow::ensure!(
                slot_m.shape() == self.params[i as usize].shape()
                    && slot_v.shape() == self.params[i as usize].shape(),
                "optimizer state for {path:?} has the wrong shape"
            );
            m.push(slot_m.clone());
            v.push(slot_v.clone());
        }
        Ok((m, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::{self, Taps};
    use crate::fixtures;
    use crate::plan::Weights;
    use crate::tensor::{Tensor, Values};

    fn open(which: (String, String), optimizer: OptimizerConfig) -> (Plan, TrainState) {
        let (config, weights) = which;
        let (plan, params) = Plan::open(&config, Weights::Inline(&weights)).unwrap();
        let state = TrainState::new(&plan, params, optimizer, 0).unwrap();
        (plan, state)
    }

    fn sgd(lr: f32) -> OptimizerConfig {
        OptimizerConfig::Sgd { lr }
    }

    fn adamw() -> OptimizerConfig {
        OptimizerConfig::AdamW {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }

    fn values(t: &Tensor) -> Vec<f32> {
        match &t.values {
            Values::F32(v) => v.clone(),
            Values::I32(v) => v.iter().map(|&i| i as f32).collect(),
        }
    }

    /// One step, the way a session takes it.
    fn step(plan: &Plan, state: &mut TrainState, rows: &[f32]) -> f32 {
        let batch = fixtures::batch_x(rows);
        let fields = plan.bind(&batch).unwrap();
        let (loss, grads, _) =
            executor::differentiate(plan, state.pass(), &fields, &Taps::new()).unwrap();
        state.advance(&loss, &grads).unwrap()
    }

    #[test]
    fn a_new_state_differentiates_everything_the_plan_allows() {
        let (plan, state) = open(fixtures::teacher_and_student(), sgd(0.1));
        assert_eq!(state.argnums(), plan.candidates);
        assert_eq!(state.step(), 0);
        assert!(state.loss().is_nan());
        assert_eq!(state.seed(), 0);
    }

    #[test]
    fn a_step_moves_only_what_is_differentiated() {
        let (plan, mut state) = open(fixtures::teacher_and_student(), sgd(0.1));
        step(&plan, &mut state, &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(state.step(), 1);
        // The teacher is frozen, so its parameter is exactly as given.
        let teacher = state.fetch(&plan, "teacher.scale").unwrap();
        assert_eq!(values(&teacher), vec![3.0, 4.0]);
        let student = state.fetch(&plan, "student.scale").unwrap();
        assert_ne!(values(&student), vec![1.0, 1.0]);
    }

    #[test]
    fn the_lr_knob_takes_effect_from_the_next_step() {
        let (plan, mut state) = open(fixtures::scaled_mean(), sgd(0.5));
        assert_eq!(state.lr(), 0.5);
        state.set_lr(0.25);
        assert_eq!(state.lr(), 0.25);
        // d(mean(x*w))/dw with x = [[2, 2]] is [1, 1] (two values, mean
        // over both), so one step at 0.25 lands at w - 0.25.
        step(&plan, &mut state, &[2.0, 2.0]);
        let w = values(&state.fetch(&plan, "m.w").unwrap());
        assert!((w[0] - 0.75).abs() < 1e-6, "{w:?}");
    }

    #[test]
    fn reseeding_restarts_the_draws() {
        let (_, mut state) = open(fixtures::scaled_mean(), sgd(0.1));
        let first = state.rng.clone();
        state.set_seed(7).unwrap();
        assert_eq!(state.seed(), 7);
        let seven = crate::tensor::to_tensor(state.pass().rng).unwrap();
        state.set_seed(0).unwrap();
        let zero = crate::tensor::to_tensor(state.pass().rng).unwrap();
        assert_eq!(values(&zero), values(&crate::tensor::to_tensor(&first).unwrap()));
        assert_ne!(values(&seven), values(&zero));
    }

    #[test]
    fn freezing_moves_what_autodiff_differentiates() {
        let (plan, mut state) = open(fixtures::teacher_and_student(), adamw());
        assert_eq!(plan.paths_of(state.argnums()), vec!["student.scale"]);
        let moved = state.set_frozen(&plan, "student.*", true);
        // Nothing would be left to train, so it is refused whole.
        let e = moved.unwrap_err().to_string();
        assert!(e.contains("would leave nothing to train"), "{e}");
        assert_eq!(plan.paths_of(state.argnums()), vec!["student.scale"]);
    }

    #[test]
    fn freezing_a_pattern_that_matches_nothing_is_refused() {
        let (plan, mut state) = open(fixtures::teacher_and_student(), sgd(0.1));
        let e = state
            .set_frozen(&plan, "nowhere.*", true)
            .unwrap_err()
            .to_string();
        assert!(e.contains("no parameter matches"), "{e}");
    }

    #[test]
    fn freezing_a_frozen_parameter_is_a_no_op_not_an_error() {
        // "teacher.scale" is a real path but not a candidate: the pattern
        // matched, so this is nothing to do rather than a mistake.
        let (plan, mut state) = open(fixtures::teacher_and_student(), sgd(0.1));
        let moved = state.set_frozen(&plan, "student.scale", false).unwrap();
        assert!(moved.is_empty());
    }

    #[test]
    fn put_writes_a_parameter_and_refuses_a_mismatch() {
        let (plan, mut state) = open(fixtures::scaled_mean(), sgd(0.1));
        let good = Tensor {
            dtype: mlx_rs::Dtype::Float32,
            shape: vec![2],
            values: Values::F32(vec![9.0, 9.0]),
        };
        state.put(&plan, "m.w", &good).unwrap();
        assert_eq!(values(&state.fetch(&plan, "m.w").unwrap()), vec![9.0, 9.0]);

        let wrong_shape = Tensor {
            dtype: mlx_rs::Dtype::Float32,
            shape: vec![3],
            values: Values::F32(vec![0.0; 3]),
        };
        let e = state
            .put(&plan, "m.w", &wrong_shape)
            .unwrap_err()
            .to_string();
        assert!(e.contains("is not"), "{e}");
        // Refused, and the parameter is still what it was.
        assert_eq!(values(&state.fetch(&plan, "m.w").unwrap()), vec![9.0, 9.0]);

        let wrong_dtype = Tensor {
            dtype: mlx_rs::Dtype::Int32,
            shape: vec![2],
            values: Values::I32(vec![0, 0]),
        };
        assert!(state.put(&plan, "m.w", &wrong_dtype).is_err());
        assert!(state.put(&plan, "m.nowhere", &good).is_err());
        assert!(state.fetch(&plan, "m.nowhere").is_err());
    }

    #[test]
    fn a_restored_run_steps_where_a_continuous_one_would() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ckpt").display().to_string();
        let rows = [1.0f32, 2.0, 3.0, 4.0];

        let (plan, mut straight) = open(fixtures::teacher_and_student(), adamw());
        for _ in 0..2 {
            step(&plan, &mut straight, &rows);
        }
        straight.save(&plan, &path, "").unwrap();
        for _ in 0..3 {
            step(&plan, &mut straight, &rows);
        }

        let (plan2, mut resumed) = open(fixtures::teacher_and_student(), adamw());
        resumed.restore(&plan2, &path).unwrap();
        assert_eq!(resumed.step(), 2);
        assert!(resumed.loss().is_nan(), "a restore has taken no step yet");
        for _ in 0..3 {
            step(&plan2, &mut resumed, &rows);
        }

        assert_eq!(resumed.step(), straight.step());
        let want = values(&straight.fetch(&plan, "student.scale").unwrap());
        let got = values(&resumed.fetch(&plan2, "student.scale").unwrap());
        for (w, g) in want.iter().zip(&got) {
            assert!((w - g).abs() < 1e-6, "{want:?} against {got:?}");
        }
    }

    #[test]
    fn a_checkpoint_of_another_graph_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ckpt").display().to_string();
        let (plan, state) = open(fixtures::scaled_mean(), sgd(0.1));
        state.save(&plan, &path, "").unwrap();

        let (other, mut into) = open(fixtures::teacher_and_student(), sgd(0.1));
        let e = into.restore(&other, &path).unwrap_err().to_string();
        assert!(e.contains("belongs to another graph"), "{e}");
    }

    #[test]
    fn a_checkpoint_of_another_optimizer_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ckpt").display().to_string();
        let (plan, state) = open(fixtures::scaled_mean(), sgd(0.1));
        state.save(&plan, &path, "").unwrap();

        let (plan2, mut into) = open(fixtures::scaled_mean(), adamw());
        let e = into.restore(&plan2, &path).unwrap_err().to_string();
        assert!(e.contains("different optimizer"), "{e}");
    }

    #[test]
    fn an_adamw_checkpoint_without_its_slots_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ckpt").display().to_string();
        let (plan, mut state) = open(fixtures::scaled_mean(), adamw());
        step(&plan, &mut state, &[1.0, 2.0]);
        let written = state.save(&plan, &path, "").unwrap();
        std::fs::remove_file(std::path::Path::new(&written).join("optimizer.safetensors")).unwrap();

        let (plan2, mut into) = open(fixtures::scaled_mean(), adamw());
        let e = into.restore(&plan2, &path).unwrap_err().to_string();
        assert!(e.contains("no optimizer state"), "{e}");
        // Refused before anything moved.
        assert_eq!(into.step(), 0);
    }

    #[test]
    fn a_refused_restore_leaves_the_run_exactly_as_it_was() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ckpt").display().to_string();
        let (plan, state) = open(fixtures::scaled_mean(), sgd(0.1));
        state.save(&plan, &path, "").unwrap();

        let (plan2, mut into) = open(fixtures::scaled_mean(), sgd(0.1));
        step(&plan2, &mut into, &[1.0, 2.0]);
        let before = values(&into.fetch(&plan2, "m.w").unwrap());

        // A parameter file the manifest promises but the directory lacks.
        std::fs::remove_file(
            std::path::Path::new(&path).join("parameters.safetensors"),
        )
        .unwrap();
        assert!(into.restore(&plan2, &path).is_err());
        assert_eq!(into.step(), 1);
        assert_eq!(values(&into.fetch(&plan2, "m.w").unwrap()), before);
    }

    #[test]
    fn export_writes_one_models_parameters_with_prefix_stripped() {
        // student.scale exports as "scale": the GraphConfig model name is
        // the one thing a serving consumer does not carry.
        let (plan, state) = open(fixtures::teacher_and_student(), sgd(0.1));
        let dir = tempfile::tempdir().unwrap();
        let export_to = dir.path().join("out").display().to_string();

        let renamed = state.export_model(&plan, "student", &export_to).unwrap();
        assert_eq!(renamed, vec![("student.scale".to_string(), "scale".to_string())]);

        let file = std::path::Path::new(&export_to).join("model.safetensors");
        let arrays = Array::load_safetensors(&file).unwrap();
        assert_eq!(arrays.len(), 1, "only the student's parameter is exported");
        let array = &arrays["scale"];
        assert_eq!(array.dtype(), Dtype::Float32);
        assert_eq!(crate::tensor::to_tensor(array).unwrap().shape, vec![2]);
        let exported = values(&crate::tensor::to_tensor(array).unwrap());
        let held = values(&state.fetch(&plan, "student.scale").unwrap());
        for (got, want) in exported.iter().zip(&held) {
            assert!((got - want).abs() < 1e-6, "{exported:?} against {held:?}");
        }
    }

    #[test]
    fn export_refuses_a_model_name_this_run_does_not_have() {
        let (plan, state) = open(fixtures::scaled_mean(), sgd(0.1));
        let dir = tempfile::tempdir().unwrap();
        let e = state
            .export_model(&plan, "elsewhere", &dir.path().display().to_string())
            .unwrap_err();
        assert!(e.to_string().contains("no model named"), "{e}");
    }
}
