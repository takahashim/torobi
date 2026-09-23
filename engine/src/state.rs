//! What a run accumulates: parameters, optimizer slots, the RNG, and the
//! counters. Everything that a checkpoint has to carry, and nothing that a
//! GraphConfig already says.
//!
//! The plan (`crate::plan`) is fixed when a run opens; this is the part
//! that moves. It moves in one place, [`TrainState::advance`], which is
//! written as a transaction: the next state is built and evaluated in full
//! before any of it is assigned, so a step that fails leaves the run
//! exactly as it was (docs/plan.md section 5A.4).

use anyhow::{Context, Result};
use crate::mlxc::transforms::eval;
use crate::mlxc::Array;

use crate::checkpoint;
use crate::optimizer::{global_norm, scaled, Config as OptimizerConfig, Optimizer};
use crate::plan::{Pattern, Plan};
use crate::tensor::{to_tensor, Tensor};

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
    /// The L2 norm of the last step's gradients, before any clipping, so a
    /// watcher can see how close a run is to its cap rather than having to
    /// guess. NaN until a step is taken, and when a step is skipped.
    last_grad_norm: f32,
    /// Gradients waiting for a step. Held here rather than by the caller
    /// because they are device arrays parallel to `argnums`, which is this
    /// module's invariant to keep: a caller holding them could freeze a
    /// parameter between two of them and hand back a vector that no longer
    /// lines up.
    accumulated: Accumulator,
}

/// Gradients summed across the parts of a batch, and the losses they came
/// from, waiting for one step.
///
/// A batch that does not fit is trained as several that do, summing their
/// gradients and updating once. The gradients of a sum are the sum of the
/// gradients, so that reaches where one step over the whole batch would.
#[derive(Default)]
struct Accumulator {
    /// One per differentiated parameter, in `argnums` order; empty when
    /// nothing is waiting.
    grads: Vec<Array>,
    /// The losses that produced them, so a step can report their mean.
    losses: Vec<f32>,
}

impl Accumulator {
    fn parts(&self) -> usize {
        self.losses.len()
    }

    /// Adds one part. The new sums are built and evaluated before they
    /// replace the old ones, so a part that fails leaves what was already
    /// held as it was. Evaluated rather than left lazy: ten parts should be
    /// ten sums, not a graph ten deep.
    fn add(&mut self, loss: f32, grads: &[Array]) -> Result<()> {
        let sums: Vec<Array> = if self.grads.is_empty() {
            grads.to_vec()
        } else {
            self.grads
                .iter()
                .zip(grads)
                .map(|(held, new)| held.add(new))
                .collect::<std::result::Result<_, _>>()?
        };
        eval(sums.iter())?;
        self.grads = sums;
        self.losses.push(loss);
        Ok(())
    }

    /// The summed gradients and the mean loss, or `None` when nothing is
    /// waiting. Left in place: they go only once the step they are for has
    /// been taken.
    fn waiting(&self) -> Option<(Vec<Array>, f32)> {
        if self.losses.is_empty() {
            return None;
        }
        let mean = self.losses.iter().sum::<f32>() / self.losses.len() as f32;
        Some((self.grads.clone(), mean))
    }
}

/// What a step came to, before any of it is the state's: the RNG and the
/// numbers always, and the new parameters and optimizer when the step is
/// taken.
struct StepOutcome {
    rng: Array,
    loss: f32,
    grad_norm: f32,
    update: Option<(Vec<Array>, Optimizer)>,
}

impl StepOutcome {
    /// A step not taken: the counters and the RNG still move, the
    /// parameters and the optimizer do not.
    fn skipped(rng: Array, loss: f32, grad_norm: f32) -> Result<Self> {
        eval(std::iter::once(&rng))?;
        Ok(Self {
            rng,
            loss,
            grad_norm,
            update: None,
        })
    }
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
            rng: crate::mlxc::random::key(seed)?,
            seed,
            step: 0,
            last_loss: f32::NAN,
            last_grad_norm: f32::NAN,
            accumulated: Accumulator::default(),
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
    /// Every parameter, in `Plan::paths` order.
    pub fn params(&self) -> &[Array] {
        &self.params
    }

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

    /// The norm of the last step's gradients, before clipping. NaN before
    /// the first step and on a step that was skipped.
    pub fn grad_norm(&self) -> f32 {
        self.last_grad_norm
    }

    /// The largest gradient norm a step may carry, or `None`.
    pub fn clip(&self) -> Option<f32> {
        self.optimizer.config().clip()
    }

    /// A knob: effect begins with the next step. `None` is no cap.
    pub fn set_clip(&mut self, clip: Option<f32>) {
        self.optimizer.config_mut().set_clip(clip);
    }

    /// The L2 norm of every parameter, over the whole model.
    ///
    /// On demand rather than per step: it is a reduction over all the
    /// weights, and unlike the gradient norm nothing in a step reads it.
    /// Frozen parameters count, so this is the size of the model and not
    /// of what is being trained.
    pub fn param_norm(&self) -> Result<f32> {
        global_norm(&self.params)
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
        self.rng = crate::mlxc::random::key(seed)?;
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
        let matcher = Pattern::parse(pattern)?;
        let mut wanted: Vec<i32> = Vec::new();
        let mut moved = Vec::new();
        let mut matched_any = false;
        for &i in &plan.candidates {
            let path = &plan.paths[i as usize];
            let matches = matcher.matches(path);
            matched_any |= matches;
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
            !moved.is_empty() || matched_any,
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
            self.accumulated.parts() == 0,
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
        let array = tensor.to_array()?;
        anyhow::ensure!(
            array.shape() == self.params[index].shape(),
            "parameter {path:?}: given shape {:?} is not {:?}",
            array.shape(),
            self.params[index].shape()
        );
        // Converted rather than refused, for the reason importing is: the
        // boundary carries f32 and a parameter may be held in something
        // else, so what a caller can say is the numbers, not the width.
        let array = array.as_dtype(self.params[index].dtype())?;
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
    /// evaluated by the time this is called. A finite loss with a
    /// non-finite gradient is refused the same way, which is why the norm
    /// is taken before the update rather than only when clipping is on.
    ///
    /// **Gradients are clipped to `clip`, if one is set.** The norm is
    /// global over every differentiated parameter, taken in f32, and when
    /// it exceeds the cap every gradient is scaled by the same factor: the
    /// direction of the step is untouched and only its length is bounded.
    /// The cap is recorded with the optimizer, and `grad_norm()` reports
    /// what the norm was, so a run can say how close it came.
    ///
    /// The counters still move. A step was attempted, its batch was
    /// consumed, and the RNG was drawn from during the forward, so the draw
    /// belongs to the step count exactly as it would have; a resumed run
    /// has to see the same sequence. What does not move is the parameters
    /// and the optimizer's slots, so a policy that lowers the rate and
    /// carries on has something clean to carry on from.
    pub fn advance(&mut self, loss: &Array, grads: &[Array]) -> Result<f32> {
        let outcome = self.outcome(loss, grads)?;
        Ok(self.commit(outcome))
    }

    /// What one step comes to, built and evaluated in full, with none of
    /// it this state's yet.
    fn outcome(&self, loss: &Array, grads: &[Array]) -> Result<StepOutcome> {
        let loss = loss.item::<f32>()?;
        let (rng, _) = crate::mlxc::random::split(&self.rng, 2)?;
        if !loss.is_finite() {
            return StepOutcome::skipped(rng, loss, f32::NAN);
        }
        let norm = global_norm(grads)?;
        if !norm.is_finite() {
            return StepOutcome::skipped(rng, loss, norm);
        }

        let grads: Vec<Array> = match self.optimizer.config().clip() {
            Some(max) if norm > max => scaled(grads, max / norm)?,
            _ => grads.to_vec(),
        };
        let mut params = self.params.clone();
        let optimizer = self.optimizer.next(&mut params, &self.argnums, &grads)?;
        eval(params.iter().chain(optimizer.arrays()).chain(std::iter::once(&rng)))?;
        Ok(StepOutcome {
            rng,
            loss,
            grad_norm: norm,
            update: Some((params, optimizer)),
        })
    }

    /// Makes a step's outcome this state's. Cannot fail, which is what lets
    /// everything that can fail happen first: whether or not the update is
    /// taken, the counters and the RNG move exactly once.
    fn commit(&mut self, outcome: StepOutcome) -> f32 {
        if let Some((params, optimizer)) = outcome.update {
            self.params = params;
            self.optimizer = optimizer;
        }
        self.rng = outcome.rng;
        self.last_loss = outcome.loss;
        self.last_grad_norm = outcome.grad_norm;
        self.step += 1;
        outcome.loss
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
        let value = loss.item::<f32>()?;
        anyhow::ensure!(
            grads.len() == self.argnums.len(),
            "these gradients are for {} parameters and {} are differentiated",
            grads.len(),
            self.argnums.len()
        );
        if !value.is_finite() {
            return Ok(value);
        }

        self.accumulated.add(value, grads)?;
        Ok(value)
    }

    /// Takes the step the accumulated gradients ask for, and reports the
    /// mean of the losses they came from.
    ///
    /// Refuses when nothing is waiting: a step from no gradients is not a
    /// step of zero, it is a caller that has lost track of where it is.
    pub fn apply(&mut self) -> Result<f32> {
        let Some((grads, mean)) = self.accumulated.waiting() else {
            anyhow::bail!("nothing has been accumulated, so there is no step to take");
        };
        // A step that fails leaves the run as it was, and that includes
        // what was waiting for it.
        let loss = self.advance(&Array::from_f32(mean)?, &grads)?;
        self.accumulated = Accumulator::default();
        Ok(loss)
    }

    /// How many parts are waiting for a step.
    pub fn accumulated(&self) -> usize {
        self.accumulated.parts()
    }

    /// Throws away what was accumulated, for a caller abandoning a batch
    /// part way through. Returns how many parts went.
    pub fn discard(&mut self) -> usize {
        let parts = self.accumulated.parts();
        self.accumulated = Accumulator::default();
        parts
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
            self.accumulated.parts() == 0,
            "{} parts are accumulated and a checkpoint does not hold them. \
             Apply them or discard them first",
            self.accumulated()
        );
        let run: serde_json::Value = if run.trim().is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(run).context("the run metadata is not JSON")?
        };
        let state = checkpoint::State {
            config_digest: &plan.config_digest,
            graph_json: &plan.graph_json,
            semantics_version: plan.semantics_version,
            step: self.step,
            optimizer: self.optimizer.config(),
            optimizer_steps: self.optimizer.steps_taken(),
            parameters: plan.paths.iter().cloned().zip(self.params.iter()).collect(),
            argnums: &self.argnums,
            slots: self.optimizer.slots(),
            rng: &self.rng,
            seed: self.seed,
            run,
        };
        Ok(checkpoint::write(dir, state)?.display().to_string())
    }

    /// Restores state written by [`TrainState::save`], refusing anything
    /// that does not belong to this run: another description, another
    /// optimizer, a parameter of another shape, a missing slot.
    ///
    /// Returns what the caller recorded in `run`, so that whoever owns the
    /// data can put its sampler back where the checkpoint left it.
    ///
    /// Reading and accepting is [`checkpoint::Loaded::accept`]; this is the
    /// commit. They are separate so that "nothing moves until everything is
    /// checked" is a fact about the types rather than a claim in a comment:
    /// an [`checkpoint::Accepted`] only exists if every check passed.
    pub fn restore(&mut self, plan: &Plan, dir: &str) -> Result<String> {
        let accepted = checkpoint::read(dir)?.accept(&checkpoint::Run {
            plan,
            params: &self.params,
            argnums: &self.argnums,
            optimizer: &self.optimizer,
        })?;

        // Everything is here and consistent; make it real before touching
        // this state, so a failure below cannot leave it half restored.
        eval(
            accepted
                .params
                .iter()
                .chain(accepted.moments.arrays())
                .chain(std::iter::once(&accepted.rng)),
        )?;

        self.params = accepted.params;
        self.optimizer
            .restore(accepted.moments, accepted.optimizer_steps);
        self.rng = accepted.rng;
        self.seed = accepted.seed;
        self.step = accepted.step;
        self.last_loss = f32::NAN;
        // Gradients for parameters that are no longer there. Going back is
        // the answer to a batch that went wrong, so they are dropped
        // rather than refused.
        self.accumulated = Accumulator::default();
        Ok(accepted.run)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor;
    use crate::interp::Taps;
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
        OptimizerConfig::Sgd { lr, clip: None }
    }

    fn adamw() -> OptimizerConfig {
        OptimizerConfig::AdamW {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            clip: None,
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
    fn the_gradient_norm_is_reported_whether_or_not_a_clip_is_set() {
        let (plan, mut state) = open(fixtures::scaled_mean(), sgd(0.5));
        assert_eq!(state.clip(), None);
        assert!(state.grad_norm().is_nan());
        step(&plan, &mut state, &[2.0, 2.0]);
        // The gradient is [1, 1], so the norm is sqrt(2).
        assert!((state.grad_norm() - 2.0f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn a_clip_bounds_the_norm_of_the_step_and_leaves_its_direction() {
        let (plan, mut state) = open(fixtures::scaled_mean(), sgd(0.5));
        state.set_clip(Some(0.1));
        assert_eq!(state.clip(), Some(0.1));
        step(&plan, &mut state, &[2.0, 2.0]);
        // The norm is still reported as it was before clipping, so a
        // watcher sees how far past the cap the run went.
        assert!((state.grad_norm() - 2.0f32.sqrt()).abs() < 1e-5);
        // The step is lr * (clip / norm) * [1, 1], whose own norm is
        // lr * clip = 0.05. The weights start at [1, 2] and both move by
        // the same amount, since one factor scales the whole vector.
        let w = values(&state.fetch(&plan, "m.w").unwrap());
        let moved = ((1.0 - w[0]).powi(2) + (2.0 - w[1]).powi(2)).sqrt();
        assert!((moved - 0.5 * 0.1).abs() < 1e-5, "{w:?}");
        assert!(((1.0 - w[0]) - (2.0 - w[1])).abs() < 1e-6, "{w:?}");
    }

    #[test]
    fn the_parameter_norm_is_over_the_whole_model() {
        let (plan, mut state) = open(fixtures::scaled_mean(), sgd(0.1));
        // The one parameter holds [1.0, 2.0].
        assert!((state.param_norm().unwrap() - 5.0f32.sqrt()).abs() < 1e-5);
        // A step moves it, so the norm it reports moves too.
        step(&plan, &mut state, &[2.0, 2.0]);
        assert!((state.param_norm().unwrap() - 5.0f32.sqrt()).abs() > 1e-4);
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
            shape: vec![2],
            values: Values::F32(vec![9.0, 9.0]),
        };
        state.put(&plan, "m.w", &good).unwrap();
        assert_eq!(values(&state.fetch(&plan, "m.w").unwrap()), vec![9.0, 9.0]);

        let wrong_shape = Tensor {
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

        // The other payload the boundary carries is converted to what
        // the parameter holds rather than refused: what a caller can say
        // is the numbers, and the width is the graph's.
        let integers = Tensor {
            shape: vec![2],
            values: Values::I32(vec![7, 8]),
        };
        state.put(&plan, "m.w", &integers).unwrap();
        assert_eq!(values(&state.fetch(&plan, "m.w").unwrap()), vec![7.0, 8.0]);

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


    /// A part that cannot be added leaves what was already waiting as it
    /// was: the sums are built before they replace the old ones.
    #[test]
    fn a_part_that_fails_to_add_leaves_what_was_accumulated() {
        let (_, mut state) = open(fixtures::scaled_mean(), sgd(0.5));
        let loss = Array::from_f32(1.0).unwrap();
        let fits = [Array::from_slice(&[1.0f32, 1.0], &[2]).unwrap()];
        let does_not = [Array::from_slice(&[1.0f32, 1.0, 1.0], &[3]).unwrap()];

        state.accumulate(&loss, &fits).unwrap();
        assert!(state.accumulate(&loss, &does_not).is_err(), "2 and 3 do not add");

        assert_eq!(state.accumulated(), 1);
        state.apply().unwrap();
        assert_eq!(state.step(), 1);
    }

}
