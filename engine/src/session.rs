//! The session: a loaded graph plus its state, and the few verbs that act
//! on it. Deliberately narrow, because this is the surface the Ruby
//! extension binds (docs/plan.md section 4).
//!
//! A session owns parameters and counters; it does not own data. Every step
//! is given its batch (docs/plan.md section 5A.2): the caller decides what
//! the model sees, and nothing here reads a file or calls back into Ruby.
//!
//! Everything below is delegation. What a run is doing lives in
//! [`crate::plan`], what it has accumulated in [`crate::state`], and how a
//! step is computed in [`crate::executor`]; this file is the vocabulary
//! those three are reachable through, and the one place a step's parts are
//! put in order.
//!
//! It comes in two halves. [`SessionCore`] holds the MLX state and does
//! the work; it is crate-private, so nothing outside can reach MLX past
//! the runtime. [`Session`] is the public face, and every one of its
//! methods that touches MLX goes through [`crate::runtime`]. Which methods
//! those are is decided here, where MLX is owned, and never by the caller
//! (notes/ENGINE_RUNTIME_BOUNDARY_PLAN.md section 4).

use std::collections::BTreeMap;

use anyhow::{Context, Result};

use crate::executor::{self, Taps};
use crate::interp::Stat;
use crate::optimizer::Config as OptimizerConfig;
use crate::plan::{Plan, Weights};
use crate::runtime::{runtime, RuntimeError};
use crate::state::TrainState;
use crate::tensor::{to_tensor, Batch, Tensor, Values};

/// What a caller gets back: either the value, or which layer refused.
pub type Outcome<T> = std::result::Result<T, RuntimeError>;

/// A loaded run. The public face; the state is behind it.
pub struct Session {
    /// Taken out by `close`, and by `Drop` if `close` never came.
    core: Option<SessionCore>,
}

/// The MLX state, and the work on it. Crate-private on purpose: reaching
/// this without the runtime is the bug the runtime exists to prevent.
pub(crate) struct SessionCore {
    /// What this run does, settled when it opened.
    plan: Plan,
    /// What it has accumulated since.
    state: TrainState,
    /// What the window is watching: node name to how much of it to bring
    /// back. Read-only; changing the set changes what a step evaluates,
    /// so it takes effect from the next one.
    taps: Taps,
    /// The last pass's tapped values, as copies.
    tapped: BTreeMap<String, Tensor>,
}

impl SessionCore {
    fn open_with(
        graph_json: &str,
        weights: Weights<'_>,
        optimizer: OptimizerConfig,
        seed: u64,
    ) -> Result<Self> {
        // One seed, used twice: a parameter built from its declaration is
        // drawn from where the run's first step will draw from, so what a
        // run does is a function of its seed and nothing else.
        let (plan, params) = Plan::open_seeded(graph_json, weights, seed)?;
        let state = TrainState::new(&plan, params, optimizer, seed)?;
        Ok(Self {
            plan,
            state,
            taps: Taps::new(),
            tapped: BTreeMap::new(),
        })
    }

    pub(crate) fn step(&self) -> usize {
        self.state.step()
    }

    pub(crate) fn loss(&self) -> f32 {
        self.state.loss()
    }

    pub(crate) fn lr(&self) -> f32 {
        self.state.lr()
    }

    /// A knob: effect begins with the next step.
    pub(crate) fn set_lr(&mut self, lr: f32) {
        self.state.set_lr(lr);
    }

    /// What update rule this session runs, as data.
    pub(crate) fn optimizer_config(&self) -> &OptimizerConfig {
        self.state.optimizer_config()
    }

    pub(crate) fn seed(&self) -> u64 {
        self.state.seed()
    }

    /// Restarts the RNG. A knob like any other: after this the draws are a
    /// function of the new seed alone.
    pub(crate) fn set_seed(&mut self, seed: u64) -> Result<()> {
        self.state.set_seed(seed)
    }

    /// Watches a named node. `stat` is "full", "mean", "norm" or "extent";
    /// a reduction costs a scalar per step where "full" costs the tensor,
    /// which is why a standing tap should reduce.
    pub(crate) fn tap(&mut self, name: &str, stat: &str) -> Result<()> {
        let stat = Stat::parse(stat)
            .with_context(|| format!("{stat:?} is not a statistic (full, mean, norm, extent)"))?;
        anyhow::ensure!(
            self.plan.node_names().iter().any(|n| n == name),
            "no value is named {name:?} here (this run has {:?})",
            self.plan.node_names()
        );
        self.taps.insert(name.to_string(), stat);
        Ok(())
    }

    /// Stops watching. Returns whether it was being watched.
    pub(crate) fn untap(&mut self, name: &str) -> bool {
        self.taps.remove(name).is_some()
    }

    /// What is being watched.
    pub(crate) fn taps(&self) -> Vec<String> {
        self.taps.keys().cloned().collect()
    }

    /// Every name a tap could ask for.
    pub(crate) fn node_names(&self) -> Vec<String> {
        self.plan.node_names().to_vec()
    }

    /// What the most recent pass's taps saw, by name.
    ///
    /// A copy of a copy: these already live on the host, and the caller
    /// gets its own. A `full` tap therefore costs the tensor again here,
    /// which is the other half of why a standing tap should reduce.
    pub(crate) fn tapped(&self) -> Vec<(String, Tensor)> {
        self.tapped
            .iter()
            .map(|(name, tensor)| {
                (
                    name.clone(),
                    Tensor {
                        dtype: tensor.dtype,
                        shape: tensor.shape.clone(),
                        values: match &tensor.values {
                            Values::F32(v) => Values::F32(v.clone()),
                            Values::I32(v) => Values::I32(v.clone()),
                        },
                    },
                )
            })
            .collect()
    }

    /// Which parameters are currently differentiated, by qualified path.
    pub(crate) fn trainable(&self) -> Vec<String> {
        self.plan.paths_of(self.state.argnums())
    }

    /// Every parameter a model declared trainable, whether or not it is
    /// frozen right now: the set `freeze` and `unfreeze` move within.
    pub(crate) fn trainable_candidates(&self) -> Vec<String> {
        self.plan.candidate_paths()
    }

    /// Freezes or unfreezes parameters whose path matches `pattern`, and
    /// returns those that moved.
    pub(crate) fn set_frozen(&mut self, pattern: &str, frozen: bool) -> Result<Vec<String>> {
        self.state.set_frozen(&self.plan, pattern, frozen)
    }

    /// Writes one parameter, by qualified path, from a copy. The window's
    /// B capability (docs/plan.md section 8.3).
    pub(crate) fn put(&mut self, path: &str, tensor: &Tensor) -> Result<()> {
        self.state.put(&self.plan, path, tensor)
    }

    /// One step on `batch`: forward, backward, optimizer update. Long-
    /// running and free of any Ruby, so the extension calls it with the GVL
    /// released.
    pub(crate) fn run_step(&mut self, batch: &Batch) -> Result<f32> {
        self.trains()?;
        let fields = self.plan.bind(batch)?;
        self.update(&fields)
    }

    /// That there is something to differentiate at all.
    ///
    /// A session opened with `train: []` is for reading: it evaluates and
    /// it differentiates by its inputs, and a step through it would be a
    /// step over no parameters. Said here rather than left to MLX, which
    /// would answer with something about empty argnums.
    fn trains(&self) -> Result<()> {
        anyhow::ensure!(
            !self.state.pass().argnums.is_empty(),
            "this session trains nothing (it was opened with no model to train), \
             so there is no step to take. It can evaluate and it can report \
             gradients by its inputs"
        );
        Ok(())
    }

    /// Adds one batch's gradients to what is waiting, without stepping.
    ///
    /// A batch too large to hold is trained as several that fit
    /// (docs/plan.md section 15.35). The taps report this pass like any
    /// other, so the parts of a batch can be watched as they go.
    pub(crate) fn accumulate(&mut self, batch: &Batch) -> Result<f32> {
        self.trains()?;
        let fields = self.plan.bind(batch)?;
        let (loss, grads, tapped) =
            executor::differentiate(&self.plan, self.state.pass(), &fields, &self.taps)?;
        let tapped = Self::to_host(tapped)?;
        let loss = self.state.accumulate(&loss, &grads)?;
        self.tapped = tapped;
        Ok(loss)
    }

    /// Takes the step the accumulated gradients ask for.
    pub(crate) fn apply(&mut self) -> Result<f32> {
        self.state.apply()
    }

    /// How many parts are waiting for a step.
    pub(crate) fn accumulated(&self) -> usize {
        self.state.accumulated()
    }

    /// Throws away what was accumulated. Returns how many parts went.
    pub(crate) fn discard(&mut self) -> usize {
        self.state.discard()
    }

    /// The loss for `batch`, without taking a step.
    ///
    /// What a validation set is read with. No gradients, so it costs a
    /// forward rather than a forward and a backward, and no randomness, so
    /// dropout stands aside and the number is the model's rather than a
    /// sample of it. Nothing about the run moves: not the parameters, not
    /// the counters, not the RNG, and not the loss a watcher reads.
    ///
    /// The taps report this pass, as they report any pass.
    pub(crate) fn evaluate(&mut self, batch: &Batch) -> Result<f32> {
        let fields = self.plan.bind(batch)?;
        let (loss, tapped) =
            executor::evaluate(&self.plan, self.state.pass().params, &fields, &self.taps)?;
        self.record(tapped)?;
        Ok(loss.item::<f32>())
    }

    /// Gradients with respect to named batch fields, by field name.
    ///
    /// The parameters do not move and are not differentiated. What this is
    /// for is a loss over values computed elsewhere: a gradient cache
    /// needs the loss differentiated by the representations it was given,
    /// so the passes that produced them can be re-run with that as their
    /// seed (docs/plan.md section 15.36).
    pub(crate) fn field_gradients(
        &self,
        batch: &Batch,
        of: &[String],
    ) -> Result<Vec<(String, Tensor)>> {
        let fields = self.plan.bind(batch)?;
        let (_, grads) =
            executor::differentiate_fields(&self.plan, self.state.pass().params, &fields, of)?;
        of.iter()
            .cloned()
            .zip(grads)
            .map(|(name, grad)| Ok((name, to_tensor(&grad)?)))
            .collect()
    }

    /// Gradients as copies, by qualified parameter path. Only differentiated
    /// parameters appear: a frozen model's have none.
    pub(crate) fn gradients(&self, batch: &Batch) -> Result<Vec<(String, Tensor)>> {
        let fields = self.plan.bind(batch)?;
        let (_, grads, _) =
            executor::differentiate(&self.plan, self.state.pass(), &fields, &Taps::new())?;
        self.trainable()
            .into_iter()
            .zip(grads)
            .map(|(path, grad)| Ok((path, to_tensor(&grad)?)))
            .collect()
    }

    /// A copy of one parameter, by qualified path. Copies, not handles:
    /// nothing that lives on the device escapes this crate.
    pub(crate) fn fetch(&self, path: &str) -> Result<Tensor> {
        self.state.fetch(&self.plan, path)
    }

    /// Qualified parameter paths, in the order the engine keeps them.
    pub(crate) fn parameter_paths(&self) -> Vec<String> {
        self.plan.paths.clone()
    }

    /// Every batch field the run reads, across the models and the objective.
    pub(crate) fn input_names(&self) -> Vec<String> {
        self.plan.input_names().to_vec()
    }

    /// Writes the run's state and the description it belongs to. Atomic
    /// (docs/plan.md section 11.2). `run` is the caller's own record
    /// (epoch, batch position, sampler state), written verbatim.
    pub(crate) fn save(&self, dir: &str, run: &str) -> Result<String> {
        self.state.save(&self.plan, dir, run)
    }

    /// Writes one model's parameters as an HF-compatible fp32 safetensors
    /// file, stripping the GraphConfig model name from each path.
    pub(crate) fn export_model(&self, model: &str, dir: &str) -> Result<Vec<(String, String)>> {
        self.state.export_model(&self.plan, model, dir)
    }

    /// Restores state written by [`Session::save`], refusing anything that
    /// does not belong to this session. Returns the caller's record.
    pub(crate) fn restore(&mut self, dir: &str) -> Result<String> {
        self.state.restore(&self.plan, dir)
    }

    /// One step over already-bound inputs: differentiate, read the taps,
    /// then commit. The taps are converted before the state moves so that
    /// a step either happens whole or not at all.
    fn update(&mut self, fields: &BTreeMap<String, mlx_rs::Array>) -> Result<f32> {
        let (loss, grads, tapped) =
            executor::differentiate(&self.plan, self.state.pass(), fields, &self.taps)?;
        // Brought back before the state moves, so a step either happens
        // whole or not at all.
        let tapped = Self::to_host(tapped)?;
        let loss = self.state.advance(&loss, &grads)?;
        self.tapped = tapped;
        Ok(loss)
    }

    /// Keeps what the taps saw, as host-side copies.
    fn record(&mut self, tapped: executor::Tapped) -> Result<()> {
        self.tapped = Self::to_host(tapped)?;
        Ok(())
    }

    fn to_host(tapped: executor::Tapped) -> Result<BTreeMap<String, Tensor>> {
        tapped
            .iter()
            .map(|(name, value)| Ok((name.clone(), to_tensor(value)?)))
            .collect()
    }
}

/// The public face. Every method that touches MLX goes through the
/// runtime; which ones those are is settled here rather than by whoever
/// is calling (notes/ENGINE_RUNTIME_BOUNDARY_PLAN.md section 4.3).
///
/// The line is exact. A method belongs on the plain side only if it makes
/// no `Array`, evaluates none, copies nothing off the device, frees none,
/// and touches neither the allocator nor the stream. Everything else waits
/// its turn.
impl Session {
    /// What `open` runs with: plain SGD, which has no state to restore,
    /// and a seed of zero.
    ///
    /// Not the library's default in any binding sense. Ruby names its own
    /// (`Torobi::Session::DEFAULT_OPTIMIZER`) and always states both, so
    /// what this decides is the command-line tool's runs and the tests'.
    /// Two places say "sgd 0.1" and neither reads the other, which is
    /// fine as long as neither is mistaken for one contract in two
    /// languages.
    const SPIKE: (OptimizerConfig, u64) = (OptimizerConfig::Sgd { lr: 0.1 }, 0);

    /// Loads a GraphConfig and its initial parameters. Parameters are given
    /// by qualified path ("student.head.weight"), which is also the order
    /// the engine keeps them in. Data comes later, one batch per step.
    ///
    /// For the command-line tool and the tests: a caller that has an
    /// update rule and a seed to state uses [`Session::open_with`].
    pub fn open(graph_json: &str, weights: Weights<'_>) -> Outcome<Self> {
        let (optimizer, seed) = Self::SPIKE;
        Self::open_with(graph_json, weights, optimizer, seed)
    }

    /// The same, with the update rule named and the run's seed given.
    ///
    /// The seed is where every draw this run makes comes from: the
    /// parameters a `fresh:` pattern builds at open, and the ops that draw
    /// at each step. A run is reproducible from its graph, its parameters
    /// and this.
    pub fn open_with(
        graph_json: &str,
        weights: Weights<'_>,
        optimizer: OptimizerConfig,
        seed: u64,
    ) -> Outcome<Self> {
        // Opening builds every parameter, an RNG key and the optimizer's
        // slots, so it waits its turn like any other MLX work.
        let core =
            runtime().execute(|| SessionCore::open_with(graph_json, weights, optimizer, seed))?;
        Ok(Self { core: Some(core) })
    }

    /// Releases the run's device memory. Idempotent, and afterwards every
    /// method refuses rather than pretending. Returns whether this call was
    /// the one that closed it.
    ///
    /// In a forked child the memory is leaked rather than freed, and the
    /// refusal says so; the device those handles name did not come along.
    pub fn close(&mut self) -> Outcome<bool> {
        let Some(core) = self.core.take() else {
            return Ok(false);
        };
        runtime().release(core)?;
        Ok(true)
    }

    pub fn closed(&self) -> bool {
        self.core.is_none()
    }

    // --- plain: no Array is made, evaluated, copied or freed ---

    pub fn step(&self) -> Outcome<usize> {
        Ok(self.core()?.step())
    }

    pub fn loss(&self) -> Outcome<f32> {
        Ok(self.core()?.loss())
    }

    pub fn lr(&self) -> Outcome<f32> {
        Ok(self.core()?.lr())
    }

    /// A knob: effect begins with the next step.
    pub fn set_lr(&mut self, lr: f32) -> Outcome<()> {
        self.core_mut()?.set_lr(lr);
        Ok(())
    }

    /// What update rule this session runs, as data.
    pub fn optimizer_config(&self) -> Outcome<OptimizerConfig> {
        Ok(self.core()?.optimizer_config().clone())
    }

    pub fn seed(&self) -> Outcome<u64> {
        Ok(self.core()?.seed())
    }

    /// Watches a named node. `stat` is "full", "mean", "norm" or "extent";
    /// a reduction costs a scalar per step where "full" costs the tensor,
    /// which is why a standing tap should reduce.
    pub fn tap(&mut self, name: &str, stat: &str) -> Outcome<()> {
        Ok(self.core_mut()?.tap(name, stat)?)
    }

    /// Stops watching. Returns whether it was being watched.
    pub fn untap(&mut self, name: &str) -> Outcome<bool> {
        Ok(self.core_mut()?.untap(name))
    }

    /// What is being watched.
    pub fn taps(&self) -> Outcome<Vec<String>> {
        Ok(self.core()?.taps())
    }

    /// Every name a tap could ask for.
    pub fn node_names(&self) -> Outcome<Vec<String>> {
        Ok(self.core()?.node_names())
    }

    /// What the most recent pass's taps saw. Already on the host, so this
    /// copies rather than reads a device.
    pub fn tapped(&self) -> Outcome<Vec<(String, Tensor)>> {
        Ok(self.core()?.tapped())
    }

    /// Which parameters are currently differentiated, by qualified path.
    pub fn trainable(&self) -> Outcome<Vec<String>> {
        Ok(self.core()?.trainable())
    }

    /// Every parameter a model declared trainable, whether or not it is
    /// frozen right now.
    pub fn trainable_candidates(&self) -> Outcome<Vec<String>> {
        Ok(self.core()?.trainable_candidates())
    }

    /// Qualified parameter paths, in the order the engine keeps them.
    pub fn parameter_paths(&self) -> Outcome<Vec<String>> {
        Ok(self.core()?.parameter_paths())
    }

    /// Every batch field the run reads.
    pub fn input_names(&self) -> Outcome<Vec<String>> {
        Ok(self.core()?.input_names())
    }

    // --- MLX: through the runtime ---

    /// Restarts the RNG. A knob like any other, but it builds a key.
    pub fn set_seed(&mut self, seed: u64) -> Outcome<()> {
        let core = self.core_mut()?;
        runtime().execute(|| core.set_seed(seed))
    }

    /// Freezes or unfreezes parameters whose path matches `pattern`, and
    /// returns those that moved. Not a scalar knob: the optimizer's slots
    /// follow, which means allocating and dropping them.
    pub fn set_frozen(&mut self, pattern: &str, frozen: bool) -> Outcome<Vec<String>> {
        let core = self.core_mut()?;
        runtime().execute(|| core.set_frozen(pattern, frozen))
    }

    /// Writes one parameter, by qualified path, from a copy.
    pub fn put(&mut self, path: &str, tensor: &Tensor) -> Outcome<()> {
        let core = self.core_mut()?;
        runtime().execute(|| core.put(path, tensor))
    }

    /// One step on `batch`: forward, backward, optimizer update.
    pub fn run_step(&mut self, batch: &Batch) -> Outcome<f32> {
        let core = self.core_mut()?;
        runtime().execute(|| core.run_step(batch))
    }

    /// Adds one batch's gradients to what is waiting, without stepping.
    pub fn accumulate(&mut self, batch: &Batch) -> Outcome<f32> {
        let core = self.core_mut()?;
        runtime().execute(|| core.accumulate(batch))
    }

    /// Takes the step the accumulated gradients ask for, and reports the
    /// mean of the losses they came from.
    pub fn apply(&mut self) -> Outcome<f32> {
        let core = self.core_mut()?;
        runtime().execute(|| core.apply())
    }

    /// How many parts are waiting for a step. Touches no MLX.
    pub fn accumulated(&self) -> Outcome<usize> {
        Ok(self.core()?.accumulated())
    }

    /// Throws away what was accumulated, and says how many parts went.
    pub fn discard(&mut self) -> Outcome<usize> {
        let core = self.core_mut()?;
        runtime().execute(|| Ok(core.discard()))
    }

    /// The loss for `batch` without taking a step: no gradients, no
    /// randomness, nothing moved. What a validation set is read with.
    pub fn evaluate(&mut self, batch: &Batch) -> Outcome<f32> {
        let core = self.core_mut()?;
        runtime().execute(|| core.evaluate(batch))
    }

    /// Gradients with respect to named batch fields, by field name. The
    /// parameters do not move.
    pub fn field_gradients(
        &self,
        batch: &Batch,
        of: &[String],
    ) -> Outcome<Vec<(String, Tensor)>> {
        let core = self.core()?;
        runtime().execute(|| core.field_gradients(batch, of))
    }

    /// Gradients as copies, by qualified parameter path. Only
    /// differentiated parameters appear.
    pub fn gradients(&self, batch: &Batch) -> Outcome<Vec<(String, Tensor)>> {
        let core = self.core()?;
        runtime().execute(|| core.gradients(batch))
    }

    /// A copy of one parameter, by qualified path. Copies, not handles:
    /// nothing that lives on the device escapes this crate.
    pub fn fetch(&self, path: &str) -> Outcome<Tensor> {
        let core = self.core()?;
        runtime().execute(|| core.fetch(path))
    }

    /// Writes the run's state and the description it belongs to. Atomic.
    /// `run` is the caller's own record, written verbatim.
    pub fn save(&self, dir: &str, run: &str) -> Outcome<String> {
        let core = self.core()?;
        runtime().execute(|| core.save(dir, run))
    }

    /// Writes one model's parameters as an HF-compatible fp32 safetensors
    /// file, stripping the GraphConfig model name from each path.
    pub fn export_model(&self, model: &str, dir: &str) -> Outcome<Vec<(String, String)>> {
        let core = self.core()?;
        runtime().execute(|| core.export_model(model, dir))
    }

    /// Restores state written by [`Session::save`], refusing anything that
    /// does not belong to this session. Returns the caller's record.
    pub fn restore(&mut self, dir: &str) -> Outcome<String> {
        let core = self.core_mut()?;
        runtime().execute(|| core.restore(dir))
    }

    /// The RNG key as host values, for the tests that watch it move.
    ///
    /// Test-only: a key is state, and nothing outside has a use for it. It
    /// goes through the runtime like everything else, because reading an
    /// array evaluates it and copies it off the device. A test that took
    /// the short way here would be submitting to MLX from a thread of its
    /// own while another test's step was in flight, which is what this
    /// crate's whole runtime exists to prevent.
    #[cfg(test)]
    pub(crate) fn rng_for_test(&self) -> Tensor {
        let core = self.core.as_ref().expect("open");
        runtime()
            .execute(|| crate::tensor::to_tensor(core.state.pass().rng))
            .expect("reading the RNG key")
    }

    fn core(&self) -> Outcome<&SessionCore> {
        self.core.as_ref().ok_or_else(closed)
    }

    fn core_mut(&mut self) -> Outcome<&mut SessionCore> {
        self.core.as_mut().ok_or_else(closed)
    }
}

/// The GC frees device memory as surely as `close` does, so both go the
/// same way. A session dropped without being closed still releases under
/// the gate, and still leaks rather than frees in a forked child.
impl Drop for Session {
    fn drop(&mut self) {
        if let Some(core) = self.core.take() {
            let _ = runtime().release(core);
        }
    }
}

fn closed() -> RuntimeError {
    RuntimeError::Engine(anyhow::anyhow!("this session is closed"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;
    use crate::tensor::Values;

    pub fn session(which: (String, String)) -> Session {
        let (config, weights) = which;
        Session::open(&config, Weights::Inline(&weights)).unwrap()
    }

    pub fn values(t: &Tensor) -> Vec<f32> {
        match &t.values {
            Values::F32(v) => v.clone(),
            Values::I32(v) => v.iter().map(|&i| i as f32).collect(),
        }
    }

    pub fn close(got: &[f32], want: &[f32]) {
        within(got, want, 1e-6);
    }

    /// The same batch with its numbers scaled.
    ///
    /// The fixture's loss is a mean and is linear in x, so this scales the
    /// loss and with it the gradients. Splitting a batch means weighting
    /// its parts, and for a mean over rows the weight is the share of the
    /// rows each part holds; the engine does not guess it because only the
    /// caller knows what its loss is a mean of.
    fn scaled(batch: &Batch, by: f32) -> Batch {
        batch
            .iter()
            .map(|(name, tensor)| {
                let values = match &tensor.values {
                    Values::F32(v) => Values::F32(v.iter().map(|x| x * by).collect()),
                    _ => panic!("this fixture is f32"),
                };
                (
                    name.clone(),
                    Tensor {
                        dtype: tensor.dtype,
                        shape: tensor.shape.clone(),
                        values,
                    },
                )
            })
            .collect()
    }

    pub fn within(got: &[f32], want: &[f32], tolerance: f32) {
        assert_eq!(got.len(), want.len(), "{got:?} against {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < tolerance, "{got:?} against {want:?}");
        }
    }

    #[test]
    fn the_gradient_is_the_one_the_arithmetic_says() {
        // loss = mean(x * w) over four values, so d(loss)/dw_j is the sum
        // of column j divided by four.
        let mut session = session(fixtures::scaled_mean());
        let batch = fixtures::batch_x(&[1.0, 2.0, 3.0, 4.0]);
        let grads = session.gradients(&batch).unwrap();
        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, "m.w");
        close(&values(&grads[0].1), &[(1.0 + 3.0) / 4.0, (2.0 + 4.0) / 4.0]);

        // w = [1, 2], so x * w is [[1, 4], [3, 8]] and the mean is 4.
        let loss = session.evaluate(&batch).unwrap();
        assert!((loss - 4.0).abs() < 1e-6, "{loss}");
    }

    #[test]
    fn a_frozen_model_has_no_gradient_to_report() {
        let session = session(fixtures::teacher_and_student());
        let grads = session.gradients(&fixtures::batch_x(&[1.0, 1.0])).unwrap();
        assert_eq!(
            grads.iter().map(|(p, _)| p.as_str()).collect::<Vec<_>>(),
            vec!["student.scale"]
        );
        assert_eq!(
            session.parameter_paths().unwrap(),
            vec!["student.scale", "teacher.scale"]
        );
        assert_eq!(session.trainable().unwrap(), vec!["student.scale"]);
    }

    #[test]
    fn a_model_that_reads_one_declared_after_it_is_refused_by_name() {
        let session = session(fixtures::reader_before_producer());
        let e = session
            .gradients(&fixtures::batch_x(&[1.0, 1.0]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("source.out has not been produced"), "{e}");
        assert!(e.contains("declared before what reads it"), "{e}");
    }

    #[test]
    fn training_moves_the_loss_toward_the_teacher() {
        let mut session = session(fixtures::teacher_and_student());
        session.set_lr(0.05).unwrap();
        let batch = fixtures::batch_x(&[1.0, 1.0, 2.0, 2.0]);
        let first = session.run_step(&batch).unwrap();
        for _ in 0..50 {
            session.run_step(&batch).unwrap();
        }
        assert_eq!(session.step().unwrap(), 51);
        assert!(session.loss().unwrap() < first * 0.1, "{} -> {}", first, session.loss().unwrap());
        // It converged on the teacher's scale, which is what the objective
        // asks for. Fifty steps of plain SGD get three digits, not six.
        within(&values(&session.fetch("student.scale").unwrap()), &[3.0, 4.0], 1e-2);
        // And the teacher did not move.
        close(&values(&session.fetch("teacher.scale").unwrap()), &[3.0, 4.0]);
    }

    #[test]
    fn a_tap_reports_what_the_step_computed() {
        let mut session = session(fixtures::scaled_mean());
        assert_eq!(session.node_names().unwrap(), vec!["m.scaled"]);
        assert!(session.tapped().unwrap().is_empty());

        session.tap("m.scaled", "mean").unwrap();
        assert_eq!(session.taps().unwrap(), vec!["m.scaled"]);
        // x * w with x = [[1, 1]] and w = [1, 2] is [[1, 2]], mean 1.5.
        session.run_step(&fixtures::batch_x(&[1.0, 1.0])).unwrap();
        let seen = session.tapped().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, "m.scaled");
        close(&values(&seen[0].1), &[1.5]);

        assert!(session.untap("m.scaled").unwrap());
        assert!(!session.untap("m.scaled").unwrap());
        assert!(session.taps().unwrap().is_empty());
    }

    #[test]
    fn a_full_tap_brings_back_the_tensor_and_a_reduction_a_scalar() {
        let mut session = session(fixtures::scaled_mean());
        session.tap("m.scaled", "full").unwrap();
        session.run_step(&fixtures::batch_x(&[1.0, 1.0])).unwrap();
        let seen = session.tapped().unwrap();
        assert_eq!(seen[0].1.shape, vec![1, 2]);
        close(&values(&seen[0].1), &[1.0, 2.0]);
    }

    /// The name a tap asks for carries the model, so a distillation can
    /// watch the student and the teacher at once. Unqualified, both models
    /// answer to "scaled" and one of them silently wins.
    #[test]
    fn two_models_of_one_shape_are_tapped_apart() {
        let mut session = session(fixtures::teacher_and_student());
        assert_eq!(
            session.node_names().unwrap(),
            vec!["student.scaled", "teacher.scaled"]
        );

        session.tap("student.scaled", "full").unwrap();
        session.tap("teacher.scaled", "full").unwrap();
        session.evaluate(&fixtures::batch_x(&[1.0, 1.0])).unwrap();

        let seen = session.tapped().unwrap();
        assert_eq!(seen.len(), 2);
        // The student's scale is [1, 1] and the teacher's is [3, 4]: each
        // tap reports its own model rather than whichever ran last.
        assert_eq!(seen[0].0, "student.scaled");
        close(&values(&seen[0].1), &[1.0, 1.0]);
        assert_eq!(seen[1].0, "teacher.scaled");
        close(&values(&seen[1].1), &[3.0, 4.0]);
    }

    /// The property accumulation exists for: the gradients of a sum are
    /// the sum of the gradients, so a batch trained as parts reaches where
    /// one step over the whole of it would have.
    #[test]
    fn parts_accumulated_reach_where_the_whole_batch_would() {
        let whole = fixtures::batch_x(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut one = session(fixtures::scaled_mean());
        one.run_step(&whole).unwrap();

        // The same rows, in two halves. The loss is a mean over rows, so
        // each half carries half the weight of the whole; the caller's
        // arithmetic, which is why the engine does not guess it.
        let mut parts = session(fixtures::scaled_mean());
        for half in [
            fixtures::batch_x(&[1.0, 2.0, 3.0, 4.0]),
            fixtures::batch_x(&[5.0, 6.0, 7.0, 8.0]),
        ] {
            parts.accumulate(&scaled(&half, 0.5)).unwrap();
        }
        assert_eq!(parts.accumulated().unwrap(), 2);
        parts.apply().unwrap();
        assert_eq!(parts.accumulated().unwrap(), 0);

        assert_eq!(parts.step().unwrap(), 1, "applying is one step");
        within(
            &values(&parts.fetch("m.w").unwrap()),
            &values(&one.fetch("m.w").unwrap()),
            1e-6,
        );
    }

    #[test]
    fn accumulating_moves_nothing_until_it_is_applied() {
        let mut session = session(fixtures::scaled_mean());
        let before = values(&session.fetch("m.w").unwrap());
        session
            .accumulate(&fixtures::batch_x(&[1.0, 2.0]))
            .unwrap();

        assert_eq!(session.step().unwrap(), 0);
        close(&values(&session.fetch("m.w").unwrap()), &before);

        assert_eq!(session.discard().unwrap(), 1);
        assert_eq!(session.accumulated().unwrap(), 0);
        close(&values(&session.fetch("m.w").unwrap()), &before);
    }

    /// The loss differentiated by what it was given rather than by what
    /// it holds. `scaled_mean` is mean(x * w) with w = [1, 2] over two
    /// elements, so the answer is w / 2 and can be read off the page.
    #[test]
    fn a_loss_can_be_differentiated_by_a_batch_field() {
        let session = session(fixtures::scaled_mean());
        let before = values(&session.fetch("m.w").unwrap());
        let batch = fixtures::batch_x(&[1.0, 2.0]);

        let grads = session
            .field_gradients(&batch, &["x".to_string()])
            .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, "x");
        close(&values(&grads[0].1), &[0.5, 1.0]);
        // The parameters are not what was differentiated, and nothing moved.
        close(&values(&session.fetch("m.w").unwrap()), &before);
        assert_eq!(session.step().unwrap(), 0);
    }

    #[test]
    fn differentiating_by_a_field_the_batch_does_not_have_is_refused() {
        let session = session(fixtures::scaled_mean());
        let batch = fixtures::batch_x(&[1.0, 2.0]);

        let e = session
            .field_gradients(&batch, &["elsewhere".to_string()])
            .unwrap_err()
            .to_string();

        assert!(e.contains("elsewhere"), "{e}");
        assert!(e.contains("\"x\""), "{e}");
    }

    #[test]
    fn a_step_from_nothing_accumulated_is_refused() {
        let mut session = session(fixtures::scaled_mean());
        let e = session.apply().unwrap_err().to_string();

        assert!(e.contains("nothing has been accumulated"), "{e}");
    }

    /// Freezing moves what a gradient is for, and a checkpoint does not
    /// hold what is waiting. Both say so rather than quietly dropping it.
    #[test]
    fn what_is_waiting_blocks_freezing_and_checkpointing() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = session(fixtures::scaled_mean());
        session
            .accumulate(&fixtures::batch_x(&[1.0, 2.0]))
            .unwrap();

        let e = session.set_frozen("m.w", true).unwrap_err().to_string();
        assert!(e.contains("accumulated"), "{e}");
        let e = session
            .save(dir.path().join("c").to_str().unwrap(), "")
            .unwrap_err()
            .to_string();
        assert!(e.contains("accumulated"), "{e}");
    }

    #[test]
    fn a_tap_on_a_name_no_node_carries_is_refused() {
        let mut session = session(fixtures::scaled_mean());
        let e = session.tap("nowhere", "mean").unwrap_err().to_string();
        assert!(e.contains("no value is named"), "{e}");
        let e = session.tap("m.scaled", "median").unwrap_err().to_string();
        assert!(e.contains("is not a statistic"), "{e}");
        assert!(session.taps().unwrap().is_empty());
    }

    #[test]
    fn taps_do_not_change_what_a_step_computes() {
        let batch = fixtures::batch_x(&[1.0, 2.0, 3.0, 4.0]);
        let mut quiet = session(fixtures::scaled_mean());
        let mut watched = session(fixtures::scaled_mean());
        watched.tap("m.scaled", "norm").unwrap();
        for _ in 0..3 {
            quiet.run_step(&batch).unwrap();
            watched.run_step(&batch).unwrap();
        }
        assert_eq!(quiet.loss().unwrap(), watched.loss().unwrap());
        close(
            &values(&watched.fetch("m.w").unwrap()),
            &values(&quiet.fetch("m.w").unwrap()),
        );
    }

    #[test]
    fn the_session_reports_what_it_reads_and_what_it_holds() {
        let session = session(fixtures::teacher_and_student());
        assert_eq!(session.input_names().unwrap(), vec!["x"]);
        assert_eq!(session.trainable_candidates().unwrap(), vec!["student.scale"]);
        assert_eq!(session.optimizer_config().unwrap().name(), "sgd");
        assert_eq!(session.seed().unwrap(), 0);
        assert!(session.loss().unwrap().is_nan());
    }

    #[test]
    fn freezing_and_thawing_move_what_a_step_reports() {
        let (config, weights) = fixtures::teacher_and_student();
        // Train both, so there is something to freeze that leaves a rest.
        let both = config.replace(r#""train":["student"]"#, r#""train":["student","teacher"]"#);
        let mut session = Session::open(&both, Weights::Inline(&weights)).unwrap();
        assert_eq!(
            session.trainable().unwrap(),
            vec!["student.scale", "teacher.scale"]
        );
        let moved = session.set_frozen("teacher.*", true).unwrap();
        assert_eq!(moved, vec!["teacher.scale"]);
        assert_eq!(session.trainable().unwrap(), vec!["student.scale"]);

        let grads = session.gradients(&fixtures::batch_x(&[1.0, 1.0])).unwrap();
        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, "student.scale");

        assert_eq!(session.set_frozen("teacher.*", false).unwrap(), vec!["teacher.scale"]);
        assert_eq!(session.trainable().unwrap(), vec!["student.scale", "teacher.scale"]);
    }

    #[test]
    fn a_checkpoint_round_trips_through_the_session() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ckpt").display().to_string();
        let batch = fixtures::batch_x(&[1.0, 2.0, 3.0, 4.0]);

        let mut session = session(fixtures::scaled_mean());
        session.run_step(&batch).unwrap();
        let written = session.save(&path, "").unwrap();
        assert!(std::path::Path::new(&written).join("manifest.json").exists());
        let want = values(&session.fetch("m.w").unwrap());

        let mut fresh = self::session(fixtures::scaled_mean());
        fresh.restore(&path).unwrap();
        assert_eq!(fresh.step().unwrap(), 1);
        close(&values(&fresh.fetch("m.w").unwrap()), &want);
    }
}

#[cfg(test)]
mod evaluation_tests {
    use super::tests::*;
    use super::*;
    use crate::fixtures;

    #[test]
    fn a_step_whose_loss_is_not_finite_is_not_taken() {
        let (config, weights) = fixtures::divides_by_zero();
        let mut session = Session::open(&config, Weights::Inline(&weights)).unwrap();
        let batch = fixtures::batch_x(&[1.0, 2.0]);

        let loss = session.run_step(&batch).unwrap();
        assert!(!loss.is_finite(), "the fixture should divide by zero");

        // Reported, so a policy sees it. Not applied, so nothing is lost.
        assert!(session.loss().unwrap().is_nan() || session.loss().unwrap().is_infinite());
        close(&values(&session.fetch("m.w").unwrap()), &[0.0, 0.0]);
        // The counter still moves: a step was attempted and its batch
        // consumed, and a hook that fires every N steps must keep firing.
        assert_eq!(session.step().unwrap(), 1);
    }

    #[test]
    fn a_run_recovers_once_the_parameters_are_usable_again() {
        // What a NaN policy does now: nothing was corrupted, so putting the
        // parameter somewhere sane is enough. No checkpoint is needed.
        let (config, weights) = fixtures::divides_by_zero();
        let mut session = Session::open(&config, Weights::Inline(&weights)).unwrap();
        let batch = fixtures::batch_x(&[1.0, 2.0]);
        session.run_step(&batch).unwrap();

        session
            .put(
                "m.w",
                &Tensor {
                    dtype: mlx_rs::Dtype::Float32,
                    shape: vec![2],
                    values: Values::F32(vec![1.0, 1.0]),
                },
            )
            .unwrap();
        let loss = session.run_step(&batch).unwrap();
        assert!(loss.is_finite(), "{loss}");
        assert_eq!(session.step().unwrap(), 2);
    }

    #[test]
    fn the_rng_moves_even_on_a_step_that_was_not_taken() {
        // The forward drew from it, so the sequence belongs to the step
        // count exactly as it would have. A resumed run has to agree.
        let (config, weights) = fixtures::divides_by_zero();
        let mut session = Session::open(&config, Weights::Inline(&weights)).unwrap();
        let before = session.rng_for_test();
        session.run_step(&fixtures::batch_x(&[1.0, 2.0])).unwrap();
        let after = session.rng_for_test();

        assert_ne!(values(&before), values(&after));
    }

    #[test]
    fn evaluating_moves_nothing() {
        let (config, weights) = fixtures::scaled_mean();
        let mut session = Session::open(&config, Weights::Inline(&weights)).unwrap();
        let batch = fixtures::batch_x(&[1.0, 2.0, 3.0, 4.0]);
        session.run_step(&batch).unwrap();
        let (step, loss) = (session.step().unwrap(), session.loss().unwrap());
        let w = values(&session.fetch("m.w").unwrap());
        let rng = values(&session.rng_for_test());

        let seen = session.evaluate(&batch).unwrap();
        assert!(seen.is_finite());
        assert_eq!(session.step().unwrap(), step);
        assert_eq!(session.loss().unwrap(), loss, "the training loss is not an evaluation");
        close(&values(&session.fetch("m.w").unwrap()), &w);
        assert_eq!(
            values(&session.rng_for_test()),
            rng
        );
    }

    #[test]
    fn evaluating_reports_the_loss_the_parameters_give() {
        // mean(x * w) with w = [1, 2] over [[1, 2], [3, 4]] is
        // (1 + 4 + 3 + 8) / 4 = 4.
        let (config, weights) = fixtures::scaled_mean();
        let mut session = Session::open(&config, Weights::Inline(&weights)).unwrap();
        let seen = session.evaluate(&fixtures::batch_x(&[1.0, 2.0, 3.0, 4.0])).unwrap();
        assert!((seen - 4.0).abs() < 1e-6, "{seen}");
    }

    #[test]
    fn evaluating_does_not_sample_dropout() {
        let (config, weights) = fixtures::with_dropout(0.5);
        let mut session = Session::open(&config, Weights::Inline(&weights)).unwrap();
        let batch = fixtures::batch_x(&[1.0, 1.0, 1.0, 1.0]);

        // Twice the same, because no key means no draw.
        let first = session.evaluate(&batch).unwrap();
        let second = session.evaluate(&batch).unwrap();
        assert_eq!(first, second);
        // And it is the loss with everything kept: mean(x * w) = 1.
        assert!((first - 1.0).abs() < 1e-6, "{first}");

        // Training does draw, so it does not agree with it every time.
        let mut drawn = Vec::new();
        for _ in 0..12 {
            drawn.push(session.run_step(&batch).unwrap());
        }
        assert!(
            drawn.iter().any(|l| (l - 1.0).abs() > 1e-6),
            "a training pass should sample dropout: {drawn:?}"
        );
    }

    #[test]
    fn a_tap_reports_the_evaluation_it_watched() {
        let (config, weights) = fixtures::with_dropout(0.0);
        let mut session = Session::open(&config, Weights::Inline(&weights)).unwrap();
        session.tap("m.scaled", "mean").unwrap();
        session.evaluate(&fixtures::batch_x(&[1.0, 3.0])).unwrap();

        let seen = session.tapped().unwrap();
        assert_eq!(seen.len(), 1);
        close(&values(&seen[0].1), &[2.0]);
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    use crate::fixtures;

    fn open(which: (String, String)) -> Session {
        Session::open(&which.0, Weights::Inline(&which.1)).unwrap()
    }

    fn written(session: &Session, run: &str) -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ckpt").display().to_string();
        session.save(&path, run).unwrap();
        (dir, path)
    }

    fn manifest(path: &str) -> serde_json::Value {
        serde_json::from_str(&crate::checkpoint::read_manifest_json(path).unwrap()).unwrap()
    }

    #[test]
    fn a_checkpoint_carries_the_description_and_not_only_its_digest() {
        let (config, weights) = fixtures::scaled_mean();
        let session = Session::open(&config, Weights::Inline(&weights)).unwrap();
        let (_dir, path) = written(&session, "");

        let graph = std::fs::read_to_string(std::path::Path::new(&path).join("graph.json")).unwrap();
        assert_eq!(graph, config);
        // Which is to say: the checkpoint can be read without the run that
        // wrote it, and the digest it claims is of what is actually there.
        assert_eq!(
            manifest(&path)["config_digest"].as_str().unwrap().len(),
            64
        );
    }

    #[test]
    fn a_graph_that_is_not_the_one_the_manifest_claims_is_refused() {
        let session = open(fixtures::scaled_mean());
        let (_dir, path) = written(&session, "");
        let graph = std::path::Path::new(&path).join("graph.json");
        std::fs::write(&graph, fixtures::teacher_and_student().0).unwrap();

        let mut into = open(fixtures::scaled_mean());
        let e = into.restore(&path).unwrap_err().to_string();
        assert!(e.contains("not the description this manifest claims"), "{e}");
    }

    #[test]
    fn the_manifest_inventories_every_parameters_dtype() {
        let session = open(fixtures::teacher_and_student());
        let (_dir, path) = written(&session, "");
        let m = manifest(&path);
        let params = m["parameters"].as_array().unwrap();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0]["path"], "student.scale");
        assert_eq!(params[0]["dtype"], "f32");
        assert_eq!(params[0]["shape"], serde_json::json!([2]));
        assert_eq!(params[0]["trained"], true);
        assert_eq!(params[1]["path"], "teacher.scale");
        assert_eq!(params[1]["trained"], false);
    }

    #[test]
    fn a_parameter_whose_dtype_disagrees_with_the_manifest_is_refused() {
        let session = open(fixtures::scaled_mean());
        let (_dir, path) = written(&session, "");
        let file = std::path::Path::new(&path).join("manifest.json");
        let mut m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        m["parameters"][0]["dtype"] = serde_json::json!("bf16");
        std::fs::write(&file, serde_json::to_vec_pretty(&m).unwrap()).unwrap();

        let mut into =
            open(fixtures::scaled_mean());
        let e = into.restore(&path).unwrap_err().to_string();
        assert!(e.contains("the manifest says bf16"), "{e}");
    }

    #[test]
    fn the_manifest_says_what_it_was_written_by_and_on() {
        let session = open(fixtures::scaled_mean());
        let (_dir, path) = written(&session, "");
        let m = manifest(&path);
        assert_eq!(m["schema_version"], 2);
        assert_eq!(m["semantics_version"], 3);
        assert_eq!(m["platform"]["os"], std::env::consts::OS);
        assert_eq!(m["platform"]["arch"], std::env::consts::ARCH);
        assert!(m["build"]["mlx_rs"].is_string(), "{m}");
        assert_eq!(m["optimizer"]["kind"], "sgd");
    }

    #[test]
    fn what_the_caller_records_comes_back_unchanged() {
        // Epoch, batch position and sampler state are not the engine's to
        // know, so they travel opaquely. What matters is that they survive.
        let run = r#"{"epoch":3,"batch":1200,"sampler":{"kind":"shuffled","seed":9}}"#;
        let mut session =
            open(fixtures::scaled_mean());
        session.run_step(&fixtures::batch_x(&[1.0, 2.0])).unwrap();
        let (_dir, path) = written(&session, run);

        let mut fresh =
            open(fixtures::scaled_mean());
        let back: serde_json::Value = serde_json::from_str(&fresh.restore(&path).unwrap()).unwrap();
        assert_eq!(back["epoch"], 3);
        assert_eq!(back["batch"], 1200);
        assert_eq!(back["sampler"]["seed"], 9);
        assert_eq!(fresh.step().unwrap(), 1);
    }

    #[test]
    fn a_run_record_that_is_not_json_is_refused_before_anything_is_written() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ckpt").display().to_string();
        let session = open(fixtures::scaled_mean());
        let e = session.save(&path, "not json").unwrap_err().to_string();
        assert!(e.contains("not JSON"), "{e}");
        assert!(!std::path::Path::new(&path).exists());
    }

    #[test]
    fn a_checkpoint_with_no_caller_record_restores_to_null() {
        let mut session =
            open(fixtures::scaled_mean());
        let (_dir, path) = written(&session, "");
        assert_eq!(session.restore(&path).unwrap(), "null");
    }

    #[test]
    fn a_checkpoint_of_an_older_schema_is_refused_rather_than_guessed_at() {
        let session = open(fixtures::scaled_mean());
        let (_dir, path) = written(&session, "");
        let file = std::path::Path::new(&path).join("manifest.json");
        let mut m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        m["schema_version"] = serde_json::json!(1);
        std::fs::write(&file, serde_json::to_vec_pretty(&m).unwrap()).unwrap();
        let e = crate::checkpoint::read_manifest_json(&path).unwrap_err().to_string();
        assert!(e.contains("schema 1 is not 2"), "{e}");
    }

    #[test]
    fn a_manifest_reads_without_a_session_to_read_it_into() {
        // What a caller consults to decide which checkpoint to resume.
        let mut session = open(fixtures::teacher_and_student());
        session.run_step(&fixtures::batch_x(&[1.0, 1.0])).unwrap();
        let (_dir, path) = written(&session, r#"{"epoch":2}"#);
        let m = manifest(&path);
        assert_eq!(m["step"], 1);
        assert_eq!(m["run"]["epoch"], 2);
        assert_eq!(m["seed"], 0);
    }

    #[test]
    fn a_session_exports_one_models_parameters_through_the_runtime() {
        // The facade path: the runtime serializes this against any step in
        // flight, and the model name is stripped from the saved keys.
        let session = open(fixtures::teacher_and_student());
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out").display().to_string();

        let pairs = session.export_model("student", &out).unwrap();
        assert_eq!(pairs, vec![("student.scale".to_string(), "scale".to_string())]);

        let arrays = mlx_rs::Array::load_safetensors(std::path::Path::new(&out).join("model.safetensors")).unwrap();
        assert!(arrays.contains_key("scale"));
        assert!(!arrays.contains_key("student.scale"));
        assert!(!arrays.contains_key("teacher.scale"), "only one model is exported");
    }

    #[test]
    fn a_session_refuses_to_export_a_model_that_is_not_here() {
        let session = open(fixtures::scaled_mean());
        let dir = tempfile::tempdir().unwrap();
        let e = session
            .export_model("elsewhere", &dir.path().display().to_string())
            .unwrap_err();
        assert!(e.to_string().contains("no model named"), "{e}");
    }
}
