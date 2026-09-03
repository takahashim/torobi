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

use std::collections::BTreeMap;

use anyhow::{Context, Result};

use crate::executor::{self, Taps};
use crate::interp::Stat;
use crate::optimizer::Config as OptimizerConfig;
use crate::plan::Plan;
use crate::state::TrainState;
use crate::tensor::{to_tensor, Batch, Tensor, Values};

pub struct Session {
    /// What this run does, settled when it opened.
    plan: Plan,
    /// What it has accumulated since.
    state: TrainState,
    /// What the window is watching: node name to how much of it to bring
    /// back. Read-only; changing the set changes what a step evaluates,
    /// so it takes effect from the next one.
    taps: Taps,
    /// The last step's tapped values, as copies.
    tapped: BTreeMap<String, Tensor>,
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
        let (plan, params) = Plan::open(graph_json, weights_json)?;
        let state = TrainState::new(&plan, params, optimizer)?;
        Ok(Self {
            plan,
            state,
            taps: Taps::new(),
            tapped: BTreeMap::new(),
        })
    }

    pub fn step(&self) -> usize {
        self.state.step()
    }

    pub fn loss(&self) -> f32 {
        self.state.loss()
    }

    pub fn lr(&self) -> f32 {
        self.state.lr()
    }

    /// A knob: effect begins with the next step.
    pub fn set_lr(&mut self, lr: f32) {
        self.state.set_lr(lr);
    }

    /// What update rule this session runs, as data.
    pub fn optimizer_config(&self) -> &OptimizerConfig {
        self.state.optimizer_config()
    }

    pub fn seed(&self) -> u64 {
        self.state.seed()
    }

    /// Restarts the RNG. A knob like any other: after this the draws are a
    /// function of the new seed alone.
    pub fn set_seed(&mut self, seed: u64) -> Result<()> {
        self.state.set_seed(seed)
    }

    /// Watches a named node. `stat` is "full", "mean", "norm" or "extent";
    /// a reduction costs a scalar per step where "full" costs the tensor,
    /// which is why a standing tap should reduce.
    pub fn tap(&mut self, name: &str, stat: &str) -> Result<()> {
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
    pub fn untap(&mut self, name: &str) -> bool {
        self.taps.remove(name).is_some()
    }

    /// What is being watched.
    pub fn taps(&self) -> Vec<String> {
        self.taps.keys().cloned().collect()
    }

    /// Every name a tap could ask for.
    pub fn node_names(&self) -> Vec<String> {
        self.plan.node_names()
    }

    /// What the last step's taps saw, by name.
    pub fn tapped(&self) -> Result<Vec<(String, Tensor)>> {
        self.tapped
            .iter()
            .map(|(name, tensor)| {
                Ok((
                    name.clone(),
                    Tensor {
                        dtype: tensor.dtype,
                        shape: tensor.shape.clone(),
                        values: match &tensor.values {
                            Values::F32(v) => Values::F32(v.clone()),
                            Values::I32(v) => Values::I32(v.clone()),
                        },
                    },
                ))
            })
            .collect()
    }

    /// Which parameters are currently differentiated, by qualified path.
    pub fn trainable(&self) -> Vec<String> {
        self.plan.paths_of(&self.state.argnums)
    }

    /// Every parameter a model declared trainable, whether or not it is
    /// frozen right now: the set `freeze` and `unfreeze` move within.
    pub fn trainable_candidates(&self) -> Vec<String> {
        self.plan.candidate_paths()
    }

    /// Freezes or unfreezes parameters whose path matches `pattern`, and
    /// returns those that moved.
    pub fn set_frozen(&mut self, pattern: &str, frozen: bool) -> Result<Vec<String>> {
        self.state.set_frozen(&self.plan, pattern, frozen)
    }

    /// Writes one parameter, by qualified path, from a copy. The window's
    /// B capability (docs/plan.md section 8.3).
    pub fn put(&mut self, path: &str, tensor: &Tensor) -> Result<()> {
        self.state.put(&self.plan, path, tensor)
    }

    /// One step on `batch`: forward, backward, optimizer update. Long-
    /// running and free of any Ruby, so the extension calls it with the GVL
    /// released.
    pub fn run_step(&mut self, batch: &Batch) -> Result<f32> {
        let fields = self.plan.bind(batch)?;
        self.update(&fields)
    }

    /// One step per batch. The batches are given up front, so the engine
    /// never asks anyone for data mid-span.
    pub fn run_steps(&mut self, batches: &[Batch]) -> Result<f32> {
        anyhow::ensure!(!batches.is_empty(), "a span needs at least one batch");
        for batch in batches {
            let fields = self.plan.bind(batch)?;
            self.update(&fields)?;
        }
        Ok(self.loss())
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
    pub fn evaluate(&mut self, batch: &Batch) -> Result<f32> {
        let fields = self.plan.bind(batch)?;
        let (loss, tapped) =
            executor::evaluate(&self.plan, &self.state.params, &fields, &self.taps)?;
        self.tapped = tapped
            .iter()
            .map(|(name, value)| Ok((name.clone(), to_tensor(value)?)))
            .collect::<Result<_>>()?;
        Ok(loss.item::<f32>())
    }

    /// The loss and one gradient per differentiated parameter for `batch`,
    /// without updating anything.
    pub fn loss_and_grads(&self, batch: &Batch) -> Result<(mlx_rs::Array, Vec<mlx_rs::Array>)> {
        let fields = self.plan.bind(batch)?;
        let (loss, grads, _) = executor::differentiate(
            &self.plan,
            &self.state.params,
            &self.state.argnums,
            &fields,
            &self.state.rng,
            &Taps::new(),
        )?;
        Ok((loss, grads))
    }

    /// Gradients as copies, by qualified parameter path. Only differentiated
    /// parameters appear: a frozen model's have none.
    pub fn gradients(&self, batch: &Batch) -> Result<Vec<(String, Tensor)>> {
        let (_, grads) = self.loss_and_grads(batch)?;
        self.trainable()
            .into_iter()
            .zip(grads)
            .map(|(path, grad)| Ok((path, to_tensor(&grad)?)))
            .collect()
    }

    /// A copy of one parameter, by qualified path. Copies, not handles:
    /// nothing that lives on the device escapes this crate.
    pub fn fetch(&self, path: &str) -> Result<Tensor> {
        self.state.fetch(&self.plan, path)
    }

    /// Qualified parameter paths, in the order the engine keeps them.
    pub fn parameter_paths(&self) -> Vec<String> {
        self.plan.paths.clone()
    }

    /// Every batch field the run reads, across the models and the objective.
    pub fn input_names(&self) -> Vec<String> {
        self.plan.input_names()
    }

    /// Writes the run's state and the description it belongs to. Atomic
    /// (docs/plan.md section 11.2). `run` is the caller's own record
    /// (epoch, batch position, sampler state), written verbatim.
    pub fn save(&self, dir: &str, run: &str) -> Result<String> {
        self.state.save(&self.plan, dir, run)
    }

    /// Restores state written by [`Session::save`], refusing anything that
    /// does not belong to this session. Returns the caller's record.
    pub fn restore(&mut self, dir: &str) -> Result<String> {
        self.state.restore(&self.plan, dir)
    }

    /// What a checkpoint says about itself, without opening it into a
    /// session: for a caller deciding which one to resume from.
    pub fn read_manifest(dir: &str) -> Result<String> {
        Ok(serde_json::to_string(&crate::checkpoint::read_manifest(dir)?)?)
    }

    /// One step over already-bound inputs: differentiate, read the taps,
    /// then commit. The taps are converted before the state moves so that
    /// a step either happens whole or not at all.
    fn update(&mut self, fields: &BTreeMap<String, mlx_rs::Array>) -> Result<f32> {
        let (loss, grads, tapped) = executor::differentiate(
            &self.plan,
            &self.state.params,
            &self.state.argnums,
            fields,
            &self.state.rng,
            &self.taps,
        )?;
        let tapped: BTreeMap<String, Tensor> = tapped
            .iter()
            .map(|(name, value)| Ok((name.clone(), to_tensor(value)?)))
            .collect::<Result<_>>()?;
        let loss = self.state.advance(&loss, &grads)?;
        self.tapped = tapped;
        Ok(loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;
    use crate::tensor::Values;

    pub fn session(which: (String, String)) -> Session {
        let (config, weights) = which;
        Session::open(&config, &weights).unwrap()
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
        let session = session(fixtures::scaled_mean());
        let batch = fixtures::batch_x(&[1.0, 2.0, 3.0, 4.0]);
        let grads = session.gradients(&batch).unwrap();
        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, "m.w");
        close(&values(&grads[0].1), &[(1.0 + 3.0) / 4.0, (2.0 + 4.0) / 4.0]);

        let (loss, _) = session.loss_and_grads(&batch).unwrap();
        // w = [1, 2], so x * w is [[1, 4], [3, 8]] and the mean is 4.
        assert!((loss.item::<f32>() - 4.0).abs() < 1e-6);
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
            session.parameter_paths(),
            vec!["student.scale", "teacher.scale"]
        );
        assert_eq!(session.trainable(), vec!["student.scale"]);
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
        session.set_lr(0.05);
        let batch = fixtures::batch_x(&[1.0, 1.0, 2.0, 2.0]);
        let first = session.run_step(&batch).unwrap();
        for _ in 0..50 {
            session.run_step(&batch).unwrap();
        }
        assert_eq!(session.step(), 51);
        assert!(session.loss() < first * 0.1, "{} -> {}", first, session.loss());
        // It converged on the teacher's scale, which is what the objective
        // asks for. Fifty steps of plain SGD get three digits, not six.
        within(&values(&session.fetch("student.scale").unwrap()), &[3.0, 4.0], 1e-2);
        // And the teacher did not move.
        close(&values(&session.fetch("teacher.scale").unwrap()), &[3.0, 4.0]);
    }

    #[test]
    fn a_span_needs_at_least_one_batch() {
        let mut session = session(fixtures::scaled_mean());
        let e = session.run_steps(&[]).unwrap_err().to_string();
        assert!(e.contains("at least one batch"), "{e}");
        assert_eq!(session.step(), 0);
    }

    #[test]
    fn a_span_takes_one_step_per_batch() {
        let mut session = session(fixtures::scaled_mean());
        let batches = vec![
            fixtures::batch_x(&[1.0, 1.0]),
            fixtures::batch_x(&[2.0, 2.0, 3.0, 3.0]),
        ];
        let loss = session.run_steps(&batches).unwrap();
        assert_eq!(session.step(), 2);
        assert_eq!(loss, session.loss());
    }

    #[test]
    fn a_tap_reports_what_the_step_computed() {
        let mut session = session(fixtures::scaled_mean());
        assert_eq!(session.node_names(), vec!["scaled"]);
        assert!(session.tapped().unwrap().is_empty());

        session.tap("scaled", "mean").unwrap();
        assert_eq!(session.taps(), vec!["scaled"]);
        // x * w with x = [[1, 1]] and w = [1, 2] is [[1, 2]], mean 1.5.
        session.run_step(&fixtures::batch_x(&[1.0, 1.0])).unwrap();
        let seen = session.tapped().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, "scaled");
        close(&values(&seen[0].1), &[1.5]);

        assert!(session.untap("scaled"));
        assert!(!session.untap("scaled"));
        assert!(session.taps().is_empty());
    }

    #[test]
    fn a_full_tap_brings_back_the_tensor_and_a_reduction_a_scalar() {
        let mut session = session(fixtures::scaled_mean());
        session.tap("scaled", "full").unwrap();
        session.run_step(&fixtures::batch_x(&[1.0, 1.0])).unwrap();
        let seen = session.tapped().unwrap();
        assert_eq!(seen[0].1.shape, vec![1, 2]);
        close(&values(&seen[0].1), &[1.0, 2.0]);
    }

    #[test]
    fn a_tap_on_a_name_no_node_carries_is_refused() {
        let mut session = session(fixtures::scaled_mean());
        let e = session.tap("nowhere", "mean").unwrap_err().to_string();
        assert!(e.contains("no value is named"), "{e}");
        let e = session.tap("scaled", "median").unwrap_err().to_string();
        assert!(e.contains("is not a statistic"), "{e}");
        assert!(session.taps().is_empty());
    }

    #[test]
    fn taps_do_not_change_what_a_step_computes() {
        let batch = fixtures::batch_x(&[1.0, 2.0, 3.0, 4.0]);
        let mut quiet = session(fixtures::scaled_mean());
        let mut watched = session(fixtures::scaled_mean());
        watched.tap("scaled", "norm").unwrap();
        for _ in 0..3 {
            quiet.run_step(&batch).unwrap();
            watched.run_step(&batch).unwrap();
        }
        assert_eq!(quiet.loss(), watched.loss());
        close(
            &values(&watched.fetch("m.w").unwrap()),
            &values(&quiet.fetch("m.w").unwrap()),
        );
    }

    #[test]
    fn the_session_reports_what_it_reads_and_what_it_holds() {
        let session = session(fixtures::teacher_and_student());
        assert_eq!(session.input_names(), vec!["x"]);
        assert_eq!(session.trainable_candidates(), vec!["student.scale"]);
        assert_eq!(session.optimizer_config().name(), "sgd");
        assert_eq!(session.seed(), 0);
        assert!(session.loss().is_nan());
    }

    #[test]
    fn freezing_and_thawing_move_what_a_step_reports() {
        let (config, weights) = fixtures::teacher_and_student();
        // Train both, so there is something to freeze that leaves a rest.
        let both = config.replace(r#""train":["student"]"#, r#""train":["student","teacher"]"#);
        let mut session = Session::open(&both, &weights).unwrap();
        assert_eq!(
            session.trainable(),
            vec!["student.scale", "teacher.scale"]
        );
        let moved = session.set_frozen("teacher.*", true).unwrap();
        assert_eq!(moved, vec!["teacher.scale"]);
        assert_eq!(session.trainable(), vec!["student.scale"]);

        let grads = session.gradients(&fixtures::batch_x(&[1.0, 1.0])).unwrap();
        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, "student.scale");

        assert_eq!(session.set_frozen("teacher.*", false).unwrap(), vec!["teacher.scale"]);
        assert_eq!(session.trainable(), vec!["student.scale", "teacher.scale"]);
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
        assert_eq!(fresh.step(), 1);
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
        let mut session = Session::open(&config, &weights).unwrap();
        let batch = fixtures::batch_x(&[1.0, 2.0]);

        let loss = session.run_step(&batch).unwrap();
        assert!(!loss.is_finite(), "the fixture should divide by zero");

        // Reported, so a policy sees it. Not applied, so nothing is lost.
        assert!(session.loss().is_nan() || session.loss().is_infinite());
        close(&values(&session.fetch("m.w").unwrap()), &[0.0, 0.0]);
        // The counter still moves: a step was attempted and its batch
        // consumed, and a hook that fires every N steps must keep firing.
        assert_eq!(session.step(), 1);
    }

    #[test]
    fn a_run_recovers_once_the_parameters_are_usable_again() {
        // What a NaN policy does now: nothing was corrupted, so putting the
        // parameter somewhere sane is enough. No checkpoint is needed.
        let (config, weights) = fixtures::divides_by_zero();
        let mut session = Session::open(&config, &weights).unwrap();
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
        assert_eq!(session.step(), 2);
    }

    #[test]
    fn the_rng_moves_even_on_a_step_that_was_not_taken() {
        // The forward drew from it, so the sequence belongs to the step
        // count exactly as it would have. A resumed run has to agree.
        let (config, weights) = fixtures::divides_by_zero();
        let mut session = Session::open(&config, &weights).unwrap();
        let before = crate::tensor::to_tensor(&session.state.rng).unwrap();
        session.run_step(&fixtures::batch_x(&[1.0, 2.0])).unwrap();
        let after = crate::tensor::to_tensor(&session.state.rng).unwrap();

        assert_ne!(values(&before), values(&after));
    }

    #[test]
    fn evaluating_moves_nothing() {
        let (config, weights) = fixtures::scaled_mean();
        let mut session = Session::open(&config, &weights).unwrap();
        let batch = fixtures::batch_x(&[1.0, 2.0, 3.0, 4.0]);
        session.run_step(&batch).unwrap();
        let (step, loss) = (session.step(), session.loss());
        let w = values(&session.fetch("m.w").unwrap());
        let rng = values(&crate::tensor::to_tensor(&session.state.rng).unwrap());

        let seen = session.evaluate(&batch).unwrap();
        assert!(seen.is_finite());
        assert_eq!(session.step(), step);
        assert_eq!(session.loss(), loss, "the training loss is not an evaluation");
        close(&values(&session.fetch("m.w").unwrap()), &w);
        assert_eq!(
            values(&crate::tensor::to_tensor(&session.state.rng).unwrap()),
            rng
        );
    }

    #[test]
    fn evaluating_reports_the_loss_the_parameters_give() {
        // mean(x * w) with w = [1, 2] over [[1, 2], [3, 4]] is
        // (1 + 4 + 3 + 8) / 4 = 4.
        let (config, weights) = fixtures::scaled_mean();
        let mut session = Session::open(&config, &weights).unwrap();
        let seen = session.evaluate(&fixtures::batch_x(&[1.0, 2.0, 3.0, 4.0])).unwrap();
        assert!((seen - 4.0).abs() < 1e-6, "{seen}");
    }

    #[test]
    fn evaluating_does_not_sample_dropout() {
        let (config, weights) = fixtures::with_dropout(0.5);
        let mut session = Session::open(&config, &weights).unwrap();
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
        let mut session = Session::open(&config, &weights).unwrap();
        session.tap("scaled", "mean").unwrap();
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
        Session::open(&which.0, &which.1).unwrap()
    }

    fn written(session: &Session, run: &str) -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ckpt").display().to_string();
        session.save(&path, run).unwrap();
        (dir, path)
    }

    fn manifest(path: &str) -> serde_json::Value {
        serde_json::from_str(&Session::read_manifest(path).unwrap()).unwrap()
    }

    #[test]
    fn a_checkpoint_carries_the_description_and_not_only_its_digest() {
        let (config, weights) = fixtures::scaled_mean();
        let session = Session::open(&config, &weights).unwrap();
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
        assert_eq!(fresh.step(), 1);
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
        let e = Session::read_manifest(&path).unwrap_err().to_string();
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
}
