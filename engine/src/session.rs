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

    /// Writes the run's state: parameters, optimizer slots, counters, and
    /// what they belong to. Atomic (docs/plan.md section 11.2).
    pub fn save(&self, dir: &str) -> Result<String> {
        self.state.save(&self.plan, dir)
    }

    /// Restores state written by [`Session::save`], refusing anything that
    /// does not belong to this session.
    pub fn restore(&mut self, dir: &str) -> Result<()> {
        self.state.restore(&self.plan, dir)
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

    fn session(which: (String, String)) -> Session {
        let (config, weights) = which;
        Session::open(&config, &weights).unwrap()
    }

    fn values(t: &Tensor) -> Vec<f32> {
        match &t.values {
            Values::F32(v) => v.clone(),
            Values::I32(v) => v.iter().map(|&i| i as f32).collect(),
        }
    }

    fn close(got: &[f32], want: &[f32]) {
        within(got, want, 1e-6);
    }

    fn within(got: &[f32], want: &[f32], tolerance: f32) {
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
        let written = session.save(&path).unwrap();
        assert!(std::path::Path::new(&written).join("manifest.json").exists());
        let want = values(&session.fetch("m.w").unwrap());

        let mut fresh = self::session(fixtures::scaled_mean());
        fresh.restore(&path).unwrap();
        assert_eq!(fresh.step(), 1);
        close(&values(&fresh.fetch("m.w").unwrap()), &want);
    }
}
