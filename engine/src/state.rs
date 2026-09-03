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
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use crate::checkpoint;
use crate::optimizer::{Config as OptimizerConfig, Optimizer};
use crate::plan::{Pattern, Plan};
use crate::tensor::{to_tensor, Tensor};

pub struct TrainState {
    /// Every parameter of every model, model by model in name order. This
    /// order is `Plan::paths`, and the contract the Ruby side follows.
    pub params: Vec<Array>,
    /// Positions in `params` that autodiff differentiates now. A subset of
    /// `Plan::candidates`; moves when the window freezes or unfreezes.
    pub argnums: Vec<i32>,
    /// The update rule and its slots. Half of what a checkpoint restores.
    optimizer: Optimizer,
    /// The RNG, held as state rather than left to a global. Every step
    /// splits it, so the sequence of draws is a function of the seed and
    /// the step count, and a resumed run draws what a continuous one would
    /// (docs/plan.md section 11.1).
    pub rng: Array,
    seed: u64,
    step: usize,
    last_loss: f32,
}

impl TrainState {
    /// The state a plan starts in: the given parameters, everything the
    /// plan declared trainable differentiated, and fresh optimizer slots.
    pub fn new(plan: &Plan, params: Vec<Array>, optimizer: OptimizerConfig) -> Result<Self> {
        let argnums = plan.candidates.clone();
        let optimizer = Optimizer::new(optimizer, &params, &argnums)?;
        let seed = 0;
        Ok(Self {
            params,
            argnums,
            optimizer,
            rng: mlx_rs::random::key(seed)?,
            seed,
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
    pub fn advance(&mut self, loss: &Array, grads: &[Array]) -> Result<f32> {
        let mut params = self.params.clone();
        let next_optimizer = self.optimizer.next(&mut params, &self.argnums, grads)?;
        let (rng, _) = mlx_rs::random::split(&self.rng, 2)?;

        let (m, v) = next_optimizer.slots();
        eval(
            params
                .iter()
                .chain(m.iter())
                .chain(v.iter())
                .chain(std::iter::once(&rng))
                .chain(std::iter::once(loss)),
        )?;

        self.params = params;
        self.optimizer = next_optimizer;
        self.rng = rng;
        self.last_loss = loss.item::<f32>();
        self.step += 1;
        Ok(self.last_loss)
    }

    /// Writes the run's state: parameters, optimizer slots, counters, and
    /// what they belong to. Atomic (docs/plan.md section 11.2).
    pub fn save(&self, plan: &Plan, dir: &str) -> Result<String> {
        let (m, v) = self.optimizer.slots();
        let state = checkpoint::State {
            config_digest: &plan.config_digest,
            step: self.step,
            optimizer: self.optimizer.config(),
            optimizer_steps: self.optimizer.steps_taken(),
            parameters: plan.paths.iter().cloned().zip(self.params.iter()).collect(),
            argnums: &self.argnums,
            slots: (m, v),
            rng: &self.rng,
            seed: self.seed,
        };
        Ok(checkpoint::write(dir, state)?.display().to_string())
    }

    /// Restores state written by [`TrainState::save`], refusing anything
    /// that does not belong to this run: another description, another
    /// optimizer, a parameter of another shape, a missing slot.
    ///
    /// Nothing is committed until everything has been read, checked and
    /// evaluated. A checkpoint that turns out to be wrong halfway leaves
    /// this run exactly as it was.
    pub fn restore(&mut self, plan: &Plan, dir: &str) -> Result<()> {
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
        } else {
            anyhow::ensure!(
                loaded.slots.is_empty(),
                "this checkpoint carries optimizer state, and {} has none",
                self.optimizer.config().name()
            );
        }

        let rng = loaded.rng.context("the checkpoint has no RNG state")?;

        // Everything is here and consistent; make it real before touching
        // this state, so a failure below cannot leave it half restored.
        eval(
            params
                .iter()
                .chain(m.iter())
                .chain(v.iter())
                .chain(std::iter::once(&rng)),
        )?;

        self.params = params;
        self.optimizer.restore(m, v, manifest.optimizer_steps);
        self.rng = rng;
        self.seed = manifest.seed;
        self.step = manifest.step;
        self.last_loss = f32::NAN;
        Ok(())
    }
}
