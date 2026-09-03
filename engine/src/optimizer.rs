//! The optimizers, owned here rather than taken from mlx-rs.
//!
//! This is the one piece of the engine the plan says to keep under our own
//! tests (docs/plan.md section 5): an optimizer is small, it is where a
//! silent mistake hides best, and its state is half of what a checkpoint
//! must reproduce. The arithmetic is MLX's; the update rule is ours.

use anyhow::Result;
use mlx_rs::ops::zeros_like;
use mlx_rs::Array;
use serde::{Deserialize, Serialize};

/// How an optimizer is configured, as it appears in a journal, a manifest
/// and a checkpoint. Data, so that a run can say what it used.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Config {
    Sgd {
        lr: f32,
    },
    /// Decoupled weight decay, as in the paper and as PyTorch's AdamW
    /// implements it: the decay is applied to the parameter, not to the
    /// gradient, so it does not enter the moment estimates.
    #[serde(rename = "adamw")]
    AdamW {
        lr: f32,
        #[serde(default = "default_beta1")]
        beta1: f32,
        #[serde(default = "default_beta2")]
        beta2: f32,
        #[serde(default = "default_eps")]
        eps: f32,
        #[serde(default)]
        weight_decay: f32,
    },
}

fn default_beta1() -> f32 {
    0.9
}
fn default_beta2() -> f32 {
    0.999
}
fn default_eps() -> f32 {
    1e-8
}

impl Config {
    pub fn lr(&self) -> f32 {
        match self {
            Config::Sgd { lr } | Config::AdamW { lr, .. } => *lr,
        }
    }

    pub fn set_lr(&mut self, value: f32) {
        match self {
            Config::Sgd { lr } | Config::AdamW { lr, .. } => *lr = value,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Config::Sgd { .. } => "sgd",
            Config::AdamW { .. } => "adamw",
        }
    }
}

/// An optimizer and its slots. One slot set per differentiated parameter,
/// in the order the config declared them.
#[derive(Clone)]
pub struct Optimizer {
    config: Config,
    /// AdamW's first and second moments. Empty for SGD.
    m: Vec<Array>,
    v: Vec<Array>,
    /// Steps taken, for the bias correction. Part of the state a checkpoint
    /// restores: resuming with t = 0 would take a wrong first step.
    t: u64,
}

impl Optimizer {
    pub fn new(config: Config, params: &[Array], argnums: &[i32]) -> Result<Self> {
        let slots = || -> Result<Vec<Array>> {
            argnums
                .iter()
                .map(|&i| Ok(zeros_like(&params[i as usize])?))
                .collect()
        };
        let (m, v) = match config {
            Config::Sgd { .. } => (Vec::new(), Vec::new()),
            Config::AdamW { .. } => (slots()?, slots()?),
        };
        Ok(Self { config, m, v, t: 0 })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut Config {
        &mut self.config
    }

    pub fn steps_taken(&self) -> u64 {
        self.t
    }

    /// The moments, for a checkpoint. Empty for SGD, which has none.
    pub fn slots(&self) -> (&[Array], &[Array]) {
        (&self.m, &self.v)
    }

    /// Moves the slots to follow a change in what is differentiated:
    /// kept for parameters that stay, dropped for those that freeze,
    /// started at zero for those that thaw.
    ///
    /// The step count is not reset. A thawed parameter joins a run in
    /// progress, and its bias correction is that run's, not a new one's.
    pub fn refit(&mut self, was: &[i32], now: &[i32], params: &[Array]) -> Result<()> {
        if !self.wants_slots() {
            return Ok(());
        }
        anyhow::ensure!(
            self.m.len() == was.len(),
            "the optimizer has {} slots for {} parameters",
            self.m.len(),
            was.len()
        );
        let mut m = Vec::with_capacity(now.len());
        let mut v = Vec::with_capacity(now.len());
        for &i in now {
            match was.iter().position(|&w| w == i) {
                Some(slot) => {
                    m.push(self.m[slot].clone());
                    v.push(self.v[slot].clone());
                }
                None => {
                    m.push(zeros_like(&params[i as usize])?);
                    v.push(zeros_like(&params[i as usize])?);
                }
            }
        }
        self.m = m;
        self.v = v;
        Ok(())
    }

    /// Restores state a checkpoint held. The shapes are the caller's to
    /// have checked against the parameters.
    pub fn restore(&mut self, m: Vec<Array>, v: Vec<Array>, t: u64) {
        self.m = m;
        self.v = v;
        self.t = t;
    }

    /// Whether this rule keeps per-parameter state. A checkpoint that
    /// disagrees with this is refused rather than restored into a shape
    /// the next step would index past.
    pub fn wants_slots(&self) -> bool {
        matches!(self.config, Config::AdamW { .. })
    }

    /// The optimizer after one update, and `params` written in place.
    /// Returns a new value rather than mutating, so the caller can evaluate
    /// everything before committing to it (see `Session::update`).
    pub fn next(&self, params: &mut [Array], argnums: &[i32], grads: &[Array]) -> Result<Self> {
        let mut next = self.clone();
        next.apply(params, argnums, grads)?;
        Ok(next)
    }

    /// Applies one update in place. `grads` is parallel to `argnums`.
    fn apply(&mut self, params: &mut [Array], argnums: &[i32], grads: &[Array]) -> Result<()> {
        anyhow::ensure!(
            argnums.len() == grads.len(),
            "{} gradients for {} differentiated parameters",
            grads.len(),
            argnums.len()
        );
        if self.wants_slots() {
            anyhow::ensure!(
                self.m.len() == argnums.len() && self.v.len() == argnums.len(),
                "the optimizer has {} slots for {} differentiated parameters; \
                 this state did not come from this session",
                self.m.len(),
                argnums.len()
            );
        }
        self.t += 1;
        match self.config {
            Config::Sgd { lr } => {
                let lr = Array::from_f32(lr);
                for (&i, grad) in argnums.iter().zip(grads) {
                    let i = i as usize;
                    params[i] = params[i].subtract(grad.multiply(&lr)?)?;
                }
            }
            Config::AdamW {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            } => {
                // Bias correction from the step count, so a resumed run
                // takes the step it would have taken.
                let t = self.t as f32;
                let bias1 = 1.0 - beta1.powf(t);
                let bias2 = 1.0 - beta2.powf(t);
                let (b1, b2) = (Array::from_f32(beta1), Array::from_f32(beta2));
                let (one_b1, one_b2) = (
                    Array::from_f32(1.0 - beta1),
                    Array::from_f32(1.0 - beta2),
                );
                let lr_a = Array::from_f32(lr);
                let eps_a = Array::from_f32(eps);
                let bias1_a = Array::from_f32(bias1);
                let bias2_a = Array::from_f32(bias2);

                for (slot, (&i, grad)) in argnums.iter().zip(grads).enumerate() {
                    let i = i as usize;
                    self.m[slot] = self.m[slot]
                        .multiply(&b1)?
                        .add(grad.multiply(&one_b1)?)?;
                    self.v[slot] = self.v[slot]
                        .multiply(&b2)?
                        .add(grad.square()?.multiply(&one_b2)?)?;

                    let m_hat = self.m[slot].divide(&bias1_a)?;
                    let v_hat = self.v[slot].divide(&bias2_a)?;
                    let denom = v_hat.sqrt()?.add(&eps_a)?;
                    let mut next = params[i].subtract(m_hat.divide(&denom)?.multiply(&lr_a)?)?;
                    if weight_decay != 0.0 {
                        // Decoupled: applied to the parameter, after the
                        // Adam step, and never through the moments.
                        let decay = Array::from_f32(lr * weight_decay);
                        next = next.subtract(params[i].multiply(&decay)?)?;
                    }
                    params[i] = next;
                }
            }
        }
        Ok(())
    }
}
