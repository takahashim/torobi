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

/// The rule and how much state it is carrying, never the slots
/// themselves: a moment is the size of the parameter it belongs to.
impl std::fmt::Debug for Optimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Optimizer({}, {} slots, {} steps)",
            self.config.name(),
            self.m.len(),
            self.t
        )
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::eval;

    fn array(values: &[f32]) -> Array {
        Array::from_slice(values, &[values.len() as i32])
    }

    fn read(array: &Array) -> Vec<f32> {
        eval(std::iter::once(array)).unwrap();
        array.as_slice::<f32>().to_vec()
    }

    fn close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{got:?} against {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{got:?} against {want:?}");
        }
    }

    const ADAMW: Config = Config::AdamW {
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };

    #[test]
    fn a_config_names_itself_the_way_a_manifest_reads_it() {
        // serde would have called this "adam_w"; a checkpoint written by
        // one spelling and read by the other would be refused as foreign.
        let json = serde_json::to_string(&ADAMW).unwrap();
        assert!(json.contains(r#""kind":"adamw""#), "{json}");
        let back: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ADAMW);
    }

    #[test]
    fn adamw_defaults_are_filled_in_from_a_bare_config() {
        let back: Config = serde_json::from_str(r#"{"kind":"adamw","lr":0.5}"#).unwrap();
        assert_eq!(
            back,
            Config::AdamW {
                lr: 0.5,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            }
        );
    }

    #[test]
    fn sgd_subtracts_the_gradient_scaled_by_the_rate() {
        let opt = Optimizer::new(Config::Sgd { lr: 0.5 }, &[array(&[1.0, 2.0])], &[0]).unwrap();
        let mut params = vec![array(&[1.0, 2.0])];
        let next = opt.next(&mut params, &[0], &[array(&[2.0, 4.0])]).unwrap();
        close(&read(&params[0]), &[0.0, 0.0]);
        assert_eq!(next.steps_taken(), 1);
        // The original is untouched: a step that fails must leave the run
        // as it was, which is why next() returns rather than mutates.
        assert_eq!(opt.steps_taken(), 0);
    }

    #[test]
    fn adamws_first_step_is_the_rate_regardless_of_the_gradients_size() {
        // With t = 1 the bias correction cancels the moments, so the step
        // is lr * g / (|g| + eps): the sign of the gradient, times lr.
        // This is the property that catches a missing correction.
        let params0 = vec![array(&[0.0, 0.0])];
        let opt = Optimizer::new(ADAMW, &params0, &[0]).unwrap();
        let mut params = params0.clone();
        opt.next(&mut params, &[0], &[array(&[1e-3, -50.0])]).unwrap();
        close(&read(&params[0]), &[-0.1, 0.1]);
    }

    #[test]
    fn adamw_carries_its_step_count_so_a_resumed_run_steps_the_same() {
        let params0 = vec![array(&[0.0])];
        let grads = [array(&[1.0])];

        let mut straight = vec![array(&[0.0])];
        let mut opt = Optimizer::new(ADAMW, &params0, &[0]).unwrap();
        for _ in 0..3 {
            opt = opt.next(&mut straight, &[0], &grads).unwrap();
        }

        // The same three steps, with the state carried across a restore.
        let mut resumed = vec![array(&[0.0])];
        let mut a = Optimizer::new(ADAMW, &params0, &[0]).unwrap();
        a = a.next(&mut resumed, &[0], &grads).unwrap();
        let (m, v) = a.slots();
        let (m, v) = (m.to_vec(), v.to_vec());
        let t = a.steps_taken();
        let mut b = Optimizer::new(ADAMW, &params0, &[0]).unwrap();
        b.restore(m, v, t);
        for _ in 0..2 {
            b = b.next(&mut resumed, &[0], &grads).unwrap();
        }

        assert_eq!(b.steps_taken(), opt.steps_taken());
        close(&read(&resumed[0]), &read(&straight[0]));
    }

    #[test]
    fn a_forgotten_step_count_would_take_a_different_step() {
        // Pins that the assertion above has teeth: restoring t = 0 does
        // not land where a continuous run lands.
        let params0 = vec![array(&[0.0])];
        let grads = [array(&[1.0])];
        let mut straight = vec![array(&[0.0])];
        let mut opt = Optimizer::new(ADAMW, &params0, &[0]).unwrap();
        for _ in 0..2 {
            opt = opt.next(&mut straight, &[0], &grads).unwrap();
        }

        let mut wrong = vec![array(&[0.0])];
        let mut a = Optimizer::new(ADAMW, &params0, &[0]).unwrap();
        a = a.next(&mut wrong, &[0], &grads).unwrap();
        let (m, v) = a.slots();
        let (m, v) = (m.to_vec(), v.to_vec());
        let mut b = Optimizer::new(ADAMW, &params0, &[0]).unwrap();
        b.restore(m, v, 0);
        b.next(&mut wrong, &[0], &grads).unwrap();

        let (want, got) = (read(&straight[0]), read(&wrong[0]));
        assert!((want[0] - got[0]).abs() > 1e-4, "{want:?} against {got:?}");
    }

    #[test]
    fn decoupled_decay_shrinks_the_parameter_and_not_the_moments() {
        let decayed = Config::AdamW {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.5,
        };
        let params0 = vec![array(&[2.0])];
        let opt = Optimizer::new(decayed, &params0, &[0]).unwrap();
        let mut params = params0.clone();
        let next = opt.next(&mut params, &[0], &[array(&[1.0])]).unwrap();
        // Adam's own step is lr, and the decay is lr * wd * param.
        close(&read(&params[0]), &[2.0 - 0.1 - 0.1 * 0.5 * 2.0]);
        // The moment saw the gradient alone: 0.1 * 1.0.
        close(&read(&next.slots().0[0]), &[0.1]);
    }

    #[test]
    fn refit_keeps_what_stays_zeroes_what_thaws_and_drops_what_freezes() {
        let params = vec![array(&[0.0]), array(&[0.0]), array(&[0.0])];
        let mut opt = Optimizer::new(ADAMW, &params, &[0, 1]).unwrap();
        let mut copy = params.clone();
        opt = opt
            .next(&mut copy, &[0, 1], &[array(&[1.0]), array(&[2.0])])
            .unwrap();
        let before: Vec<Vec<f32>> = opt.slots().0.iter().map(read).collect();
        assert_eq!(before.len(), 2);

        // Freeze 0, thaw 2: parameter 1's slot follows it to the front.
        opt.refit(&[0, 1], &[1, 2], &params).unwrap();
        let after: Vec<Vec<f32>> = opt.slots().0.iter().map(read).collect();
        assert_eq!(after.len(), 2);
        close(&after[0], &before[1]);
        close(&after[1], &[0.0]);
        // A thawed parameter joins a run in progress; the count is the
        // run's, so its bias correction is too.
        assert_eq!(opt.steps_taken(), 1);
    }

    #[test]
    fn refit_is_nothing_for_an_optimizer_with_no_slots() {
        let params = vec![array(&[0.0]), array(&[0.0])];
        let mut opt = Optimizer::new(Config::Sgd { lr: 0.1 }, &params, &[0]).unwrap();
        opt.refit(&[0], &[0, 1], &params).unwrap();
        assert!(opt.slots().0.is_empty());
        assert!(!opt.wants_slots());
    }

    #[test]
    fn a_gradient_count_that_does_not_match_is_refused() {
        let params = vec![array(&[0.0]), array(&[0.0])];
        let opt = Optimizer::new(Config::Sgd { lr: 0.1 }, &params, &[0, 1]).unwrap();
        let mut copy = params.clone();
        let e = opt
            .next(&mut copy, &[0, 1], &[array(&[1.0])])
            .unwrap_err()
            .to_string();
        assert!(e.contains("1 gradients for 2"), "{e}");
    }

    #[test]
    fn slots_that_do_not_belong_to_this_run_are_refused_not_indexed_past() {
        let params = vec![array(&[0.0]), array(&[0.0])];
        let mut opt = Optimizer::new(ADAMW, &params, &[0, 1]).unwrap();
        opt.restore(Vec::new(), Vec::new(), 7);
        let mut copy = params.clone();
        let e = opt
            .next(&mut copy, &[0, 1], &[array(&[1.0]), array(&[1.0])])
            .unwrap_err()
            .to_string();
        assert!(e.contains("did not come from this session"), "{e}");
    }
}
