//! Building a parameter from what the graph says it should start as.
//!
//! A GraphConfig declares an initializer per parameter, and until now
//! nobody read it: every run was handed its parameters. Fine-tuning is
//! what changes that. A classification head put on top of a published
//! encoder exists in no checkpoint, so it has to come from its
//! declaration, while everything under it comes from the file.
//!
//! Drawn from the run's seed, not from a global. `docs/plan.md` section
//! 11.1 lists "parameter 初期化 seed" among the state a reproducible run
//! must manage explicitly, and this is where that begins: the same seed
//! and the same graph give the same starting parameters, on any machine.

use anyhow::{Context, Result};
use mlx_rs::Array;
use serde_json::Value;

use crate::graph::ParameterSpec;

/// One parameter, as its declaration says it should start.
///
/// `key` is this parameter's own, split from the run's by the caller, so
/// that what one parameter draws does not depend on how many were built
/// before it.
pub fn build(spec: &ParameterSpec, key: &Array) -> Result<Array> {
    let kind = spec
        .initializer
        .get("type")
        .and_then(Value::as_str)
        .with_context(|| format!("parameter {:?}: its initializer has no type", spec.path))?;
    let shape = &spec.shape;

    Ok(match kind {
        "zeros" => Array::zeros::<f32>(shape)?,
        "ones" => Array::ones::<f32>(shape)?,
        "normal" => {
            let std = number(spec, "std").unwrap_or(0.02);
            mlx_rs::random::normal::<f32>(shape, None, Some(std), Some(key))?
        }
        // Kaiming uniform as PyTorch's `Linear` uses it: bound
        // sqrt(6 / fan_in) with the default gain, where fan_in is the
        // last dimension of a [out, in] weight.
        "kaiming_uniform" => {
            let fan_in = *shape.last().with_context(|| {
                format!("parameter {:?}: kaiming_uniform wants a shape", spec.path)
            })? as f32;
            anyhow::ensure!(
                fan_in > 0.0,
                "parameter {:?}: kaiming_uniform wants a positive last dimension",
                spec.path
            );
            let bound = (6.0f32 / fan_in).sqrt();
            mlx_rs::random::uniform::<_, f32>(-bound, bound, shape, Some(key))?
        }
        other => anyhow::bail!(
            "parameter {:?}: initializer {other:?} is not one this engine builds \
             (zeros, ones, normal, kaiming_uniform)",
            spec.path
        ),
    })
}

fn number(spec: &ParameterSpec, key: &str) -> Option<f32> {
    spec.initializer.get(key).and_then(Value::as_f64).map(|v| v as f32)
}
