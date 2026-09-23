//! The one activation the engine takes from a library rather than writing
//! out in its own interpreter.

use std::f32::consts::PI;

use super::error::Result;
use super::ops::tanh;
use super::Array;

/// GELU's tanh approximation:
/// 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³))).
///
/// The same operations, in the same order and with the same constant
/// dtypes (f32, and an i32 exponent), as mlx-rs's `nn::gelu_approximate`.
/// What mlx-rs adds is `compile`, which fuses these into one kernel; that
/// changes how fast this runs, not what it computes. Not compiled here:
/// compiling through mlx-c means handing MLX a closure with a stable id
/// (`mlx_detail_compile`) and owning the cache that id keys, and one
/// activation in one model family is not worth being the first thing the
/// engine compiles. `rake oracle:gemma3_forward` is what holds its numbers.
pub fn gelu_approximate(x: impl AsRef<Array>) -> Result<Array> {
    let x = x.as_ref();
    let cubed = x.power(Array::from_i32(3))?;
    let inner = x.add(Array::from_f32(0.044715).multiply(&cubed)?)?;
    let scaled = Array::from_f32(2.0 / PI).sqrt()?.multiply(&inner)?;
    Array::from_f32(0.5)
        .multiply(x)?
        .multiply(Array::from_f32(1.0).add(tanh(&scaled)?)?)
}
