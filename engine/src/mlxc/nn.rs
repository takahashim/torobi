//! The one activation the engine takes from a library rather than writing
//! out in its own interpreter.

use std::f32::consts::PI;

use super::error::Result;
use super::ops::tanh;
use super::{Array, Dtype};

/// GELU's tanh approximation:
/// 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³))).
///
/// The same operations, in the same order, as mlx-rs's
/// `nn::gelu_approximate`, with one difference: the constants are in
/// `x`'s dtype rather than f32. mlx-rs's f32 constants promote a bf16 `x`
/// to f32, which turned everything after the first activation of a bf16
/// Gemma into f32 (2026-11-09); for an f32 `x` the two are the same
/// function to the bit. What mlx-rs adds is `compile`, which fuses these
/// into one kernel; that
/// changes how fast this runs, not what it computes. Not compiled here:
/// compiling through mlx-c means handing MLX a closure with a stable id
/// (`mlx_detail_compile`) and owning the cache that id keys, and one
/// activation in one model family is not worth being the first thing the
/// engine compiles. `rake oracle:gemma3_forward` is what holds its numbers.
pub fn gelu_approximate(x: impl AsRef<Array>) -> Result<Array> {
    let x = x.as_ref();
    let like = |value: f32| -> Result<Array> {
        let array = Array::from_f32(value)?;
        match x.dtype() {
            Dtype::Bfloat16 | Dtype::Float16 => array.as_dtype(x.dtype()),
            _ => Ok(array),
        }
    };
    let cubed = x.power(Array::from_i32(3)?)?;
    let inner = x.add(like(0.044715)?.multiply(&cubed)?)?;
    let scaled = like(2.0 / PI)?.sqrt()?.multiply(&inner)?;
    like(0.5)?.multiply(x)?.multiply(like(1.0)?.add(tanh(&scaled)?)?)
}
