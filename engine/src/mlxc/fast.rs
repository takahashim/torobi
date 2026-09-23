//! MLX's fused kernels.

use super::error::Result;
use super::handle::Stream;
use super::{sys, Array};

/// What masks the scores before the softmax: an additive array, or the
/// causal triangle MLX builds itself.
#[derive(Debug)]
pub enum ScaledDotProductAttentionMask<'a> {
    Array(&'a Array),
    Causal,
}

impl<'a> From<&'a Array> for ScaledDotProductAttentionMask<'a> {
    fn from(mask: &'a Array) -> Self {
        ScaledDotProductAttentionMask::Array(mask)
    }
}

/// softmax(q kᵀ · scale + mask) v, through MLX's own kernel.
///
/// mlx-c says "no mask" and "no sinks" with an empty array and names the
/// mask's kind in a string: `""` for an array (or none) and `"causal"` for
/// the triangle. The empty arrays are owned and freed here; mlx-rs made
/// them and let them go. `force_fused` is `false`, as mlx-rs passes it,
/// so MLX falls back to the unfused form where its kernel does not apply.
pub fn scaled_dot_product_attention<'a>(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    values: impl AsRef<Array>,
    scale: f32,
    mask: impl Into<Option<ScaledDotProductAttentionMask<'a>>>,
    sinks: impl Into<Option<&'a Array>>,
) -> Result<Array> {
    let none = Array::empty();
    let (mode, mask) = match mask.into() {
        None => (c"", &none),
        Some(ScaledDotProductAttentionMask::Array(mask)) => (c"", mask),
        Some(ScaledDotProductAttentionMask::Causal) => (c"causal", &none),
    };
    let sinks = sinks.into().unwrap_or(&none);
    let stream = Stream::default_device();
    Array::try_from_op(|res| unsafe {
        sys::mlx_fast_scaled_dot_product_attention(
            res,
            queries.as_ref().as_raw(),
            keys.as_ref().as_raw(),
            values.as_ref().as_raw(),
            scale,
            mode.as_ptr(),
            mask.as_raw(),
            sinks.as_raw(),
            false,
            stream.as_raw(),
        )
    })
}
