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
    let (q, k, v) = (queries.as_ref().raw()?, keys.as_ref().raw()?, values.as_ref().raw()?);
    // The empty handle is mlx-c's "none" here, so it is passed as it is
    // (`as_raw`); an array the caller did hand over must have been made.
    let none = Array::empty();
    let (mode, mask) = match mask.into() {
        None => (c"", none.as_raw()),
        Some(ScaledDotProductAttentionMask::Array(mask)) => (c"", mask.raw()?),
        Some(ScaledDotProductAttentionMask::Causal) => (c"causal", none.as_raw()),
    };
    let sinks = match sinks.into() {
        Some(sinks) => sinks.raw()?,
        None => none.as_raw(),
    };
    let stream = Stream::current();
    Array::try_from_op(|res| unsafe {
        sys::mlx_fast_scaled_dot_product_attention(
            res,
            q,
            k,
            v,
            scale,
            mode.as_ptr(),
            mask,
            sinks,
            false,
            stream.as_raw(),
        )
    })
}
