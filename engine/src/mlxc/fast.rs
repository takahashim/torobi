//! MLX's fused kernels.

use super::error::Result;
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
/// the triangle ([`NoArray`]). `force_fused` is `false`, as mlx-rs passes it,
/// so MLX falls back to the unfused form where its kernel does not apply.
pub fn scaled_dot_product_attention<'a>(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    values: impl AsRef<Array>,
    scale: f32,
    mask: impl Into<Option<ScaledDotProductAttentionMask<'a>>>,
    sinks: impl Into<Option<&'a Array>>,
) -> Result<Array> {
    let none = NoArray::new();
    let (mode, mask) = match mask.into() {
        None => (c"", none.as_raw()),
        Some(ScaledDotProductAttentionMask::Array(mask)) => (c"", mask.as_raw()),
        Some(ScaledDotProductAttentionMask::Causal) => (c"causal", none.as_raw()),
    };
    let sinks = sinks.into().map_or(none.as_raw(), Array::as_raw);
    let (q, k, v) = (queries.as_ref().as_raw(), keys.as_ref().as_raw(), values.as_ref().as_raw());
    Array::on_stream("mlx_fast_scaled_dot_product_attention", |res, s| unsafe {
        sys::mlx_fast_scaled_dot_product_attention(res, q, k, v, scale, mode.as_ptr(), mask, sinks, false, s)
    })
}

/// mlx-c's "no array", for an optional argument: an empty handle. Its own
/// type because an [`Array`] always holds one, and owned so it is freed;
/// mlx-rs made these and let them go.
struct NoArray(sys::mlx_array);

impl NoArray {
    fn new() -> Self {
        NoArray(unsafe { sys::mlx_array_new() })
    }

    fn as_raw(&self) -> sys::mlx_array {
        self.0
    }
}

impl Drop for NoArray {
    fn drop(&mut self) {
        unsafe { sys::mlx_array_free(self.0) };
    }
}
