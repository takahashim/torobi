//! MLX's quantized matrices: a weight held in a few bits per value, with
//! a scale and a bias per group of values, and a matmul that reads it
//! packed (docs/qlora.md).
//!
//! Three calls, all of `mlx_quantize`'s family. None of them is an IR op:
//! `quantize` is what a bf16 file becomes when it is read into a run that
//! declared its base quantized, `dequantize` is how such a weight is read
//! back as numbers, and `quantized_matmul` is the one that appears in a
//! graph. The shape rules live in the IR (`lib/torobi/shape.rb`); what is
//! here is the call and what MLX says when it refuses.
//!
//! **The gradient.** MLX's `QuantizedMatmul` differentiates with respect
//! to `x`, and with respect to the scales and biases, and refuses the
//! packed weight: there is no gradient to a rounded integer. Torobi never
//! asks for any of the three (they are declared `trainable: false`, and a
//! declaration that says otherwise is refused when the graph is built), so
//! what the tests below fix is which refusals are MLX's and which are
//! Torobi's own.

use std::ffi::CStr;

use super::error::{check, Exception, Result};
use super::handle::{NoArray, Stream, Vector};
use super::{sys, Array, Dtype};

/// How the values in a group are mapped to the integers that stand for
/// them. Only `affine` (`q = round((w - bias) / scale)`), which is what
/// every published `*-4bit` checkpoint uses; MLX also has the block
/// floating-point modes (`mxfp4`, `mxfp8`, `nvfp4`), which carry no bias
/// and are not read anywhere here yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Affine,
}

impl Mode {
    /// The mode by the name a checkpoint's `config.json` gives it.
    pub fn parse(name: &str) -> Result<Mode> {
        match name {
            "affine" => Ok(Mode::Affine),
            other => Err(Exception::custom(format!(
                "quantization mode {other:?} is not one this engine reads; affine is"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Mode::Affine => "affine",
        }
    }

    fn as_cstr(self) -> &'static CStr {
        match self {
            Mode::Affine => c"affine",
        }
    }
}

/// What a quantized matrix was made with, which every call that reads it
/// has to be told again: nothing in the three arrays says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quantization {
    /// How many consecutive values along the last axis share one scale
    /// and one bias. MLX's kernels exist for 32, 64 and 128.
    pub group_size: i32,
    /// Bits per value. MLX's kernels exist for 2, 3, 4, 5, 6 and 8.
    pub bits: i32,
    pub mode: Mode,
}

impl Quantization {
    /// How many values one `u32` of the packed weight holds, which is what
    /// turns the packed shape `[d_out, d_in × bits / 32]` back into
    /// `[d_out, d_in]`.
    pub fn values_per_word(self) -> i32 {
        32 / self.bits
    }
}

/// A matrix as `quantize` leaves it: packed `u32` values of shape
/// `[d_out, d_in × bits / 32]`, and a scale and bias per group of shape
/// `[d_out, d_in / group_size]` in the matrix's own dtype. The biases are
/// what the affine mode has and the block floating-point modes do not.
pub struct Quantized {
    pub weight: Array,
    pub scales: Array,
    pub biases: Option<Array>,
}

fn some(value: i32) -> sys::mlx_optional_int {
    sys::mlx_optional_int { value, has_value: true }
}

fn dtype_or_default(dtype: Option<Dtype>) -> sys::mlx_optional_dtype {
    match dtype {
        Some(dtype) => sys::mlx_optional_dtype { value: dtype.to_raw(), has_value: true },
        None => sys::mlx_optional_dtype { value: Dtype::Float32.to_raw(), has_value: false },
    }
}

/// `w` quantized along its last axis: `mlx_quantize`. The last dimension
/// must be a multiple of the group size, and MLX says so when it is not.
///
/// `global_scale` is passed empty: it belongs to `nvfp4` alone.
pub fn quantize(w: impl AsRef<Array>, quantization: Quantization) -> Result<Quantized> {
    let stream = Stream::current()?;
    let none = NoArray::new();
    let mut out = Vector::new();
    check(
        unsafe {
            sys::mlx_quantize(
                out.as_out(),
                w.as_ref().as_raw(),
                some(quantization.group_size),
                some(quantization.bits),
                quantization.mode.as_cstr().as_ptr(),
                none.as_raw(),
                stream.as_raw(),
            )
        },
        "mlx_quantize",
    )?;
    let mut arrays = out.arrays()?;
    // Affine hands back three (weight, scales, biases); the block
    // floating-point modes hand back two. Read from the back so the two
    // that are always there are named right either way.
    let biases = if arrays.len() == 3 { arrays.pop() } else { None };
    let scales = arrays
        .pop()
        .ok_or_else(|| Exception::custom("mlx_quantize returned no scales"))?;
    let weight = arrays
        .pop()
        .ok_or_else(|| Exception::custom("mlx_quantize returned no weight"))?;
    Ok(Quantized { weight, scales, biases })
}

/// The numbers a quantized matrix stands for, as `[d_out, d_in]` in
/// `dtype`, or in the scales' dtype when none is asked for:
/// `mlx_dequantize`.
pub fn dequantize(
    weight: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: Option<&Array>,
    quantization: Quantization,
    dtype: Option<Dtype>,
) -> Result<Array> {
    let none = NoArray::new();
    let (w, s, b) = (weight.as_ref().as_raw(), scales.as_ref().as_raw(), none.or(biases));
    Array::on_stream("mlx_dequantize", |res, stream| unsafe {
        sys::mlx_dequantize(
            res,
            w,
            s,
            b,
            some(quantization.group_size),
            some(quantization.bits),
            quantization.mode.as_cstr().as_ptr(),
            none.as_raw(),
            dtype_or_default(dtype),
            stream,
        )
    })
}

/// `x @ dequantize(w)ᵀ` (with `transpose`, which is how a linear's
/// `[d_out, d_in]` weight is read) or `x @ dequantize(w)`, computed from
/// the packed weight without unpacking it: `mlx_quantized_matmul`.
pub fn quantized_matmul(
    x: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: Option<&Array>,
    transpose: bool,
    quantization: Quantization,
) -> Result<Array> {
    let none = NoArray::new();
    let (x, w, s, b) = (
        x.as_ref().as_raw(),
        weight.as_ref().as_raw(),
        scales.as_ref().as_raw(),
        none.or(biases),
    );
    Array::on_stream("mlx_quantized_matmul", |res, stream| unsafe {
        sys::mlx_quantized_matmul(
            res,
            x,
            w,
            s,
            b,
            transpose,
            some(quantization.group_size),
            some(quantization.bits),
            quantization.mode.as_cstr().as_ptr(),
            stream,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlxc::transforms::value_and_grad_with_argnums;
    use crate::runtime;

    fn run<R>(f: impl FnOnce() -> Result<R>) -> R {
        runtime::runtime()
            .execute(|| Ok(f()?))
            .expect("the runtime should let this through")
    }

    fn values(a: &Array) -> Vec<f32> {
        a.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap().as_slice::<f32>().unwrap().to_vec()
    }

    /// The quantization every published `*-4bit` checkpoint was made with.
    const FOUR_BIT_G64: Quantization = Quantization { group_size: 64, bits: 4, mode: Mode::Affine };

    /// A `[d_out, d_in]` matrix with a different spread in every row, so a
    /// group's scale and bias both matter: row r holds values in
    /// `[-(r + 1) / d_out, (r + 1) / d_out]`, not on the grid.
    fn weight(d_out: usize, d_in: usize) -> Vec<f32> {
        (0..d_out * d_in)
            .map(|i| {
                let (r, c) = (i / d_in, i % d_in);
                let range = (r + 1) as f32 / d_out as f32;
                range * ((c as f32 * 0.37 + r as f32 * 1.3).sin())
            })
            .collect()
    }

    fn matrix(d_out: usize, d_in: usize) -> Result<Array> {
        Array::from_slice(&weight(d_out, d_in), &[d_out as i32, d_in as i32])
    }

    fn close(got: &[f32], want: &[f32], tolerance: f32) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g - w).abs() <= tolerance * (1.0 + w.abs()),
                "{g} against {w} (tolerance {tolerance})"
            );
        }
    }

    /// The three arrays have the shapes and dtypes docs/qlora.md section
    /// 2.4 says, which are the shapes a checkpoint holds them in.
    #[test]
    fn a_quantized_matrix_is_a_packed_weight_and_a_scale_and_bias_per_group() {
        let (w, s, b) = run(|| {
            let q = quantize(matrix(8, 128)?, FOUR_BIT_G64)?;
            let b = q.biases.expect("affine has biases");
            Ok((
                (q.weight.dtype(), q.weight.shape().to_vec()),
                (q.scales.dtype(), q.scales.shape().to_vec()),
                (b.dtype(), b.shape().to_vec()),
            ))
        });
        assert_eq!(w, (Dtype::Uint32, vec![8, 128 * 4 / 32]));
        assert_eq!(s, (Dtype::Float32, vec![8, 2]));
        assert_eq!(b, (Dtype::Float32, vec![8, 2]));
    }

    /// Four bits put sixteen points across a group's range, so nothing
    /// comes back further from where it was than one step of that grid,
    /// and something comes back moved: a round trip that lost nothing
    /// would mean the test is not quantizing.
    #[test]
    fn the_round_trip_error_is_within_one_step_of_the_grid() {
        let (d_out, d_in) = (8, 128);
        let original = weight(d_out, d_in);
        let back = run(|| {
            let q = quantize(matrix(d_out, d_in)?, FOUR_BIT_G64)?;
            Ok(values(&dequantize(&q.weight, &q.scales, q.biases.as_ref(), FOUR_BIT_G64, None)?))
        });
        assert_eq!(back.len(), original.len());
        let mut moved = false;
        for r in 0..d_out {
            for g in 0..d_in / 64 {
                let group = r * d_in + g * 64..r * d_in + (g + 1) * 64;
                let (lo, hi) = original[group.clone()]
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
                let step = (hi - lo) / 15.0;
                for i in group {
                    let error = (back[i] - original[i]).abs();
                    assert!(error <= step + 1e-6, "row {r}: {} came back as {} (step {step})", original[i], back[i]);
                    moved |= error > 1e-6;
                }
            }
        }
        assert!(moved, "a 4-bit round trip changes something");
    }

    /// The packed matmul and the unpacked one are the same product. In
    /// f32 to the accumulation order; in bf16, the dtype a model is held
    /// in, to one rounding of the output.
    ///
    /// The reference is taken in f32 either way. The kernel unpacks each
    /// weight as `scale · q + bias` in f32 and rounds only its result,
    /// while a bf16 `dequantize` rounds every weight before the product:
    /// 128 of those roundings summed came to 2 % where one rounding of
    /// the output is 0.4 %, which is what the first version of this test
    /// found (2026-11-08).
    #[test]
    fn quantized_matmul_is_the_matmul_of_the_dequantized_weight() {
        for (dtype, tolerance) in [(Dtype::Float32, 1e-4), (Dtype::Bfloat16, 1e-2)] {
            let (got, want) = run(|| {
                let w = matrix(8, 128)?.as_dtype(dtype)?;
                let x = Array::from_slice(&weight(3, 128), &[3, 128])?.as_dtype(dtype)?;
                let q = quantize(&w, FOUR_BIT_G64)?;
                let packed = quantized_matmul(&x, &q.weight, &q.scales, q.biases.as_ref(), true, FOUR_BIT_G64)?;
                let unpacked =
                    dequantize(&q.weight, &q.scales, q.biases.as_ref(), FOUR_BIT_G64, Some(Dtype::Float32))?;
                let plain = x.as_dtype(Dtype::Float32)?.matmul(unpacked.transpose_axes(&[1, 0])?)?;
                assert_eq!(packed.dtype(), dtype);
                assert_eq!(packed.shape(), &[3, 8]);
                Ok((values(&packed), values(&plain)))
            });
            close(&got, &want, tolerance);
        }
    }

    /// d/dx of sum(x Wᵀ) is a row of column sums of W, for every row of
    /// x: the gradient a LoRA's activations need, and the one MLX has.
    #[test]
    fn the_gradient_with_respect_to_x_comes_back() {
        let (grad, shape, want) = run(|| {
            let q = quantize(matrix(8, 128)?, FOUR_BIT_G64)?;
            let x = Array::from_slice(&weight(3, 128), &[3, 128])?;
            let unpacked = dequantize(&q.weight, &q.scales, q.biases.as_ref(), FOUR_BIT_G64, None)?;
            let want = values(&unpacked.sum_axes(&[0], false)?);
            let biases = q.biases.as_ref();
            let f = |inputs: &[Array]| -> Result<Vec<Array>> {
                Ok(vec![quantized_matmul(&inputs[0], &q.weight, &q.scales, biases, true, FOUR_BIT_G64)?.sum(false)?])
            };
            let (_, grads) = value_and_grad_with_argnums(f, &[0])(std::slice::from_ref(&x))?;
            Ok((values(&grads[0]), grads[0].shape().to_vec(), want))
        });
        assert_eq!(shape, vec![3, 128]);
        for row in grad.chunks(128) {
            close(row, &want, 1e-4);
        }
    }

    /// There is no gradient to the packed weight, and MLX is the one that
    /// says so. This is the fact docs/qlora.md section 3.2 rests on: a
    /// quantized weight cannot be trained, so a graph that declares one
    /// trainable is refused before MLX would be asked.
    #[test]
    fn the_gradient_with_respect_to_the_packed_weight_is_mlx_s_refusal() {
        let said = run(|| {
            let q = quantize(matrix(8, 128)?, FOUR_BIT_G64)?;
            let x = Array::from_slice(&weight(3, 128), &[3, 128])?;
            let biases = q.biases.expect("affine has biases");
            let f = |inputs: &[Array]| -> Result<Vec<Array>> {
                Ok(vec![quantized_matmul(&inputs[0], &inputs[1], &inputs[2], Some(&inputs[3]), true, FOUR_BIT_G64)?
                    .sum(false)?])
            };
            let refused = value_and_grad_with_argnums(f, &[1])(&[x, q.weight, q.scales, biases]);
            Ok(refused.err().map(|e| e.what().to_string()))
        });
        assert!(said.is_some(), "MLX differentiated a packed weight");
    }

    /// The scales and biases do have a gradient in MLX. Torobi does not
    /// train them either, and that is Torobi's policy rather than MLX's
    /// limit, which is why this is pinned next to the refusal above: if
    /// MLX ever stops handing these back, section 3.2 has to change with
    /// this test, not quietly.
    #[test]
    fn the_gradients_with_respect_to_scales_and_biases_are_mlx_s_to_give() {
        let shapes = run(|| {
            let q = quantize(matrix(8, 128)?, FOUR_BIT_G64)?;
            let x = Array::from_slice(&weight(3, 128), &[3, 128])?;
            let biases = q.biases.expect("affine has biases");
            let f = |inputs: &[Array]| -> Result<Vec<Array>> {
                Ok(vec![quantized_matmul(&inputs[0], &inputs[1], &inputs[2], Some(&inputs[3]), true, FOUR_BIT_G64)?
                    .sum(false)?])
            };
            let (_, grads) = value_and_grad_with_argnums(f, &[2, 3])(&[x, q.weight, q.scales, biases])?;
            Ok(grads.iter().map(|g| (g.dtype(), g.shape().to_vec())).collect::<Vec<_>>())
        });
        assert_eq!(shapes, vec![(Dtype::Float32, vec![8, 2]), (Dtype::Float32, vec![8, 2])]);
    }

    /// A last dimension the group size does not divide is MLX's refusal,
    /// in MLX's words. The IR refuses it earlier, with the linear's name;
    /// this fixes what the engine would say if something got past.
    #[test]
    fn a_width_the_group_size_does_not_divide_is_refused_by_mlx() {
        let said = run(|| Ok(quantize(matrix(4, 48)?, FOUR_BIT_G64).err().map(|e| e.what().to_string())));
        let said = said.expect("48 is not a multiple of 64");
        assert!(said.contains("quantize") || said.contains("group"), "{said}");
    }

    #[test]
    fn a_mode_is_read_by_the_name_a_checkpoint_gives_it() {
        assert_eq!(Mode::parse("affine").unwrap(), Mode::Affine);
        assert_eq!(Mode::Affine.name(), "affine");
        let said = Mode::parse("mxfp4").unwrap_err();
        assert!(said.what().contains("mxfp4"), "{}", said.what());
        assert_eq!(FOUR_BIT_G64.values_per_word(), 8);
    }
}
