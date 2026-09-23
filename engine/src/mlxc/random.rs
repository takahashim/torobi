//! Keys and draws.
//!
//! Every draw here takes its key from the caller. mlx-rs falls back to a
//! global state seeded from the clock when no key is given, and on Linux
//! that clock is what failed to link (docs/vendoring.md). The engine never
//! draws without a key, because a run is reproducible only if every draw
//! comes from the run's seed, so there is no fallback to port: a missing
//! key is an error.

use super::error::{Exception, Result};
use super::handle::Stream;
use super::{sys, Array, ArrayElement};

fn given<'a>(key: impl Into<Option<&'a Array>>) -> Result<&'a Array> {
    key.into().ok_or_else(|| {
        Exception::custom("a draw was asked for without a key; the engine keeps no global one")
    })
}

/// A key from a seed.
pub fn key(seed: u64) -> Result<Array> {
    Array::try_from_op(|res| unsafe { sys::mlx_random_key(res, seed) })
}

/// Two keys from one, as mlx-rs returns them: split `num` ways, then the
/// first and second of those. The engine asks for two and uses both.
pub fn split(key: impl AsRef<Array>, num: i32) -> Result<(Array, Array)> {
    let stream = Stream::default_device();
    let keys = Array::try_from_op(|res| unsafe {
        sys::mlx_random_split_num(res, key.as_ref().as_raw(), num, stream.as_raw())
    })?;
    Ok((keys.take_axis(Array::from_i32(0), 0)?, keys.take_axis(Array::from_i32(1), 0)?))
}

/// Uniform on [lower, upper), in `T`'s dtype.
pub fn uniform<'a, E: Into<Array>, T: ArrayElement>(
    lower: E,
    upper: E,
    shape: &[i32],
    key: impl Into<Option<&'a Array>>,
) -> Result<Array> {
    let (lower, upper): (Array, Array) = (lower.into(), upper.into());
    let key = given(key)?;
    let stream = Stream::default_device();
    Array::try_from_op(|res| unsafe {
        sys::mlx_random_uniform(
            res,
            lower.as_raw(),
            upper.as_raw(),
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.to_raw(),
            key.as_raw(),
            stream.as_raw(),
        )
    })
}

/// Normal with mean `loc` (0 when `None`) and standard deviation `scale`
/// (1 when `None`), in `T`'s dtype.
pub fn normal<'a, T: ArrayElement>(
    shape: &[i32],
    loc: impl Into<Option<f32>>,
    scale: impl Into<Option<f32>>,
    key: impl Into<Option<&'a Array>>,
) -> Result<Array> {
    let (loc, scale) = (loc.into().unwrap_or(0.0), scale.into().unwrap_or(1.0));
    let key = given(key)?;
    let stream = Stream::default_device();
    Array::try_from_op(|res| unsafe {
        sys::mlx_random_normal(
            res,
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.to_raw(),
            loc,
            scale,
            key.as_raw(),
            stream.as_raw(),
        )
    })
}

/// `true` with probability `p` (0.5 when `None`), at each position of
/// `shape`.
pub fn bernoulli<'a>(
    p: impl Into<Option<&'a Array>>,
    shape: &[i32],
    key: impl Into<Option<&'a Array>>,
) -> Result<Array> {
    let half = Array::from_f32(0.5);
    let p = p.into().unwrap_or(&half);
    let key = given(key)?;
    let stream = Stream::default_device();
    Array::try_from_op(|res| unsafe {
        sys::mlx_random_bernoulli(
            res,
            p.as_raw(),
            shape.as_ptr(),
            shape.len(),
            key.as_raw(),
            stream.as_raw(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlxc::Dtype;
    use crate::runtime;

    fn run<R>(f: impl FnOnce() -> Result<R>) -> R {
        runtime::runtime()
            .execute(|| Ok(f()?))
            .expect("the runtime should let this through")
    }

    /// The numbers a seed draws are MLX's, and not the binding's to
    /// change: a checkpoint written before the move resumes into the same
    /// dropout masks after it. These are what mlx-rs 0.32.0 drew from
    /// seed 42, recorded through it before the move.
    #[test]
    fn a_seed_draws_what_it_drew_through_mlx_rs() {
        let (key, first, second, normal, uniform, kept) = run(|| {
            let root = key(42)?;
            let (first, second) = split(&root, 2)?;
            let normal = super::normal::<f32>(&[3], None, Some(0.02), Some(&first))?;
            let uniform = super::uniform::<_, f32>(-0.5f32, 0.5f32, &[3], Some(&second))?;
            let kept = bernoulli(&Array::from_f32(0.75), &[8], &first)?;
            Ok((
                root.as_slice::<u32>().to_vec(),
                first.as_slice::<u32>().to_vec(),
                second.as_slice::<u32>().to_vec(),
                normal.as_slice::<f32>().to_vec(),
                uniform.as_slice::<f32>().to_vec(),
                kept.as_slice::<bool>().to_vec(),
            ))
        });
        assert_eq!(key, RECORDED.key);
        assert_eq!(first, RECORDED.first);
        assert_eq!(second, RECORDED.second);
        assert_eq!(normal, RECORDED.normal);
        assert_eq!(uniform, RECORDED.uniform);
        assert_eq!(kept, RECORDED.kept);
    }

    struct Recorded {
        key: [u32; 2],
        first: [u32; 2],
        second: [u32; 2],
        normal: [f32; 3],
        uniform: [f32; 3],
        kept: [bool; 8],
    }

    const RECORDED: Recorded = Recorded {
        key: [0, 42],
        first: [2465931498, 3679230171],
        second: [255383827, 267815257],
        normal: [0.012666016, 0.019221846, 0.02725154],
        uniform: [-0.21482974, 0.111944914, -0.32434744],
        kept: [true, true, true, false, true, true, false, true],
    };

    #[test]
    fn a_draw_has_the_shape_and_dtype_it_was_asked_for() {
        let got = run(|| {
            let k = key(1)?;
            let n = normal::<f32>(&[2, 3], None, None, Some(&k))?;
            let b = bernoulli(None, &[4], &k)?;
            Ok((n.shape().to_vec(), n.dtype(), b.shape().to_vec(), b.dtype()))
        });
        assert_eq!(got, (vec![2, 3], Dtype::Float32, vec![4], Dtype::Bool));
    }

    #[test]
    fn a_draw_without_a_key_is_refused_rather_than_seeded_from_a_clock() {
        let said = run(|| Ok(normal::<f32>(&[1], None, None, None).unwrap_err().to_string()));
        assert!(said.contains("without a key"), "{said}");
    }
}
