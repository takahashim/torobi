//! Keys and draws.
//!
//! Every draw takes its key from the caller, by type: there is no global
//! state to fall back on. mlx-rs kept one, seeded from a clock, and on
//! Linux that clock is what failed to link (docs/vendoring.md). The engine
//! never wanted it, because a run is reproducible only if every draw comes
//! from the run's seed.

use super::error::Result;
use super::{sys, Array, ArrayElement};

/// A key from a seed.
pub fn key(seed: u64) -> Result<Array> {
    Array::try_from_op("mlx_random_key", |res| unsafe { sys::mlx_random_key(res, seed) })
}

/// Two keys from one, as mlx-rs returned them: split `num` ways, then the
/// first and second of those. The engine asks for two and uses both.
pub fn split(key: &Array, num: i32) -> Result<(Array, Array)> {
    let keys = Array::on_stream("mlx_random_split_num", |res, s| unsafe {
        sys::mlx_random_split_num(res, key.as_raw(), num, s)
    })?;
    Ok((keys.take_axis(Array::from_i32(0)?, 0)?, keys.take_axis(Array::from_i32(1)?, 0)?))
}

/// Uniform on [lower, upper), in `T`'s dtype.
pub fn uniform<T: ArrayElement>(lower: f32, upper: f32, shape: &[i32], key: &Array) -> Result<Array> {
    let (lower, upper) = (Array::from_f32(lower)?, Array::from_f32(upper)?);
    Array::on_stream("mlx_random_uniform", |res, s| unsafe {
        sys::mlx_random_uniform(
            res,
            lower.as_raw(),
            upper.as_raw(),
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.to_raw(),
            key.as_raw(),
            s,
        )
    })
}

/// Normal with mean `loc` and standard deviation `scale`, in `T`'s dtype.
pub fn normal<T: ArrayElement>(shape: &[i32], loc: f32, scale: f32, key: &Array) -> Result<Array> {
    Array::on_stream("mlx_random_normal", |res, s| unsafe {
        sys::mlx_random_normal(res, shape.as_ptr(), shape.len(), T::DTYPE.to_raw(), loc, scale, key.as_raw(), s)
    })
}

/// `true` with probability `p` at each position of `shape`.
pub fn bernoulli(p: &Array, shape: &[i32], key: &Array) -> Result<Array> {
    Array::on_stream("mlx_random_bernoulli", |res, s| unsafe {
        sys::mlx_random_bernoulli(res, p.as_raw(), shape.as_ptr(), shape.len(), key.as_raw(), s)
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
            let normal = super::normal::<f32>(&[3], 0.0, 0.02, &first)?;
            let uniform = super::uniform::<f32>(-0.5, 0.5, &[3], &second)?;
            let kept = bernoulli(&Array::from_f32(0.75)?, &[8], &first)?;
            Ok((
                root.as_slice::<u32>()?.to_vec(),
                first.as_slice::<u32>()?.to_vec(),
                second.as_slice::<u32>()?.to_vec(),
                normal.as_slice::<f32>()?.to_vec(),
                uniform.as_slice::<f32>()?.to_vec(),
                kept.as_slice::<bool>()?.to_vec(),
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
            let n = normal::<f32>(&[2, 3], 0.0, 1.0, &k)?;
            let b = bernoulli(&Array::from_f32(0.5)?, &[4], &k)?;
            Ok((n.shape().to_vec(), n.dtype(), b.shape().to_vec(), b.dtype()))
        });
        assert_eq!(got, (vec![2, 3], Dtype::Float32, vec![4], Dtype::Bool));
    }
}
