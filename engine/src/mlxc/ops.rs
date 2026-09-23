//! The operations, one mlx-c call each.
//!
//! Which call is the whole difficulty: `sum` and `sum_axes` are different
//! functions in mlx-c (`mlx_sum` reduces everything, `mlx_sum_axes` only
//! what it is told), and picking the wrong one compiles and gives a wrong
//! number. Each one here is the call mlx-rs 0.32.0 made for the same
//! method, and docs/vendoring.md ("mlx-c, call by call") lists them.
//!
//! Every op goes through [`Array::on_stream`], which runs it on the current
//! stream and names the mlx-c call in the error if MLX gives no reason.

use super::error::Result;
use super::handle::Vector;
use super::{sys, Array, ArrayElement, Dtype};

/// `a.$name()`: one array in, one out.
macro_rules! unary {
    ($($name:ident => $call:ident),* $(,)?) => {
        impl Array {
            $(
                pub fn $name(&self) -> Result<Array> {
                    Array::on_stream(stringify!($call), |res, s| unsafe { sys::$call(res, self.as_raw(), s) })
                }
            )*
        }
    };
}

/// `a.$name(b)`, broadcasting as MLX does.
macro_rules! binary {
    ($($name:ident => $call:ident),* $(,)?) => {
        impl Array {
            $(
                pub fn $name(&self, other: impl AsRef<Array>) -> Result<Array> {
                    let b = other.as_ref().as_raw();
                    Array::on_stream(stringify!($call), |res, s| unsafe { sys::$call(res, self.as_raw(), b, s) })
                }
            )*
        }
    };
}

/// `a.$name(keep_dims)`: a reduction over every axis. `None` keeps no
/// dimensions, as mlx-rs's default did.
macro_rules! reduce_all {
    ($($name:ident => $call:ident),* $(,)?) => {
        impl Array {
            $(
                pub fn $name(&self, keep_dims: impl Into<Option<bool>>) -> Result<Array> {
                    let keep = keep_dims.into().unwrap_or(false);
                    Array::on_stream(stringify!($call), |res, s| unsafe { sys::$call(res, self.as_raw(), keep, s) })
                }
            )*
        }
    };
}

/// `a.$name(axes, keep_dims)`: a reduction over the axes named, negative
/// ones counting from the end. An empty list reduces nothing, which is
/// MLX's reading and mlx-rs's.
macro_rules! reduce_axes {
    ($($name:ident => $call:ident),* $(,)?) => {
        impl Array {
            $(
                pub fn $name(&self, axes: &[i32], keep_dims: impl Into<Option<bool>>) -> Result<Array> {
                    let keep = keep_dims.into().unwrap_or(false);
                    Array::on_stream(stringify!($call), |res, s| unsafe {
                        sys::$call(res, self.as_raw(), axes.as_ptr(), axes.len(), keep, s)
                    })
                }
            )*
        }
    };
}

/// `a.$name(ints)`: a shape, or a list of axes, and nothing else.
macro_rules! with_ints {
    ($($name:ident => $call:ident),* $(,)?) => {
        impl Array {
            $(
                pub fn $name(&self, ints: &[i32]) -> Result<Array> {
                    Array::on_stream(stringify!($call), |res, s| unsafe {
                        sys::$call(res, self.as_raw(), ints.as_ptr(), ints.len(), s)
                    })
                }
            )*
        }
    };
}

unary! {
    abs => mlx_abs,
    negative => mlx_negative,
    sqrt => mlx_sqrt,
    rsqrt => mlx_rsqrt,
    square => mlx_square,
    exp => mlx_exp,
    log => mlx_log,
    cos => mlx_cos,
    sin => mlx_sin,
}

binary! {
    add => mlx_add,
    subtract => mlx_subtract,
    multiply => mlx_multiply,
    divide => mlx_divide,
    matmul => mlx_matmul,
    power => mlx_power,
}

reduce_all! {
    sum => mlx_sum,
    mean => mlx_mean,
    max => mlx_max,
    min => mlx_min,
}

reduce_axes! {
    sum_axes => mlx_sum_axes,
    mean_axes => mlx_mean_axes,
    max_axes => mlx_max_axes,
}

with_ints! {
    reshape => mlx_reshape,
    transpose_axes => mlx_transpose_axes,
    expand_dims_axes => mlx_expand_dims_axes,
    squeeze_axes => mlx_squeeze_axes,
}

impl Array {
    /// The array in another dtype.
    pub fn as_dtype(&self, dtype: Dtype) -> Result<Array> {
        Array::on_stream("mlx_astype", |res, s| unsafe {
            sys::mlx_astype(res, self.as_raw(), dtype.to_raw(), s)
        })
    }

    /// The rows (or whatever `axis` is) that `indices` names, in its order.
    pub fn take_axis(&self, indices: impl AsRef<Array>, axis: i32) -> Result<Array> {
        let at = indices.as_ref().as_raw();
        Array::on_stream("mlx_take_axis", |res, s| unsafe {
            sys::mlx_take_axis(res, self.as_raw(), at, axis, s)
        })
    }

    /// The same values laid out row-major, sharing the buffer when it
    /// already is. Column-major is not accepted as contiguous, which is
    /// mlx-rs's default and what `as_slice` needs.
    pub fn contiguous(&self) -> Result<Array> {
        Array::on_stream("mlx_contiguous", |res, s| unsafe {
            sys::mlx_contiguous(res, self.as_raw(), false, s)
        })
    }

    pub fn zeros<T: ArrayElement>(shape: &[i32]) -> Result<Array> {
        Array::zeros_dtype(shape, T::DTYPE)
    }

    pub fn ones<T: ArrayElement>(shape: &[i32]) -> Result<Array> {
        Array::on_stream("mlx_ones", |res, s| unsafe {
            sys::mlx_ones(res, shape.as_ptr(), shape.len(), T::DTYPE.to_raw(), s)
        })
    }

    fn zeros_dtype(shape: &[i32], dtype: Dtype) -> Result<Array> {
        Array::on_stream("mlx_zeros", |res, s| unsafe {
            sys::mlx_zeros(res, shape.as_ptr(), shape.len(), dtype.to_raw(), s)
        })
    }
}

/// `$name(a)`: the ops mlx-rs offers as free functions rather than methods.
macro_rules! unary_function {
    ($($name:ident => $call:ident),* $(,)?) => {
        $(
            pub fn $name(a: impl AsRef<Array>) -> Result<Array> {
                let a = a.as_ref().as_raw();
                Array::on_stream(stringify!($call), |res, s| unsafe { sys::$call(res, a, s) })
            }
        )*
    };
}

unary_function! {
    tanh => mlx_tanh,
    sigmoid => mlx_sigmoid,
    erf => mlx_erf,
    stop_gradient => mlx_stop_gradient,
}

/// Zeros of `a`'s shape and dtype. Through `mlx_zeros` rather than
/// `mlx_zeros_like`, as mlx-rs does: the array is read for its shape, and
/// nothing about it enters the graph.
pub fn zeros_like(a: impl AsRef<Array>) -> Result<Array> {
    let a = a.as_ref();
    Array::zeros_dtype(a.shape(), a.dtype())
}

pub fn maximum(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
    let (a, b) = (a.as_ref().as_raw(), b.as_ref().as_raw());
    Array::on_stream("mlx_maximum", |res, s| unsafe { sys::mlx_maximum(res, a, b, s) })
}

/// Softmax along one axis. `precise: None` is `false`: MLX then computes
/// in the input's precision, which for f32 is f32.
pub fn softmax_axis(a: impl AsRef<Array>, axis: i32, precise: impl Into<Option<bool>>) -> Result<Array> {
    let a = a.as_ref().as_raw();
    let precise = precise.into().unwrap_or(false);
    Array::on_stream("mlx_softmax_axis", |res, s| unsafe { sys::mlx_softmax_axis(res, a, axis, precise, s) })
}

pub fn logsumexp_axes(
    a: impl AsRef<Array>,
    axes: &[i32],
    keep_dims: impl Into<Option<bool>>,
) -> Result<Array> {
    let a = a.as_ref().as_raw();
    let keep = keep_dims.into().unwrap_or(false);
    Array::on_stream("mlx_logsumexp_axes", |res, s| unsafe {
        sys::mlx_logsumexp_axes(res, a, axes.as_ptr(), axes.len(), keep, s)
    })
}

/// The arrays end to end along an existing axis.
pub fn concatenate(arrays: &[impl AsRef<Array>], axis: i32) -> Result<Array> {
    let vector = Vector::of(arrays.iter().map(AsRef::as_ref))?;
    Array::on_stream("mlx_concatenate_axis", |res, s| unsafe {
        sys::mlx_concatenate_axis(res, vector.as_raw(), axis, s)
    })
}

/// The arrays side by side along a new axis.
pub fn stack(arrays: &[impl AsRef<Array>], axis: i32) -> Result<Array> {
    let vector = Vector::of(arrays.iter().map(AsRef::as_ref))?;
    Array::on_stream("mlx_stack_axis", |res, s| unsafe { sys::mlx_stack_axis(res, vector.as_raw(), axis, s) })
}

pub mod indexing {
    use super::*;

    /// At each position, the element of `a` along `axis` that `indices`
    /// names there. `None` for the axis reads `a` flattened, as mlx-rs
    /// does.
    pub fn take_along_axis(
        a: impl AsRef<Array>,
        indices: impl AsRef<Array>,
        axis: impl Into<Option<i32>>,
    ) -> Result<Array> {
        let (flat, axis) = match axis.into() {
            None => (Some(a.as_ref().reshape(&[-1])?), 0),
            Some(axis) => (None, axis),
        };
        let a = flat.as_ref().unwrap_or(a.as_ref()).as_raw();
        let at = indices.as_ref().as_raw();
        Array::on_stream("mlx_take_along_axis", |res, s| unsafe { sys::mlx_take_along_axis(res, a, at, axis, s) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime;

    fn run<R>(f: impl FnOnce() -> Result<R>) -> R {
        runtime::runtime()
            .execute(|| Ok(f()?))
            .expect("the runtime should let this through")
    }

    fn values(a: &Array) -> Vec<f32> {
        a.contiguous().unwrap().as_slice::<f32>().unwrap().to_vec()
    }

    fn close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{got:?} against {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{got:?} against {want:?}");
        }
    }

    /// [[1, 2, 3], [4, 5, 6]]: small enough to check by hand, and not
    /// square, so an axis mistaken for the other shows.
    fn grid() -> Array {
        Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap()
    }

    #[test]
    fn the_unary_ops_are_the_functions_they_name() {
        let x = [0.25f32, 1.0, 4.0];
        let got = run(|| {
            let a = Array::from_slice(&x, &[3])?;
            Ok([
                values(&a.negative()?),
                values(&a.negative()?.abs()?),
                values(&a.sqrt()?),
                values(&a.rsqrt()?),
                values(&a.square()?),
                values(&a.exp()?),
                values(&a.log()?),
                values(&a.cos()?),
                values(&a.sin()?),
            ])
        });
        let want: [Vec<f32>; 9] = [
            x.map(|v| -v).to_vec(),
            x.to_vec(),
            x.map(f32::sqrt).to_vec(),
            x.map(|v| 1.0 / v.sqrt()).to_vec(),
            x.map(|v| v * v).to_vec(),
            x.map(f32::exp).to_vec(),
            x.map(f32::ln).to_vec(),
            x.map(f32::cos).to_vec(),
            x.map(f32::sin).to_vec(),
        ];
        for (got, want) in got.iter().zip(&want) {
            close(got, want);
        }
    }

    #[test]
    fn the_binary_ops_broadcast_a_scalar_and_keep_their_order() {
        let got = run(|| {
            let a = Array::from_slice(&[6.0f32, 8.0], &[2])?;
            let two = Array::from_f32(2.0)?;
            Ok([
                values(&a.add(&two)?),
                values(&a.subtract(&two)?),
                values(&two.subtract(&a)?),
                values(&a.multiply(&two)?),
                values(&a.divide(&two)?),
                values(&two.divide(&a)?),
            ])
        });
        close(&got[0], &[8.0, 10.0]);
        close(&got[1], &[4.0, 6.0]);
        close(&got[2], &[-4.0, -6.0]);
        close(&got[3], &[12.0, 16.0]);
        close(&got[4], &[3.0, 4.0]);
        close(&got[5], &[1.0 / 3.0, 0.25]);
    }

    #[test]
    fn matmul_is_rows_by_columns() {
        let (got, shape) = run(|| {
            let identityish = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2])?;
            let product = grid().matmul(&identityish)?;
            Ok((values(&product), product.shape().to_vec()))
        });
        assert_eq!(shape, vec![2, 2]);
        close(&got, &[4.0, 5.0, 10.0, 11.0]);
    }

    /// The whole-array reductions and the per-axis ones are different
    /// mlx-c functions, and the mistake this guards is calling one for the
    /// other: both compile, and the numbers differ.
    #[test]
    fn a_reduction_over_everything_is_not_one_over_an_axis() {
        let got = run(|| {
            let g = grid();
            Ok([
                (values(&g.sum(false)?), g.sum(false)?.shape().to_vec()),
                (values(&g.sum(true)?), g.sum(true)?.shape().to_vec()),
                (values(&g.sum_axes(&[1], false)?), g.sum_axes(&[1], false)?.shape().to_vec()),
                (values(&g.sum_axes(&[-1], true)?), g.sum_axes(&[-1], true)?.shape().to_vec()),
                (values(&g.sum_axes(&[0], None)?), g.sum_axes(&[0], None)?.shape().to_vec()),
            ])
        });
        assert_eq!(got[0], (vec![21.0], vec![]));
        assert_eq!(got[1], (vec![21.0], vec![1, 1]));
        assert_eq!(got[2], (vec![6.0, 15.0], vec![2]));
        assert_eq!(got[3], (vec![6.0, 15.0], vec![2, 1]));
        assert_eq!(got[4], (vec![5.0, 7.0, 9.0], vec![3]));
    }

    #[test]
    fn mean_max_and_min_reduce_the_way_sum_does() {
        let got = run(|| {
            let g = grid();
            Ok([
                values(&g.mean(false)?),
                values(&g.mean_axes(&[-1], false)?),
                values(&g.max(false)?),
                values(&g.max_axes(&[0], false)?),
                values(&g.min(None)?),
            ])
        });
        close(&got[0], &[3.5]);
        close(&got[1], &[2.0, 5.0]);
        close(&got[2], &[6.0]);
        close(&got[3], &[4.0, 5.0, 6.0]);
        close(&got[4], &[1.0]);
    }

    /// No axes, no reduction: mlx-c reads the empty list literally, and so
    /// does mlx-rs. A graph that means "everything" says `None`.
    #[test]
    fn an_empty_list_of_axes_reduces_nothing() {
        let (got, shape) = run(|| {
            let summed = grid().sum_axes(&[], false)?;
            Ok((values(&summed), summed.shape().to_vec()))
        });
        assert_eq!(shape, vec![2, 3]);
        close(&got, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn the_shape_ops_move_axes_and_not_values() {
        let got = run(|| {
            let g = grid();
            let t = g.transpose_axes(&[1, 0])?;
            let r = g.reshape(&[3, -1])?;
            let e = g.expand_dims_axes(&[1])?;
            let s = e.squeeze_axes(&[1])?;
            let last = g.expand_dims_axes(&[-1])?;
            Ok([
                (t.shape().to_vec(), values(&t)),
                (r.shape().to_vec(), values(&r)),
                (e.shape().to_vec(), values(&e)),
                (s.shape().to_vec(), values(&s)),
                (last.shape().to_vec(), values(&last)),
            ])
        });
        let flat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(got[0], (vec![3, 2], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
        assert_eq!(got[1], (vec![3, 2], flat.clone()));
        assert_eq!(got[2], (vec![2, 1, 3], flat.clone()));
        assert_eq!(got[3], (vec![2, 3], flat.clone()));
        assert_eq!(got[4], (vec![2, 3, 1], flat));
    }

    #[test]
    fn take_axis_selects_along_the_axis_it_is_given() {
        let got = run(|| {
            let g = grid();
            let rows = g.take_axis(Array::from_slice(&[1i32, 0, 1], &[3])?, 0)?;
            let cols = g.take_axis(Array::from_slice(&[2i32], &[1])?, -1)?;
            Ok([(rows.shape().to_vec(), values(&rows)), (cols.shape().to_vec(), values(&cols))])
        });
        assert_eq!(got[0], (vec![3, 3], vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        assert_eq!(got[1], (vec![2, 1], vec![3.0, 6.0]));
    }

    #[test]
    fn zeros_and_ones_have_the_dtype_asked_for() {
        let got = run(|| {
            let z = Array::zeros::<f32>(&[2, 2])?;
            let o = Array::ones::<i32>(&[3])?;
            Ok((z.dtype(), values(&z), o.dtype(), o.as_slice::<i32>()?.to_vec()))
        });
        assert_eq!(got, (Dtype::Float32, vec![0.0; 4], Dtype::Int32, vec![1, 1, 1]));
    }

    #[test]
    fn a_strided_array_reads_in_order_once_contiguous() {
        let got = run(|| Ok(grid().transpose_axes(&[1, 0])?.contiguous()?.as_slice::<f32>()?.to_vec()));
        assert_eq!(got, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn a_dtype_changes_on_the_device() {
        let got = run(|| {
            let b = grid().as_dtype(Dtype::Bfloat16)?;
            let back = b.as_dtype(Dtype::Float32)?;
            Ok((b.dtype(), values(&back)))
        });
        assert_eq!(got, (Dtype::Bfloat16, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    }

    /// MLX's refusal comes back in MLX's words, not the shim's.
    #[test]
    fn a_shape_mlx_refuses_is_an_error_that_says_why() {
        let said = run(|| Ok(grid().reshape(&[4, 2]).unwrap_err().what().to_string()));
        assert!(said.contains("reshape"), "{said}");
    }
}
