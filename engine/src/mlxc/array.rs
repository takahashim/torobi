//! `Array`: an `mlx_array` handle, freed when it goes, and the ways in and
//! out of one.
//!
//! **An `Array` always holds an array.** mlx-c says "no array" with an
//! empty handle, and hands one back when a constructor fails; neither ever
//! becomes an `Array`. The slot mlx-c writes a result into is an [`Out`]
//! until the call has succeeded, constructors return `Result`, and the
//! empty handle an optional argument wants is its own type
//! (`fast::NoArray`). So nothing that takes an `Array` has to ask whether
//! it is one.
//!
//! The operations on it are in `ops`. What is here is what every one of
//! them is built from ([`Array::on_stream`], [`Array::try_from_op`]), and
//! copying values in and out.

use std::ffi::c_int;

use super::error::{check, failure, install, Exception, Result};
use super::handle::Stream;
use super::sys;

/// An `mlx_array`. One reference to an array MLX owns; dropping it gives
/// the reference back.
pub struct Array(sys::mlx_array);

// MLX arrays may move between threads; what may not happen is two threads
// submitting work at once, and the runtime's gate is what stops that
// (`crate::runtime`).
unsafe impl Send for Array {}

impl Drop for Array {
    fn drop(&mut self) {
        unsafe { sys::mlx_array_free(self.0) };
    }
}

impl Clone for Array {
    /// Another reference to the same array, not a copy of its data.
    fn clone(&self) -> Self {
        Array::try_from_op("mlx_array_set", |res| unsafe { sys::mlx_array_set(res, self.0) })
            .expect("mlx_array_set only fails when out of memory")
    }
}

impl AsRef<Array> for Array {
    fn as_ref(&self) -> &Array {
        self
    }
}

/// The dtype and shape, never the values: those are on the device, and
/// printing them would evaluate the array.
impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Array({:?} {:?})", self.dtype(), self.shape())
    }
}

/// The slot an mlx-c call writes an array into.
///
/// Empty until the call succeeds, and freed whatever happens: the handle
/// is owned before the call, so one that fails halfway through writing
/// leaks nothing. It becomes an [`Array`] only through [`Out::written`],
/// which is where "the call failed" and "the call returned nothing" both
/// become errors.
pub(crate) struct Out(sys::mlx_array);

impl Out {
    pub(crate) fn new() -> Self {
        Out(unsafe { sys::mlx_array_new() })
    }

    pub(crate) fn as_mut(&mut self) -> *mut sys::mlx_array {
        &mut self.0
    }

    /// The array the call wrote, given the status it returned.
    #[track_caller]
    pub(crate) fn written(self, status: c_int, operation: &str) -> Result<Array> {
        check(status, operation)?;
        if self.0.ctx.is_null() {
            return Err(failure(operation));
        }
        let array = Array(self.0);
        std::mem::forget(self);
        Ok(array)
    }
}

impl Drop for Out {
    fn drop(&mut self) {
        unsafe { sys::mlx_array_free(self.0) };
    }
}

/// MLX's dtypes, under the names mlx-rs gave them, since those names reach
/// messages and checkpoints' error text through `{:?}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    Bool,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
    Bfloat16,
    Complex64,
}

impl Dtype {
    /// Both directions of the mapping, in one place. A `#[repr(u32)]` enum
    /// would give one direction for free and still need this for the other.
    const TABLE: [(Dtype, sys::mlx_dtype); 14] = [
        (Dtype::Bool, sys::mlx_dtype__MLX_BOOL),
        (Dtype::Uint8, sys::mlx_dtype__MLX_UINT8),
        (Dtype::Uint16, sys::mlx_dtype__MLX_UINT16),
        (Dtype::Uint32, sys::mlx_dtype__MLX_UINT32),
        (Dtype::Uint64, sys::mlx_dtype__MLX_UINT64),
        (Dtype::Int8, sys::mlx_dtype__MLX_INT8),
        (Dtype::Int16, sys::mlx_dtype__MLX_INT16),
        (Dtype::Int32, sys::mlx_dtype__MLX_INT32),
        (Dtype::Int64, sys::mlx_dtype__MLX_INT64),
        (Dtype::Float16, sys::mlx_dtype__MLX_FLOAT16),
        (Dtype::Float32, sys::mlx_dtype__MLX_FLOAT32),
        (Dtype::Float64, sys::mlx_dtype__MLX_FLOAT64),
        (Dtype::Bfloat16, sys::mlx_dtype__MLX_BFLOAT16),
        (Dtype::Complex64, sys::mlx_dtype__MLX_COMPLEX64),
    ];

    pub(crate) fn to_raw(self) -> sys::mlx_dtype {
        Self::TABLE.iter().find(|(d, _)| *d == self).map(|(_, raw)| *raw).unwrap()
    }

    fn from_raw(raw: sys::mlx_dtype) -> Dtype {
        Self::TABLE
            .iter()
            .find(|(_, r)| *r == raw)
            .map(|(d, _)| *d)
            .unwrap_or_else(|| panic!("mlx-c reported dtype {raw}, which this build does not know"))
    }
}

/// A Rust type an array's values can be read into or made from.
///
/// f16 and bf16 are not here and cannot be: mlx-c declares their
/// accessors only on ARM (`mlx/c/half.h`), and reading half precision into
/// a host variable is not something the engine does. It converts on the
/// device first.
pub trait ArrayElement: Copy {
    const DTYPE: Dtype;
    #[doc(hidden)]
    unsafe fn item(out: *mut Self, array: sys::mlx_array) -> c_int;
    #[doc(hidden)]
    unsafe fn data(array: sys::mlx_array) -> *const Self;
}

macro_rules! element {
    ($t:ty, $dtype:ident, $item:ident, $data:ident) => {
        impl ArrayElement for $t {
            const DTYPE: Dtype = Dtype::$dtype;
            unsafe fn item(out: *mut Self, array: sys::mlx_array) -> c_int {
                unsafe { sys::$item(out, array) }
            }
            unsafe fn data(array: sys::mlx_array) -> *const Self {
                unsafe { sys::$data(array) }
            }
        }
    };
}

element!(f32, Float32, mlx_array_item_float32, mlx_array_data_float32);
element!(i32, Int32, mlx_array_item_int32, mlx_array_data_int32);
element!(u32, Uint32, mlx_array_item_uint32, mlx_array_data_uint32);
element!(bool, Bool, mlx_array_item_bool, mlx_array_data_bool);

impl Array {
    /// Runs an mlx-c call that writes a new array, and owns what it wrote.
    /// `operation` names the call for when MLX fails without saying why.
    #[track_caller]
    pub(crate) fn try_from_op(
        operation: &str,
        op: impl FnOnce(*mut sys::mlx_array) -> c_int,
    ) -> Result<Array> {
        install();
        let mut out = Out::new();
        let status = op(out.as_mut());
        out.written(status, operation)
    }

    /// The same, for the calls that run on a stream, which is every op:
    /// on [`Stream::current`], the one place that decides which.
    #[track_caller]
    pub(crate) fn on_stream(
        operation: &str,
        op: impl FnOnce(*mut sys::mlx_array, sys::mlx_stream) -> c_int,
    ) -> Result<Array> {
        let stream = Stream::current()?;
        Array::try_from_op(operation, |res| op(res, stream.as_raw()))
    }

    pub(crate) fn as_raw(&self) -> sys::mlx_array {
        self.0
    }

    /// Owns what a constructor made. Constructors return no status: on
    /// failure mlx-c reports through the handler and hands back an empty
    /// handle, which is turned into the error it was, here, rather than
    /// into an `Array`. On CUDA this happens for real: making an array
    /// allocates device memory there, and with no usable device it fails.
    #[track_caller]
    fn made(constructor: &str, make: impl FnOnce() -> sys::mlx_array) -> Result<Array> {
        install();
        let raw = make();
        if raw.ctx.is_null() {
            Err(failure(constructor))
        } else {
            Ok(Array(raw))
        }
    }

    pub fn from_f32(value: f32) -> Result<Array> {
        Array::made("mlx_array_new_float32", || unsafe { sys::mlx_array_new_float32(value) })
    }

    pub fn from_i32(value: i32) -> Result<Array> {
        Array::made("mlx_array_new_int", || unsafe { sys::mlx_array_new_int(value) })
    }

    /// An array holding a copy of `data`, in row-major order.
    ///
    /// Refused when the data does not fill the shape, before mlx-c is asked,
    /// since it would read past the end of `data`.
    pub fn from_slice<T: ArrayElement>(data: &[T], shape: &[i32]) -> Result<Array> {
        let wanted: i64 = shape.iter().map(|&d| d as i64).product();
        if data.len() as i64 != wanted {
            return Err(Exception::custom(format!(
                "{} values for shape {shape:?}, which holds {wanted}",
                data.len()
            )));
        }
        Array::made("mlx_array_new_data", || unsafe {
            sys::mlx_array_new_data(
                data.as_ptr().cast(),
                shape.as_ptr(),
                shape.len() as c_int,
                T::DTYPE.to_raw(),
            )
        })
    }

    pub fn ndim(&self) -> usize {
        unsafe { sys::mlx_array_ndim(self.0) }
    }

    pub fn size(&self) -> usize {
        unsafe { sys::mlx_array_size(self.0) }
    }

    pub fn shape(&self) -> &[i32] {
        let ndim = self.ndim();
        if ndim == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(sys::mlx_array_shape(self.0), ndim) }
    }

    pub fn dtype(&self) -> Dtype {
        Dtype::from_raw(unsafe { sys::mlx_array_dtype(self.0) })
    }

    pub fn eval(&self) -> Result<()> {
        check(unsafe { sys::mlx_array_eval(self.0) }, "mlx_array_eval")
    }

    /// The single value this holds, converted to `T` on the device first
    /// when it is held as something else. Refused for anything but one
    /// value.
    pub fn item<T: ArrayElement>(&self) -> Result<T> {
        if self.size() != 1 {
            return Err(Exception::custom(format!(
                "an item was asked of an array of {} values",
                self.size()
            )));
        }
        self.eval()?;
        if self.dtype() != T::DTYPE {
            return self.as_dtype(T::DTYPE)?.item();
        }
        let mut out = std::mem::MaybeUninit::<T>::uninit();
        check(unsafe { T::item(out.as_mut_ptr(), self.0) }, "reading an item")?;
        Ok(unsafe { out.assume_init() })
    }

    /// The values, borrowed from the evaluated array.
    ///
    /// Refused for another dtype, and for a layout that is not row-major
    /// contiguous (call `contiguous` first). An empty array reads as an
    /// empty slice.
    pub fn as_slice<T: ArrayElement>(&self) -> Result<&[T]> {
        if self.dtype() != T::DTYPE {
            return Err(Exception::custom(format!(
                "dtype mismatch: expected {:?}, found {:?}",
                T::DTYPE,
                self.dtype()
            )));
        }
        self.eval()?;
        let size = self.size();
        if size == 0 {
            return Ok(&[]);
        }
        let mut row_major = false;
        check(
            unsafe { sys::_mlx_array_is_row_contiguous(&mut row_major, self.0) },
            "_mlx_array_is_row_contiguous",
        )?;
        if !row_major {
            return Err(Exception::custom(
                "array data is not contiguous row-major; call `contiguous()` first",
            ));
        }
        let data = unsafe { T::data(self.0) };
        if data.is_null() {
            return Err(Exception::custom("the data pointer is null"));
        }
        Ok(unsafe { std::slice::from_raw_parts(data, size) })
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

    #[test]
    fn values_go_in_and_come_back_out() {
        let (floats, ints, flags) = run(|| {
            let f = Array::from_slice(&[1.5f32, -2.0, 0.0], &[3])?;
            let i = Array::from_slice(&[7i32, 0, -1, 4], &[2, 2])?;
            let b = Array::from_slice(&[true, false], &[2])?;
            assert_eq!(f.shape(), &[3]);
            assert_eq!(i.shape(), &[2, 2]);
            assert_eq!(i.dtype(), Dtype::Int32);
            Ok((
                f.as_slice::<f32>()?.to_vec(),
                i.as_slice::<i32>()?.to_vec(),
                b.as_slice::<bool>()?.to_vec(),
            ))
        });
        assert_eq!(floats, vec![1.5, -2.0, 0.0]);
        assert_eq!(ints, vec![7, 0, -1, 4]);
        assert_eq!(flags, vec![true, false]);
    }

    #[test]
    fn a_scalar_has_no_shape_and_one_item() {
        let (shape, ndim, value) = run(|| {
            let x = Array::from_f32(2.5)?;
            Ok((x.shape().to_vec(), x.ndim(), x.item::<f32>()?))
        });
        assert_eq!((shape, ndim, value), (vec![], 0, 2.5));
    }

    #[test]
    fn an_item_is_converted_on_the_device_when_the_dtype_differs() {
        let value = run(|| Array::from_slice(&[3i32], &[1])?.item::<f32>());
        assert_eq!(value, 3.0);
    }

    #[test]
    fn an_item_of_many_values_is_refused_rather_than_guessed() {
        let refused = run(|| Ok(Array::from_slice(&[1.0f32, 2.0], &[2])?.item::<f32>().is_err()));
        assert!(refused);
    }

    #[test]
    fn a_slice_of_another_dtype_is_refused() {
        let said = run(|| Ok(Array::from_slice(&[1i32], &[1])?.as_slice::<f32>().unwrap_err().to_string()));
        assert!(said.contains("Float32") && said.contains("Int32"), "{said}");
    }

    #[test]
    fn an_empty_array_reads_as_nothing() {
        let got = run(|| Ok(Array::from_slice::<f32>(&[], &[0, 3])?.as_slice::<f32>()?.len()));
        assert_eq!(got, 0);
    }

    #[test]
    fn data_that_does_not_fill_its_shape_is_refused_before_mlx_reads_it() {
        let said = run(|| Ok(Array::from_slice(&[1.0f32, 2.0], &[3]).unwrap_err().to_string()));
        assert!(said.contains("2 values for shape [3]"), "{said}");
    }

    #[test]
    fn a_clone_is_another_reference_to_the_same_values() {
        let got = run(|| {
            let a = Array::from_slice(&[4.0f32, 5.0], &[2])?;
            let b = a.clone();
            drop(a);
            Ok(b.as_slice::<f32>()?.to_vec())
        });
        assert_eq!(got, vec![4.0, 5.0]);
    }

    /// A constructor mlx-c could not serve is an error, in MLX's words, at
    /// the constructor, rather than an array that fails later.
    ///
    /// On a Mac making an array cannot fail, so the failure is staged: MLX's
    /// report, then the empty handle a failing constructor returns.
    #[test]
    fn a_constructor_that_fails_is_an_error_where_it_failed() {
        let said = run(|| {
            crate::mlxc::error::report("cudaMallocManaged(&data, size) failed");
            Ok(Array::made("mlx_array_new_float32", || unsafe { sys::mlx_array_new() })
                .unwrap_err()
                .what()
                .to_string())
        });
        assert!(said.starts_with("cudaMallocManaged"), "{said}");
    }

    /// A call that succeeds and writes nothing is an error too, not an
    /// array: the other way an empty handle could have got in.
    #[test]
    fn a_call_that_writes_nothing_is_an_error() {
        let said = run(|| Ok(Array::try_from_op("nothing", |_| 0).unwrap_err().what().to_string()));
        assert_eq!(said, "nothing failed");
    }

    #[test]
    fn every_dtype_survives_the_round_trip_to_mlx_c() {
        for (dtype, _) in Dtype::TABLE {
            assert_eq!(Dtype::from_raw(dtype.to_raw()), dtype);
        }
    }
}
