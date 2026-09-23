//! `Array`: an `mlx_array` handle, freed when it goes, and the ways in and
//! out of one.
//!
//! The operations on it are in `ops`. What is here is what every one of
//! them is built from: taking ownership of what mlx-c returns
//! ([`Array::try_from_op`]), and copying values in and out.

use std::ffi::c_int;

use super::error::{check, install, Exception, Result};
use super::handle::Stream;
use super::sys;

/// An `mlx_array`. One reference to an array MLX owns; dropping it gives
/// the reference back.
pub struct Array(sys::mlx_array);

// MLX arrays may move between threads; what may not happen is two threads
// submitting work at once, and the runtime's gate is what stops that
// (`crate::runtime`). mlx-rs makes the same claim.
unsafe impl Send for Array {}

impl Drop for Array {
    fn drop(&mut self) {
        unsafe { sys::mlx_array_free(self.0) };
    }
}

impl Clone for Array {
    /// Another reference to the same array, not a copy of its data.
    fn clone(&self) -> Self {
        Array::try_from_op(|res| unsafe { sys::mlx_array_set(res, self.0) })
            .expect("mlx_array_set only fails when out of memory")
    }
}

impl AsRef<Array> for Array {
    fn as_ref(&self) -> &Array {
        self
    }
}

impl From<f32> for Array {
    fn from(value: f32) -> Self {
        Array::from_f32(value)
    }
}

/// The dtype and shape, never the values: those are on the device, and
/// printing them would evaluate the array.
impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Array({:?} {:?})", self.dtype(), self.shape())
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
    ///
    /// The array is owned before the call rather than after, so a failing
    /// call frees whatever it may have half-written on the way out.
    #[track_caller]
    pub(crate) fn try_from_op(op: impl FnOnce(*mut sys::mlx_array) -> c_int) -> Result<Array> {
        install();
        let mut out = Array(unsafe { sys::mlx_array_new() });
        let status = op(&mut out.0);
        check(status, "an MLX operation")?;
        Ok(out)
    }

    pub(crate) fn as_raw(&self) -> sys::mlx_array {
        self.0
    }

    pub fn from_f32(value: f32) -> Array {
        Array(unsafe { sys::mlx_array_new_float32(value) })
    }

    /// An empty handle, which is how mlx-c spells "no array" for an
    /// optional argument.
    pub(crate) fn empty() -> Array {
        Array(unsafe { sys::mlx_array_new() })
    }

    pub(crate) fn from_i32(value: i32) -> Array {
        Array(unsafe { sys::mlx_array_new_int(value) })
    }

    /// An array holding a copy of `data`, in row-major order.
    ///
    /// Panics when the data does not fill the shape, as mlx-rs does: that
    /// is a caller's arithmetic, not something MLX can report.
    pub fn from_slice<T: ArrayElement>(data: &[T], shape: &[i32]) -> Array {
        let wanted: i64 = shape.iter().map(|&d| d as i64).product();
        assert_eq!(data.len() as i64, wanted, "{} values for shape {shape:?}", data.len());
        Array(unsafe {
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
        install();
        check(unsafe { sys::mlx_array_eval(self.0) }, "mlx_array_eval")
    }

    /// The array in another dtype. Here rather than in `ops` because
    /// reading a value needs it.
    pub fn as_dtype(&self, dtype: Dtype) -> Result<Array> {
        let stream = Stream::default_device();
        Array::try_from_op(|res| unsafe {
            sys::mlx_astype(res, self.0, dtype.to_raw(), stream.as_raw())
        })
    }

    /// The single value this holds, converted to `T` on the device first
    /// when it is held as something else.
    ///
    /// Panics on anything but one value or on MLX refusing, as mlx-rs's
    /// `item_cast` does; `try_item_cast` is the fallible form.
    pub fn item_cast<T: ArrayElement>(&self) -> T {
        self.try_item_cast().unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_item_cast<T: ArrayElement>(&self) -> Result<T> {
        if self.size() != 1 {
            return Err(Exception::custom(format!(
                "an item was asked of an array of {} values",
                self.size()
            )));
        }
        self.eval()?;
        if self.dtype() != T::DTYPE {
            return self.as_dtype(T::DTYPE)?.try_item_cast();
        }
        let mut out = std::mem::MaybeUninit::<T>::uninit();
        check(unsafe { T::item(out.as_mut_ptr(), self.0) }, "reading an item")?;
        Ok(unsafe { out.assume_init() })
    }

    /// The values, borrowed from the evaluated array.
    ///
    /// Panics where mlx-rs's `as_slice` does: on another dtype, and on a
    /// layout that is not row-major contiguous (call `contiguous` first).
    /// An empty array reads as an empty slice, where mlx-rs panicked.
    pub fn as_slice<T: ArrayElement>(&self) -> &[T] {
        self.try_as_slice().unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_as_slice<T: ArrayElement>(&self) -> Result<&[T]> {
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
            let f = Array::from_slice(&[1.5f32, -2.0, 0.0], &[3]);
            let i = Array::from_slice(&[7i32, 0, -1, 4], &[2, 2]);
            let b = Array::from_slice(&[true, false], &[2]);
            assert_eq!(f.shape(), &[3]);
            assert_eq!(i.shape(), &[2, 2]);
            assert_eq!(i.dtype(), Dtype::Int32);
            Ok((
                f.as_slice::<f32>().to_vec(),
                i.as_slice::<i32>().to_vec(),
                b.as_slice::<bool>().to_vec(),
            ))
        });
        assert_eq!(floats, vec![1.5, -2.0, 0.0]);
        assert_eq!(ints, vec![7, 0, -1, 4]);
        assert_eq!(flags, vec![true, false]);
    }

    #[test]
    fn a_scalar_has_no_shape_and_one_item() {
        let (shape, ndim, value) = run(|| {
            let x = Array::from_f32(2.5);
            Ok((x.shape().to_vec(), x.ndim(), x.item_cast::<f32>()))
        });
        assert_eq!((shape, ndim, value), (vec![], 0, 2.5));
    }

    #[test]
    fn an_item_is_converted_on_the_device_when_the_dtype_differs() {
        let value = run(|| Ok(Array::from_slice(&[3i32], &[1]).item_cast::<f32>()));
        assert_eq!(value, 3.0);
    }

    #[test]
    fn an_item_of_many_values_is_refused_rather_than_guessed() {
        let refused = run(|| Ok(Array::from_slice(&[1.0f32, 2.0], &[2]).try_item_cast::<f32>().is_err()));
        assert!(refused);
    }

    #[test]
    fn a_slice_of_another_dtype_is_refused() {
        let said = run(|| {
            Ok(Array::from_slice(&[1i32], &[1]).try_as_slice::<f32>().unwrap_err().to_string())
        });
        assert!(said.contains("Float32") && said.contains("Int32"), "{said}");
    }

    #[test]
    fn an_empty_array_reads_as_nothing() {
        let got = run(|| Ok(Array::from_slice::<f32>(&[], &[0, 3]).as_slice::<f32>().len()));
        assert_eq!(got, 0);
    }

    #[test]
    fn a_clone_is_another_reference_to_the_same_values() {
        let got = run(|| {
            let a = Array::from_slice(&[4.0f32, 5.0], &[2]);
            let b = a.clone();
            drop(a);
            Ok(b.as_slice::<f32>().to_vec())
        });
        assert_eq!(got, vec![4.0, 5.0]);
    }

    #[test]
    fn every_dtype_survives_the_round_trip_to_mlx_c() {
        for (dtype, _) in Dtype::TABLE {
            assert_eq!(Dtype::from_raw(dtype.to_raw()), dtype);
        }
    }
}
