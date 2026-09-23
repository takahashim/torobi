//! The mlx-c handles that are not arrays, each freed when it goes.
//!
//! Every handle is owned from the moment mlx-c returns it, so that an
//! early `?` anywhere after that frees it on the way out. The only places
//! ownership changes hands are named: [`Vector::read`] for a vector mlx-c
//! still owns, and the trampoline's hand-over in `transforms`.

use super::error::{check, Result};
use super::{sys, Array};

/// An `mlx_stream`.
pub(crate) struct Stream(sys::mlx_stream);

impl Stream {
    /// The default stream of the default device, which is what mlx-rs asks
    /// for on every op (`Stream::thread_local_or_default`): the GPU unless
    /// something moved the default.
    pub(crate) fn default_device() -> Self {
        unsafe {
            let mut device = sys::mlx_device_new();
            sys::mlx_get_default_device(&mut device);
            let mut stream = sys::mlx_stream_new();
            sys::mlx_get_default_stream(&mut stream, device);
            sys::mlx_device_free(device);
            Self(stream)
        }
    }

    /// The CPU's default stream, which is where mlx-rs loads safetensors.
    pub(crate) fn cpu() -> Self {
        Self(unsafe { sys::mlx_default_cpu_stream_new() })
    }

    pub(crate) fn as_raw(&self) -> sys::mlx_stream {
        self.0
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { sys::mlx_stream_free(self.0) };
    }
}

/// An `mlx_vector_array` this side owns.
pub(crate) struct Vector(pub(crate) sys::mlx_vector_array);

impl Vector {
    pub(crate) fn new() -> Self {
        Self(unsafe { sys::mlx_vector_array_new() })
    }

    /// A vector holding a new reference to each array.
    pub(crate) fn of<'a>(arrays: impl IntoIterator<Item = &'a Array>) -> Result<Self> {
        let vector = Self::new();
        for array in arrays {
            check(
                unsafe { sys::mlx_vector_array_append_value(vector.0, array.as_raw()) },
                "mlx_vector_array_append_value",
            )?;
        }
        Ok(vector)
    }

    /// The arrays inside, each one owned by the caller.
    pub(crate) fn arrays(&self) -> Result<Vec<Array>> {
        unsafe { Self::read(self.0) }
    }

    /// The arrays inside a vector that belongs to someone else.
    ///
    /// `mlx_vector_array_get` hands out a new reference rather than a
    /// borrow, so what comes back is the caller's to free, and the vector
    /// is left alone.
    ///
    /// # Safety
    ///
    /// `raw` must be a live vector for the duration of the call.
    pub(crate) unsafe fn read(raw: sys::mlx_vector_array) -> Result<Vec<Array>> {
        let count = unsafe { sys::mlx_vector_array_size(raw) };
        (0..count)
            .map(|index| {
                Array::try_from_op(|res| unsafe { sys::mlx_vector_array_get(res, raw, index) })
            })
            .collect()
    }
}

impl Drop for Vector {
    fn drop(&mut self) {
        unsafe { sys::mlx_vector_array_free(self.0) };
    }
}
