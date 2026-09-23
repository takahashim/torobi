//! The mlx-c handles that are not arrays, each freed when it goes.
//!
//! Every handle is owned from the moment mlx-c returns it, so that an
//! early `?` anywhere after that frees it on the way out. The raw handle is
//! reached only through `as_raw` (to pass in) and `as_out` (for mlx-c to
//! write into). The one place ownership changes hands is [`Vector::read`],
//! for a vector mlx-c still owns.

use super::error::{check, install, Result};
use super::{sys, Array};

/// An `mlx_stream`.
pub(crate) struct Stream(sys::mlx_stream);

impl Stream {
    /// The stream every op runs on: the default stream of the default
    /// device, the GPU unless something moved the default. The one place
    /// that decides it.
    ///
    /// Made afresh per op, as mlx-rs made it. Measured on an M-series Mac
    /// in release (2026-09-23): making and freeing one costs 0.35 µs, half
    /// of what building a trivial op costs, but ruri-v3-130m's graph is 691
    /// nodes and a step takes 140 ms, so it is about 0.2% of a step. Not
    /// worth a stream kept per thread, which would be freed at thread exit:
    /// exactly when MLX's CUDA backend is least safe to talk to.
    pub(crate) fn current() -> Self {
        install();
        unsafe {
            let mut device = sys::mlx_device_new();
            sys::mlx_get_default_device(&mut device);
            let mut stream = sys::mlx_stream_new();
            sys::mlx_get_default_stream(&mut stream, device);
            sys::mlx_device_free(device);
            Self(stream)
        }
    }

    /// The CPU's default stream, which is where MLX reads files.
    pub(crate) fn cpu() -> Self {
        install();
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
pub(crate) struct Vector(sys::mlx_vector_array);

impl Vector {
    pub(crate) fn new() -> Self {
        Self(unsafe { sys::mlx_vector_array_new() })
    }

    /// A vector holding a new reference to each array. An array that was
    /// never made is refused here, with its cause.
    pub(crate) fn of<'a>(arrays: impl IntoIterator<Item = &'a Array>) -> Result<Self> {
        let vector = Self::new();
        for array in arrays {
            check(
                unsafe { sys::mlx_vector_array_append_value(vector.0, array.raw()?) },
                "mlx_vector_array_append_value",
            )?;
        }
        Ok(vector)
    }

    pub(crate) fn as_raw(&self) -> sys::mlx_vector_array {
        self.0
    }

    /// Where mlx-c writes a vector it returns.
    pub(crate) fn as_out(&mut self) -> *mut sys::mlx_vector_array {
        &mut self.0
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

