//! MLX's allocator, which is process-wide.
//!
//! mlx-rs had nothing for these, so the engine called mlx-sys directly
//! (`crate::memory`). They are here now so that a failure reports what MLX
//! said, through the same handler as everything else.

use super::error::{check, install, Result};
use super::sys;

fn read(call: unsafe extern "C" fn(*mut usize) -> std::ffi::c_int, name: &str) -> Result<usize> {
    ready();
    let mut bytes = 0usize;
    check(unsafe { call(&mut bytes) }, name)?;
    Ok(bytes)
}

pub fn active() -> Result<usize> {
    read(sys::mlx_get_active_memory, "mlx_get_active_memory")
}

pub fn cache() -> Result<usize> {
    read(sys::mlx_get_cache_memory, "mlx_get_cache_memory")
}

pub fn peak() -> Result<usize> {
    read(sys::mlx_get_peak_memory, "mlx_get_peak_memory")
}

pub fn limit() -> Result<usize> {
    read(sys::mlx_get_memory_limit, "mlx_get_memory_limit")
}

/// Every call here starts with this: these are the one kind of mlx-c
/// call that neither makes a handle nor is handed one (`error::install`).
fn ready() {
    install();
}

pub fn clear_cache() -> Result<()> {
    ready();
    check(unsafe { sys::mlx_clear_cache() }, "mlx_clear_cache")
}

pub fn reset_peak() -> Result<()> {
    ready();
    check(unsafe { sys::mlx_reset_peak_memory() }, "mlx_reset_peak_memory")
}

/// Caps allocation at `bytes` (0 lifts the cap) and returns the cap now in
/// force: mlx-c writes the new limit into its out-parameter, not the old.
pub fn set_limit(bytes: usize) -> Result<usize> {
    ready();
    let mut now = 0usize;
    check(unsafe { sys::mlx_set_memory_limit(&mut now, bytes) }, "mlx_set_memory_limit")?;
    Ok(now)
}
