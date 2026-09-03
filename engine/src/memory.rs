//! What the device is holding.
//!
//! Ruby's GC does not see device memory, so a long run needs numbers it can
//! watch and a limit it can set (docs/plan.md section 11.3). MLX's
//! allocator is process-wide, which is why these are free functions rather
//! than methods on a session: two sessions share one pool, and that is a
//! fact to report, not to hide.
//!
//! Being process-wide is also why they go through the runtime. They reach
//! the same allocator every running session is using, so they take their
//! turn like a step does. [`Memory`] is the way in; the functions under it
//! are crate-private, or a caller could reach the allocator around the
//! gate.

use anyhow::{bail, Result};

use crate::runtime::{runtime, RuntimeError};

/// The process's device memory, through the one runtime.
pub struct Memory;

impl Memory {
    /// A reading, in bytes: active, cache, peak and the limit in force.
    pub fn report() -> Result<serde_json::Value, RuntimeError> {
        runtime().execute(report)
    }

    /// Frees what the allocator holds but is not using. Returns
    /// [before, after].
    pub fn clear_cache() -> Result<(usize, usize), RuntimeError> {
        runtime().execute(clear_cache)
    }

    /// Caps what this process may allocate on the device, in bytes; 0
    /// lifts the cap. Returns the cap now in force.
    pub fn set_limit(bytes: usize) -> Result<usize, RuntimeError> {
        runtime().execute(|| set_limit(bytes))
    }

    /// Forgets the high-water mark, so the next span is measured on its
    /// own.
    pub fn reset_peak() -> Result<(), RuntimeError> {
        runtime().execute(reset_peak)
    }
}

/// A reading of the process's device memory, in bytes.
fn report() -> Result<serde_json::Value> {
    Ok(serde_json::json!({
        "active": get(mlx_sys::mlx_get_active_memory)?,
        "cache": get(mlx_sys::mlx_get_cache_memory)?,
        "peak": get(mlx_sys::mlx_get_peak_memory)?,
        "limit": get(mlx_sys::mlx_get_memory_limit)?,
    }))
}

/// Frees what the allocator is holding but not using. Returns the cache
/// size before and after, so a caller can see whether it helped.
fn clear_cache() -> Result<(usize, usize)> {
    let before = get(mlx_sys::mlx_get_cache_memory)?;
    if unsafe { mlx_sys::mlx_clear_cache() } != 0 {
        bail!("clearing the cache failed");
    }
    Ok((before, get(mlx_sys::mlx_get_cache_memory)?))
}

/// Caps what the process may allocate on the device, and returns the cap
/// now in force. Zero means no cap.
///
/// (mlx-c writes the new limit into its out-parameter, not the old one, so
/// a caller that wants to restore a limit reads it first.)
fn set_limit(bytes: usize) -> Result<usize> {
    let mut current = 0usize;
    if unsafe { mlx_sys::mlx_set_memory_limit(&mut current, bytes) } != 0 {
        bail!("setting the memory limit failed");
    }
    Ok(current)
}

/// Forgets the high-water mark, so a later reading is about what follows.
fn reset_peak() -> Result<()> {
    if unsafe { mlx_sys::mlx_reset_peak_memory() } != 0 {
        bail!("resetting the peak failed");
    }
    Ok(())
}

fn get(f: unsafe extern "C" fn(*mut usize) -> std::os::raw::c_int) -> Result<usize> {
    let mut value = 0usize;
    if unsafe { f(&mut value) } != 0 {
        bail!("reading device memory failed");
    }
    Ok(value)
}
