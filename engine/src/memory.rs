//! What the device is holding.
//!
//! Ruby's GC does not see device memory, so a long run needs numbers it can
//! watch and a limit it can set (docs/plan.md section 11.3). MLX's
//! allocator is process-wide, which is why these are free functions rather
//! than methods on a session: two sessions share one pool, and that is a
//! fact to report, not to hide.

use anyhow::{bail, Result};

/// A reading of the process's device memory, in bytes.
pub fn report() -> Result<serde_json::Value> {
    Ok(serde_json::json!({
        "active": get(mlx_sys::mlx_get_active_memory)?,
        "cache": get(mlx_sys::mlx_get_cache_memory)?,
        "peak": get(mlx_sys::mlx_get_peak_memory)?,
        "limit": get(mlx_sys::mlx_get_memory_limit)?,
    }))
}

/// Frees what the allocator is holding but not using. Returns the cache
/// size before and after, so a caller can see whether it helped.
pub fn clear_cache() -> Result<(usize, usize)> {
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
pub fn set_limit(bytes: usize) -> Result<usize> {
    let mut current = 0usize;
    if unsafe { mlx_sys::mlx_set_memory_limit(&mut current, bytes) } != 0 {
        bail!("setting the memory limit failed");
    }
    Ok(current)
}

/// Forgets the high-water mark, so a later reading is about what follows.
pub fn reset_peak() -> Result<()> {
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
