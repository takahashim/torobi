//! Where MLX finds its Metal kernels.
//!
//! MLX looks for `mlx.metallib` beside its own code, through `dladdr`,
//! unless it is told otherwise first. `mlx_metal_set_metallib_path` is that
//! telling: set before the device initializes, the named file is tried
//! first and the default search is the fallback (mlx-swift issue #415).
//!
//! It exists so a platform gem can fetch the kernels into a directory it
//! can write and point MLX there, rather than writing beside the installed
//! bundle, which the gem's own directory need not allow.

use super::error::{check, install, Exception, Result};
use super::sys;

/// Points MLX at `mlx.metallib` before it looks.
pub fn set_metallib_path(path: &str) -> Result<()> {
    install();
    let path = std::ffi::CString::new(path)
        .map_err(|_| Exception::custom("a metallib path may not hold a NUL"))?;
    check(
        unsafe { sys::mlx_metal_set_metallib_path(path.as_ptr()) },
        "mlx_metal_set_metallib_path",
    )
}
