//! Records what this engine is built against, so a run can say so, and
//! puts MLX's kernels where a test binary can find them.

use std::path::{Path, PathBuf};
use std::{env, fs};

fn main() {
    println!("cargo:rerun-if-changed=Cargo.toml");
    record_mlx_version();
    link_metallib_beside_test_binaries();
}

/// The vendoring ledger asks that every artifact report which MLX it is
/// built against (docs/vendoring.md). The mlx-rs pin now lives in
/// Cargo.toml as a version rather than a git revision, since it comes
/// from crates.io; read it here rather than repeating it in the source,
/// where the two would drift. Cargo.lock is what fixes the commit.
fn record_mlx_version() {
    let manifest = fs::read_to_string("Cargo.toml").expect("reading Cargo.toml");
    for (crate_name, var) in [
        ("mlx-rs", "TOROBI_MLX_RS_VERSION"),
        ("mlx-sys", "TOROBI_MLX_SYS_VERSION"),
    ] {
        // The first quoted string on the line is the version, whether it
        // is written `mlx-sys = "=0.6.0"` or `mlx-rs = { version = "=..." }`.
        let version = manifest
            .lines()
            .find(|line| line.trim_start().starts_with(crate_name))
            .and_then(|line| line.split('"').nth(1))
            .map(|value| value.trim_start_matches('='))
            .unwrap_or("unpinned");
        println!("cargo:rustc-env={var}={version}");
    }
}

/// MLX finds its Metal kernels through dladdr, so mlx.metallib has to sit
/// beside whichever binary loaded it: `target/<profile>/torobi-engine`
/// for the command line, and `target/<profile>/deps/` for the test
/// binaries cargo puts there. Without it every one of those aborts with
/// "Failed to load the default metallib" (and it aborts, so nothing can
/// even report it).
///
/// Where the metallib is depends on how MLX was had. Upstream mlx-sys
/// writes it to `MLX_RS_METAL_PATH` when MLX is built from source, and
/// Torobi points that variable at the pre-built prefix's `lib/` when one
/// is used (ext/torobi/mlx_prebuilt.rb), so both arrive here through the
/// same variable. Falling back to the profile directory keeps a build
/// that sets nothing (a plain `cargo test` with a source MLX) working,
/// and in that case the file is already where it belongs.
///
/// Symlinks rather than copies: the file is 105MB.
fn link_metallib_beside_test_binaries() {
    let Some(profile_dir) = profile_dir() else {
        return;
    };
    let from_env = env::var_os("MLX_RS_METAL_PATH").map(PathBuf::from);
    let named = from_env
        .as_deref()
        .unwrap_or(profile_dir.as_path())
        .join("mlx.metallib");
    if !named.exists() {
        // The print is the point: silently having no metallib means the
        // binaries abort rather than fail, which reports nothing at all.
        println!("cargo:warning=no mlx.metallib at {}", named.display());
        return;
    }
    // Beside the binary the command line runs...
    let binary_side = profile_dir.join("mlx.metallib");
    if named != binary_side {
        link(&named, &binary_side);
    }
    // ...and beside the test binaries.
    let deps = profile_dir.join("deps");
    if deps.is_dir() {
        link(&named, &deps.join("mlx.metallib"));
    }
}

/// Creates `link` pointing at `named`, unless something is already there.
fn link(named: &Path, link: &Path) {
    if link.exists() || link.symlink_metadata().is_ok() {
        return;
    }
    if let Err(e) = symlink(named, link) {
        println!("cargo:warning=could not link {}: {e}", named.display());
    }
}

#[cfg(unix)]
fn symlink(original: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(original, link)
}

#[cfg(not(unix))]
fn symlink(original: &Path, link: &Path) -> std::io::Result<()> {
    fs::copy(link.parent().unwrap().join(original), link).map(|_| ())
}

/// `target/<triple>/<profile>`, walked back out of OUT_DIR. Cargo gives
/// build scripts no direct way to ask.
fn profile_dir() -> Option<PathBuf> {
    let out = PathBuf::from(env::var("OUT_DIR").ok()?);
    // .../target/<profile>/build/<pkg>-<hash>/out
    Some(out.parent()?.parent()?.parent()?.to_path_buf())
}
