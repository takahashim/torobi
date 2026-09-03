//! Records what this engine is built against, so a run can say so, and
//! puts MLX's kernels where a test binary can find them.

use std::path::{Path, PathBuf};
use std::{env, fs};

fn main() {
    println!("cargo:rerun-if-changed=Cargo.toml");
    record_mlx_revision();
    link_metallib_beside_test_binaries();
}

/// The vendoring ledger asks that every artifact report its revisions
/// (docs/vendoring.md). The mlx-rs pin lives in Cargo.toml as a git rev;
/// read it here rather than repeating it in the source, where the two
/// would drift.
fn record_mlx_revision() {
    let manifest = fs::read_to_string("Cargo.toml").expect("reading Cargo.toml");
    let rev = manifest
        .lines()
        .find(|line| line.trim_start().starts_with("mlx-rs"))
        .and_then(|line| line.split("rev = \"").nth(1))
        .and_then(|rest| rest.split('"').next())
        .unwrap_or("unpinned");
    println!("cargo:rustc-env=TOROBI_MLX_RS_REV={rev}");
}

/// MLX finds its Metal kernels through dladdr, so mlx.metallib has to sit
/// beside whichever binary loaded it. mlx-sys puts it in the profile
/// directory, which serves `target/release/torobi-engine` but not a test
/// binary, which cargo builds into `deps/`. Without this, every test that
/// touches an array aborts the process with "Failed to load the default
/// metallib" (and it aborts, so no test can report it).
///
/// A symlink rather than a copy: the file is 105MB.
fn link_metallib_beside_test_binaries() {
    let Some(profile_dir) = profile_dir() else {
        return;
    };
    let source = profile_dir.join("mlx.metallib");
    if !source.exists() {
        // A clean build runs mlx-sys's script first, but if the layout
        // ever changes, say so rather than failing the build.
        println!("cargo:warning=no mlx.metallib in {}", profile_dir.display());
        return;
    }
    let deps = profile_dir.join("deps");
    if !deps.is_dir() {
        return;
    }
    let link = deps.join("mlx.metallib");
    if link.exists() || link.symlink_metadata().is_ok() {
        return;
    }
    if let Err(e) = symlink(Path::new("../mlx.metallib"), &link) {
        println!("cargo:warning=could not link mlx.metallib into deps/: {e}");
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
