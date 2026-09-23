//! Records what this engine is built against, so a run can say so, and
//! puts MLX's kernels where a test binary can find them.

use std::path::{Path, PathBuf};
use std::{env, fs};

fn main() {
    println!("cargo:rerun-if-changed=Cargo.toml");
    record_mlx_version();
    bind_mlx_c();
    link_metallib_beside_test_binaries();
}

/// The four functions whose declarations mlx-c gates on the architecture
/// (`mlx/c/half.h`), and the two types they take. Excluding them is what
/// lets one set of bindings serve both platforms. They read fp16 into a
/// host variable, which x86_64 has no type for; computing in fp16 is
/// untouched, and the engine converts at its boundary anyway.
const APPLE_ONLY: [&str; 6] = [
    "mlx_array_item_float16",
    "mlx_array_item_bfloat16",
    "mlx_array_data_float16",
    "mlx_array_data_bfloat16",
    "float16_t",
    "bfloat16_t",
];

/// mlx-c, bound here rather than through mlx-rs (docs/vendoring.md).
///
/// Nothing is compiled: the prefix `TOROBI_MLX_PREFIX` names already holds
/// `libmlxc.a` and the headers bindgen reads.
fn bind_mlx_c() {
    println!("cargo:rerun-if-env-changed=TOROBI_MLX_PREFIX");
    let Some(prefix) = env::var_os("TOROBI_MLX_PREFIX").map(PathBuf::from) else {
        println!("cargo:warning=TOROBI_MLX_PREFIX is not set; see ext/torobi/mlx_prebuilt.rb");
        return;
    };
    let include = prefix.join("include");
    let umbrella = include.join("mlx/c/mlx.h");
    assert!(umbrella.is_file(), "{} holds no mlx-c headers", include.display());

    println!("cargo:rustc-link-search=native={}", prefix.join("lib").display());
    for archive in ["mlx", "mlxc", "gguflib"] {
        println!("cargo:rustc-link-lib=static={archive}");
    }
    if env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple") {
        for library in ["c++", "dylib=objc", "framework=Foundation", "framework=Metal", "framework=Accelerate"] {
            println!("cargo:rustc-link-lib={library}");
        }
    } else {
        println!("cargo:rustc-link-lib=stdc++");
        link_cuda();
    }

    println!("cargo:rerun-if-changed={}", umbrella.display());
    let mut builder = bindgen::Builder::default()
        .header(umbrella.to_string_lossy())
        .clang_arg(format!("-I{}", include.display()))
        .allowlist_item("mlx_.*")
        .allowlist_item("MLX_.*")
        // The layout queries mlx-c marks private with a leading underscore.
        // `as_slice` needs `_mlx_array_is_row_contiguous`, as mlx-rs does.
        .allowlist_item("_mlx_array_.*")
        .layout_tests(false)
        .generate_comments(false);
    for name in APPLE_ONLY {
        builder = builder.blocklist_item(name);
    }
    let out = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    builder
        .generate()
        .expect("mlx-c's headers should describe themselves")
        .write_to_file(out.join("bindings.rs"))
        .expect("the bindings should be writable");
}

/// A static MLX built for CUDA leaves these to whoever links it. `stubs`
/// carries the `libcuda.so` a machine with no driver links against.
fn link_cuda() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    let root = env::var_os("CUDA_HOME")
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    for leaf in ["lib64", "lib64/stubs", "lib", "lib/stubs"] {
        let dir = root.join(leaf);
        if dir.is_dir() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }
    for library in ["cudart", "cublasLt", "cufft", "nvrtc", "cuda", "cudnn"] {
        println!("cargo:rustc-link-lib={library}");
    }
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

/// Creates `link` pointing at `named`, replacing whatever is there.
///
/// **Replaced rather than left alone when it exists.** MLX matches its
/// kernels to the library by name, and after the MLX under it moves
/// versions an old link still loads and then cannot find the kernel a step
/// asks for (`Unable to load kernel ...`). That is what a stale link from
/// the previous dependency did: the tests that did not need the moved
/// kernel passed and one did not. A link already pointing at `named` is
/// kept, so the common case still does nothing.
fn link(named: &Path, link: &Path) {
    if fs::read_link(link).is_ok_and(|target| target == named) {
        return;
    }
    if link.symlink_metadata().is_ok() {
        if let Err(e) = fs::remove_file(link) {
            println!("cargo:warning=could not replace {}: {e}", link.display());
            return;
        }
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
