//! Records what this engine is built against, so a run can say so, and
//! puts MLX's kernels where a test binary can find them.

use std::path::{Path, PathBuf};
use std::{env, fs};

fn main() {
    println!("cargo:rerun-if-env-changed=TOROBI_MLX_PREFIX");
    let prefix = env::var_os("TOROBI_MLX_PREFIX").map(PathBuf::from);
    record_mlx_version(prefix.as_deref());
    bind_mlx_c(prefix.as_deref());
    link_metallib_beside_test_binaries(prefix.as_deref());
    link_jit_headers_beside_test_binaries(prefix.as_deref());
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

/// mlx-c, bound here (docs/vendoring.md, "mlx-c, bound here").
///
/// Nothing is compiled: the prefix `TOROBI_MLX_PREFIX` names already holds
/// `libmlxc.a` and the headers bindgen reads.
///
/// Refused outright without one. There is no other MLX to fall back to,
/// and a build that went ahead would fail at `include!` with a message
/// about a missing file rather than about the missing prefix.
fn bind_mlx_c(prefix: Option<&Path>) {
    let prefix = prefix.unwrap_or_else(|| {
        panic!(
            "TOROBI_MLX_PREFIX is not set. `rake` and `rake compile` set it to the \
             pre-built MLX they fetch (lib/torobi/mlx_prebuilt.rb); a plain cargo \
             build needs it pointed at an MLX install prefix that holds mlx-c"
        )
    });
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
        link_cpu_linux();
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

/// MLX's CPU backend on Linux calls BLAS and LAPACK (`cblas_sgemm`,
/// `sgetrf_`, `ssyevd_` and the rest), and a static `libmlx.a` leaves them
/// to whoever links it. MLX's own CMake package names OpenBLAS for both
/// (`share/cmake/MLX/MLXTargets.cmake`, which is what the archive was
/// built against), so this names it too: `libopenblas-dev` on Ubuntu.
fn link_cpu_linux() {
    println!("cargo:rustc-link-lib=openblas");
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
/// built against (docs/vendoring.md). The prefix knows: a pre-built
/// archive carries `MANIFEST.txt` beside its prefix, naming the MLX
/// release and the mlx-c commit it was built from. A prefix built by hand
/// may carry none, and then the build says so rather than guessing.
fn record_mlx_version(prefix: Option<&Path>) {
    let manifest = prefix.and_then(|prefix| {
        [prefix.join("MANIFEST.txt"), prefix.parent()?.join("MANIFEST.txt")]
            .into_iter()
            .find(|path| path.is_file())
    });
    let said = manifest.as_deref().and_then(|path| {
        println!("cargo:rerun-if-changed={}", path.display());
        fs::read_to_string(path).ok()
    });
    for (key, var) in [("mlx", "TOROBI_MLX_VERSION"), ("mlx-c", "TOROBI_MLX_C_REVISION")] {
        let value = said
            .as_deref()
            .and_then(|text| {
                text.lines().find_map(|line| {
                    let mut words = line.split_whitespace();
                    (words.next() == Some(key)).then(|| words.next()).flatten()
                })
            })
            .unwrap_or("unstated: the MLX prefix carries no MANIFEST.txt");
        println!("cargo:rustc-env={var}={value}");
    }
}

/// MLX finds its Metal kernels through dladdr, so mlx.metallib has to sit
/// beside whichever binary loaded it: `target/<profile>/torobi-engine`
/// for the command line, and `target/<profile>/deps/` for the test
/// binaries cargo puts there. Without it every one of those aborts with
/// "Failed to load the default metallib" (and it aborts, so nothing can
/// even report it).
///
/// The metallib is in the prefix that is linked, at `lib/mlx.metallib`,
/// which is where the pre-built archive puts it. Only Apple's MLX has
/// one: CUDA compiles its kernels at run time.
///
/// Symlinks rather than copies: the file is 105MB.
fn link_metallib_beside_test_binaries(prefix: Option<&Path>) {
    if env::var("CARGO_CFG_TARGET_VENDOR").as_deref() != Ok("apple") {
        return;
    }
    let (Some(profile_dir), Some(prefix)) = (profile_dir(), prefix) else {
        return;
    };
    let named = prefix.join("lib").join("mlx.metallib");
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

/// Linux's counterpart to the metallib: MLX compiles the CUDA kernels it
/// did not build ahead of time at run time, and reads NVIDIA's headers
/// (`cccl`, `cute`, `cutlass`) from one directory above the object that
/// loads them. The prefix holds them under `include/`; point the two
/// places a binary looks - `target/<profile>/include` for the test
/// binaries, `target/include` for the command line - at that directory.
///
/// A symlink to the whole `include`, not a copy: the tree is tens of
/// megabytes, and MLX reads only the three directories out of it.
fn link_jit_headers_beside_test_binaries(prefix: Option<&Path>) {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }
    let (Some(profile_dir), Some(prefix)) = (profile_dir(), prefix) else {
        return;
    };
    let include = prefix.join("include");
    if !include.join("cccl").is_dir() {
        // The print is the point: without these, the first JIT-compiled
        // kernel fails a long way from here.
        println!("cargo:warning=no cccl headers at {}", include.display());
        return;
    }
    link(&include, &profile_dir.join("include"));
    if let Some(target) = profile_dir.parent() {
        link(&include, &target.join("include"));
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
