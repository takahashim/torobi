# Dependency ledger

What the engine is built against, and how that was decided.

| what | state |
|---|---|
| mlx-rs / mlx-sys / mlx-c | **git dependency on `https://github.com/OminiX-ai/OminiX-MLX.git`, pinned to `4988a3fcfa48b8cb5d0780a501b92c6a41401523`.** Cargo.lock records the same commit; cargo resolves the mlx-c submodule itself |
| MLX core | **not built from source here**: mlx-sys finds no Metal compiler on this machine and downloads OminiX's pre-built binary `mlx-prebuilt-v0.1.0-macos-arm64.tar.gz`. The exact MLX revision is whatever that tarball pins, which is the one thing this ledger still cannot name |
| mlx.metallib | 105 MB. MLX locates it through `dladdr`, i.e. **beside whichever library holds the MLX symbols**: `target/release/` for the CLI, the install directory for the extension. `ext/torobi/extconf.rb` appends a Makefile rule that installs it beside the bundle; `rake metallib` does the same for a checkout. Any distribution must ship it beside the bundle |

## Why a git dependency rather than a vendored copy

The plan (docs/plan.md section 5) first imagined selective vendoring:
copying array / ops / fast / transforms / io into this tree and pruning the
rest. A pinned git dependency does the same job for less:

- it costs no source tree of ours to carry or to re-sync
- cargo records the exact commit in Cargo.lock, so the pin is enforced
  rather than described
- it builds anywhere, which is the whole point (below)

What it gives up is local pruning and local patching. If a patch becomes
necessary, the answer is a fork with its own pin, not a vendored copy.

The fork rather than crates.io's mlx-rs 0.25: it tracks MLX 0.32 and falls
back to a pre-built MLX when no Metal compiler is present, which is what
lets the gem build on a machine without Xcode.

## How OminiX builds MLX, and how to leave

`mlx-sys/build.rs` decides in three ways:

| condition | what it does |
|---|---|
| `MLX_PREBUILT_PATH` is set | links `libmlx.a` / `libmlxc.a` / `mlx.metallib` from that directory |
| `xcrun -sdk macosx metal --version` works | builds MLX from source through cmake, patching `device.cpp` / `device.h` to disable NAX for older Metal |
| no Metal compiler (this machine) | downloads `mlx-prebuilt-v0.1.0-macos-arm64.tar.gz` from OminiX's releases |

bindgen runs in every mode, from the mlx-c headers in the submodule.

Two things follow.

**The unknown MLX revision is findable.** The build script says the
pre-built form "is what ships in the official `mlx-metal` wheel, which is
the only way to obtain a build of MLX's Metal kernels without the Metal
compiler". So the tarball is a repackaged wheel, and the version that
wheel names is the MLX revision this ledger cannot currently state.

**There is an exit, and it is narrow.** What Torobi depends on OminiX for
is not the Rust API - the engine touches about twelve functions
(`matmul`, the four arithmetic ops, `square`, `mean`, `transpose_axes`,
`contiguous`, `stop_gradient`, `value_and_grad_with_argnums`, `eval`,
`Array::from_slice`, `Exception`) - but the build machinery: building
without Xcode. That is replaceable:

1. obtain the official `mlx-metal` wheel (pip download, unpack), and
2. point `MLX_PREBUILT_PATH` at it.

Then upstream mlx-rs from crates.io would do, at the cost of matching the
wheel's MLX version to what that release's C API expects, which is exactly
the work OminiX is doing for us. `patch_metal_version` matters only on the
build-from-source path, so it does not follow us out.

When to take the exit: if OminiX stops tracking MLX, if a patch of our own
becomes necessary (fork instead), or if the distribution decision
(docs/plan.md 11.4) lands on a platform gem - because then nothing is built
at install time and the value of the auto-download disappears. The
dependency and the distribution question are two faces of one decision.

## Updating the pin

One version at a time, as its own change: move the rev, rebuild, and run
the differential, convergence, memory and installed-gem tests
(docs/plan.md section 12). `Torobi::Native.build_info` reports what a build
was made from; it should learn to report this rev.

## What the installed-gem smoke test found

`ruby test/installed_gem_smoke.rb` builds the gem, installs it into an
empty GEM_HOME, and runs one step from outside the checkout.

**2026-09-03, with the path dependency**: the mechanics worked (a 56 KB
source gem built the extension, the metallib landed where dladdr looks, a
step ran), but the gem only built on this machine: `engine/Cargo.toml`
named `/Users/maki/git/OminiX-MLX/mlx-rs` by absolute path, and that path
shipped inside the package. Moving the checkout aside and installing again
failed with `make: *** [target/release/libtorobi.dylib] Error 101`.

**2026-09-03, with the git dependency**: the same test **with the local
checkout renamed away** builds, installs and runs a step (loss 2.5). The
package no longer depends on anything outside itself and its pinned
remotes. That is the finding this ledger existed to reach.
