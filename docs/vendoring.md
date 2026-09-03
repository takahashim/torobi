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
