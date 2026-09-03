# Vendoring ledger (G0)

The engine's dependency on mlx-rs is not yet vendored. Current state:

| what | state |
|---|---|
| mlx-rs / mlx-sys / mlx-c | path dependency on `/Users/maki/git/OminiX-MLX` at rev `4988a3f` (M1 spike only) |
| MLX core | **not built from source here**: mlx-sys found no Metal compiler and downloaded OminiX's pre-built binary `mlx-prebuilt-v0.1.0-macos-arm64.tar.gz` (from the OminiX-MLX releases page). The exact MLX revision is whatever that tarball pins |
| mlx.metallib | 105 MB. MLX locates it through `dladdr`, i.e. **beside whichever library holds the MLX symbols**: `target/release/` for the CLI, the install directory for the extension. `ext/torobi/extconf.rb` appends a Makefile rule that installs it beside the bundle; `rake metallib` does the same for a checkout. Any distribution must ship it beside the bundle |

Before anything ships, replace the path dependency with selective vendoring
(array / ops / fast / transforms / io; prune the rest) and record here: the
origin commit, what was pruned, and the exact MLX / mlx-c revisions.
`Torobi::Native.build_info` must report all of them (it currently reports
the placeholder above).

## What the installed-gem smoke test found (2026-09-03)

`ruby test/installed_gem_smoke.rb` builds the gem, installs it into an empty
GEM_HOME, and runs one step from outside the checkout. Two findings, one
good and one blocking:

- **The mechanics work.** A 56 KB source gem builds the extension at install
  time, the metallib lands where dladdr looks, and a step runs. Nothing
  about the design of the package is wrong.
- **The gem only builds on this machine.** `engine/Cargo.toml` names
  `/Users/maki/git/OminiX-MLX/mlx-rs` by absolute path, and that path ships
  inside the gem. Moving the checkout aside and installing again fails with
  `make: *** [target/release/libtorobi.dylib] Error 101`, as it must.

So distribution is feasible in shape but impossible in fact until the
vendoring above is done. That is the honest state, and it is why the plan
keeps the distribution decision open rather than claiming a source gem
works.
