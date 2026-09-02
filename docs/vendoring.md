# Vendoring ledger (G0)

The engine's dependency on mlx-rs is not yet vendored. Current state:

| what | state |
|---|---|
| mlx-rs / mlx-sys / mlx-c | path dependency on `/Users/maki/git/OminiX-MLX` at rev `4988a3f` (M1 spike only) |
| MLX core | **not built from source here**: mlx-sys found no Metal compiler and downloaded OminiX's pre-built binary `mlx-prebuilt-v0.1.0-macos-arm64.tar.gz` (from the OminiX-MLX releases page). The exact MLX revision is whatever that tarball pins |
| mlx.metallib | 105 MB. MLX locates it through `dladdr`, i.e. **beside whichever library holds the MLX symbols**: `target/release/` for the CLI, `lib/torobi/` for the extension bundle. `rake metallib` copies it there. Any distribution must ship it beside the bundle |

Before anything ships, replace the path dependency with selective vendoring
(array / ops / fast / transforms / io; prune the rest) and record here: the
origin commit, what was pruned, and the exact MLX / mlx-c revisions.
`torobi-engine --build-info` must report all of them.
