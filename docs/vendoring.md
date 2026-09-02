# Vendoring ledger (G0)

The engine's dependency on mlx-rs is not yet vendored. Current state:

| what | state |
|---|---|
| mlx-rs / mlx-sys / mlx-c | path dependency on `/Users/maki/git/OminiX-MLX` at rev `4988a3f` (M1 spike only) |
| MLX core | whatever that checkout's mlx-c build pins |

Before anything ships, replace the path dependency with selective vendoring
(array / ops / fast / transforms / io; prune the rest) and record here: the
origin commit, what was pruned, and the exact MLX / mlx-c revisions.
`torobi-engine --build-info` must report all of them.
