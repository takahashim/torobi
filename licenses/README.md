# Third-party notices

Torobi is MIT (see [`../LICENSE`](../LICENSE)). The engine links MLX's
compiled code, so a package that carries `torobi.bundle` carries these
notices with it; the binding also follows `mlx-rs`'s API and takes a few
small pieces from it, so its notice is here too.

| file | what | licence | holder |
|---|---|---|---|
| `MLX.txt` | MLX, the library itself | MIT | Apple Inc. (ml-explore) |
| `mlx-c.txt` | mlx-c, the C API the engine binds | MIT | ml-explore |
| `gguflib.txt` | gguflib, which MLX vendors | MIT | Georgi Gerganov |
| `mlx-rs.txt` | the binding's reference, not a dependency | MIT | Minghua Wu, David Chavez |

`docs/vendoring.md` is the long form: where each came from, and what a
redistributor owes.
