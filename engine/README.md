# torobi-engine

Torobi's execution engine: interprets a GraphConfig on MLX, through its
own binding of mlx-c (`src/mlxc/`).
It runs what `Torobi::Models` describes, ModernBERT included, and it is
what the Ruby extension holds.

`torobi-engine`, the binary, is the same engine with a command line on it:
one graph, one batch, gradients or a short training run. Useful when
something aborts, because the trace is the engine's with no Ruby in it.

```
rake engine:check    # from the repository root: build it, and hold it to
                     # closed-form gradients (engine/check)
```

The Ruby extension is built separately, through `rake compile` at the
repository root; `cargo build` alone cannot link it (it needs Ruby's
linker flags).

`cargo` needs `TOROBI_MLX_PREFIX` pointed at an MLX install prefix that
holds mlx-c; `rake` sets it to the pre-built one it fetches. See
../docs/vendoring.md.
