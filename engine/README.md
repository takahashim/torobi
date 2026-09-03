# torobi-engine

Torobi's execution engine: interprets a GraphConfig on MLX (via mlx-rs).
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

The dependency on mlx-rs is a pinned path dependency for now; see
../docs/vendoring.md.
