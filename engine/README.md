# torobi-engine

Torobi's execution engine: interprets a GraphConfig on MLX (via mlx-rs).
M1 state: a spike that trains linear regression from the Ruby-built graph.

```
ruby spike/gen.rb                        # graph.json + bindings.json
cargo build --release -p torobi-engine   # the workspace shares ../target
ruby spike/verify.rb                     # closed-form oracle + SGD run
```

The Ruby extension is built separately, through `rake compile` at the
repository root; `cargo build` alone cannot link it (it needs Ruby's
linker flags).

The dependency on mlx-rs is a pinned path dependency for now; see
../docs/vendoring.md.
