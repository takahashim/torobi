# torobi-engine

Torobi's execution engine: interprets a GraphConfig on MLX (via mlx-rs).
M1 state: a spike that trains linear regression from the Ruby-built graph.

```
ruby spike/gen.rb                                  # graph.json + bindings.json
cargo build --release
ruby spike/verify.rb                               # closed-form oracle + SGD run
```

The dependency on mlx-rs is a pinned path dependency for now; see
../docs/vendoring.md.
