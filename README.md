# Torobi(とろ火)

A Ruby training framework for Apple Silicon, built on MLX.

Torobi is for slow cooking: fine-tuning and distilling known model
architectures on a local Mac, declaratively and safely. If you are in a
hurry, rent a CUDA GPU instead; this tool does not compete on speed.

Ruby owns the language: models and objectives are described once in a
Graph DSL and become an immutable, serializable `GraphConfig`. A Rust
engine (via MLX) owns the execution. See [docs/plan.md](docs/plan.md)
for the design.

Status: early development (M0). Nothing here is usable yet.

## License

MIT
