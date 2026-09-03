# Changelog

What changed, for somebody deciding whether to upgrade. The reasoning
behind each change is in [docs/plan.md](docs/plan.md) section 15, which
is the long form and stays that way; this is the short one.

Nothing has been released yet. Until 0.0.1 is pushed, everything below
is what the first release will hold rather than a record of what moved
between two versions people have.

## Unreleased

### Added

- **A Graph DSL and an immutable `GraphConfig`.** Models and objectives
  are described once in plain Ruby, at build time, and become a frozen
  IR with a digest. Shape and dtype mistakes are reported where the
  graph is written rather than on the first step.
- **`Torobi::Session`**, the whole of the run: `step!`, `run`,
  `evaluate`, `fetch`, `tap`, `freeze`/`thaw`, `accumulate`/`apply!`,
  `field_gradients`, checkpoints and resume. A run is opened with
  `weights:` or `pretrained:` and closed, and what it reads and holds is
  reported rather than assumed.
- **An MLX engine in Rust**, which owns execution. The boundary is about
  a dozen calls wide, releases the GVL, and serializes everything that
  reaches MLX through one runtime.
- **`Torobi::Models::ModernBERT`**: the published architecture as a
  graph, built from a checkpoint's own `config.json`, with the
  checkpoint's own parameter names so a model imports with no renaming.
  `classifier` for a cross-encoder, `embedder` for a sentence embedder,
  and `batch` for the ids, masks and pooling weights a step needs.
- **`Torobi::GradCache`**, which trains a contrastive batch larger than
  the machine can hold, landing where the whole batch would have.
- **`Torobi::Export`**: a run's weights written as a HuggingFace /
  sentence-transformers checkpoint, carrying the source model's
  tokenizer and configuration so the result loads as the model it is.
- **The window**: a journal of what a run did, hooks and knobs while it
  runs, two kinds of replay, and `Torobi::Runner` for a long run in a
  process of its own with a memory cap.
- **`rake mlx:pin`**, and a prebuilt MLX fetched by digest rather than
  by URL alone.

### Known limitations

- Decoder architectures, LoRA, quantized ops and variable length
  attention are not implemented (docs/plan.md section 9.1, M6).
- `Torobi::Memory.limit=` does not refuse an allocation that exceeds it;
  a run that must stay under a number wants `Policies::MemoryGuard`.
- The API still moves. Nothing here is a compatibility promise yet.
