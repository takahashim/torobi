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
- **`Session#forward`**, which asks a model what it produces rather than
  what its loss is: named outputs as `TensorData`, from the same
  deterministic pass `evaluate` runs, stopping before the objective. A
  config that trains nothing (`train: []`) needs no loss at all, which
  is what a run opened for inference looks like. Serving is still out of
  scope: no HTTP, no tokenizer, no KV cache.
- **`Torobi::Models::Qwen2`**, the first decoder: a causal language
  model as a graph, declaring exactly the 290 parameters Qwen2.5-0.5B
  holds, with the tied output projection that checkpoint expects. No KV
  cache and no sampling loop, which stay out of scope; what it is for is
  fine-tuning, and one forward is `Session#forward`.
- **The vocabulary a decoder is written in**: `max`, a `cross_entropy`
  that takes the largest logit out before it exponentiates, attention
  with fewer key heads than query heads (grouped-query attention) and a
  `causal:` mode instead of a handed-over triangle, and `g.parameter` for
  a weight read twice (a tied embedding). Attention now runs through
  MLX's fused kernel, which is what makes the first two of those
  possible; it differentiates, and there is a test that it does.
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

- Qwen2 is described and differentiates, but has not been held to a
  reference implementation's numbers; that needs the weights and a
  second implementation to run them. Other decoder architectures, LoRA,
  quantized ops and variable length attention are not implemented
  (docs/plan.md section 9.1, M6).
- `Torobi::Memory.limit=` does not refuse an allocation that exceeds it;
  a run that must stay under a number wants `Policies::MemoryGuard`.
- The API still moves. Nothing here is a compatibility promise yet.
