# Torobi(とろ火)

A Ruby training framework for Apple Silicon, built on MLX.

Torobi is for slow cooking: fine-tuning and distilling known model architectures on a local Mac, declaratively and safely.

Models and objectives are described once in a Graph DSL and become an immutable, serializable `GraphConfig`.
A Rust engine (via MLX) owns the execution, and the boundary between them is about a dozen calls wide.

## Requirements

- An Apple Silicon Mac
- Ruby 3.2 or newer
- Rust toolchain
- MLX and mlx-c

## Install

```ruby
gem "torobi", github: "takahashim/torobi"
```

## A first run

```ruby
require "torobi"

# What to compute. Plain Ruby, run once, at build time.
model = Torobi.graph do |g|
  x = g.input :x, [nil, 2]
  y = g.input :y, [nil, 1]
  g.output :loss, g.mse(g.linear(x, 1, name: "l"), y)
end
config = Torobi::GraphConfig.new(models: { m: model })

# What to start from. A file or a published checkpoint, or values here.
weights = { params: { "m.l.weight" => { shape: [1, 2], data: [0.0, 0.0] },
                      "m.l.bias" => { shape: [1], data: [0.0] } } }

batch = { x: Torobi::TensorData.nested([[1.0, 2.0], [2.0, 1.0]]),
          y: Torobi::TensorData.nested([[3.0], [3.0]]) }

Torobi::Session.open(config, weights:, seed: 1,
                     optimizer: { kind: :adamw, lr: 0.05 }) do |s|
  s.repeat(batch, steps: 100)
  puts s.loss
  puts s.fetch("m.l.weight").to_a.inspect
end
```

## Fine-tuning a published model

The parameter paths are the checkpoint's own, so a published model
imports with no renaming, and what the file does not hold is named
rather than guessed:

```ruby
config = Torobi::Models::ModernBERT.from_config_file("ruri-v3-130m/config.json")
student = Torobi::Models::ModernBERT.classifier(config, seq: 128, encoder_prefix: "")

Torobi::Session.open(
  Torobi::GraphConfig.new(models: { student: }, objective:),
  pretrained: { student: "ruri-v3-130m/model.safetensors" },
  fresh: ["student.head.*", "student.classifier.*"],
  optimizer: { kind: :adamw, lr: 2e-5 }, seed: 1, io: journal
) { |s| s.run(batches) }
```

`experiments/distill_reranker.rb` is that with the rest of an experiment
around it: a teacher's scores, a validation metric, and a record. It is
the run described in [docs/plan.md](docs/plan.md) section 15.28, where a
130M encoder reached nDCG@10 0.7680 against its 310M teacher's 0.7770.

## Training a sentence embedder

An embedder is the encoder, a mean over the tokens a row actually has,
and the normalization that makes a dot product a cosine. The contrastive
loss is not in the model: it reads across the batch, which makes it the
recipe's. `GradCache` is how the batch gets larger than the machine,
encoding the parts one at a time and still taking the loss over all of
them:

```ruby
model = Torobi::Models::ModernBERT.embedder(config, seq: 128, pooling: :mean)
cache = Torobi::GradCache.new(session, loss: over_vectors,
                              tap: "student.embedding", into: :vectors, seed: :seed)
cache.step(parts)   # one optimizer step over every part
```

`test/contrastive_test.rb` is the whole of it on a tiny model, held to
the step the same batch would have taken in one piece.

## Asking a model what it produces

Training asks what the loss is. Everything else asks what the model
says, which is a different question and now a first-class one:

```ruby
Torobi::Session.open(
  Torobi::GraphConfig.new(models: { m: embedder }, train: []),
  pretrained: { m: "runs/001/model.safetensors" }
) do |s|
  s.output_names                             # => ["m.embedding"]
  s.forward(batch)["m.embedding"]            # => TensorData [rows, dim]
  s.forward(batch, outputs: ["m.logits"])    # or just the one
end
```

It is the pass `evaluate` runs, stopping before the objective: no
gradients, no randomness, and nothing about the run moves. The batch
needs what the models read and no more, so a run trained against labels
does not need them to be asked what it thinks. `train: []` is a run
opened to be read, and needs no loss at all.

What this is not is serving. No HTTP, no tokenizer, no continuous
batching, no KV cache, no generation loop: those belong to whatever
serves the model (docs/plan.md section 14).

## Running a long one

A training run belongs in a process of its own with a cap on what it may
hold, because MLX draws from the same memory as the rest of the machine
and a run that asks for more than there is takes the machine with it:

```ruby
runner = Torobi::Runner.new(["ruby", "train.rb"], dir: "runs/001",
                            memory_limit: 8 * 1024**3).start
runner.progress          # => {step: 1400, loss: 0.21, at: "..."}
runner.stop              # asks it to stop at the next step boundary
runner.wait.finished?    # => true
```

## Development

```
bundle exec rake     # rubocop, the Ruby tests, the engine's Rust tests, engine:check
bundle exec rake smoke   # build the gem, install it somewhere clean, take a step there
```

The Ruby tests build the extension, which links a prebuilt MLX fetched
once and checked against the digest in `ext/torobi/mlx_prebuilt.json`
(`rake mlx:pin` moves it). CI runs the same `rake` on a `macos-15`
runner, on the oldest Ruby the gemspec claims and on the newest.

`.rubocop.yml` says which rules are on and why the rest are not. What is
off is the part that would argue about design; what is on is the part a
reader cannot be relied on to catch.

## Status

The milestones in [docs/plan.md](docs/plan.md) section 9.1 are met
through M4: a distillation of a published ModernBERT runs to the end and
leaves a record.

Decoders are started rather than done: `Models::Qwen2` declares exactly
what Qwen2.5-0.5B holds and its gradients agree with its forward, but it
has not been held to a reference implementation's numbers. Generation is
deliberately absent (no KV cache, no sampling loop); what Torobi does
with a decoder is fine-tune it. LoRA, quantized ops and variable length
attention are not implemented.

The version is 0.0.1 and the API still moves.

## License

MIT.

The engine builds against MLX (MIT, ml-explore) through OminiX-MLX's
mlx-rs (MIT or Apache-2.0), which is a different repository from the
mlx-rs on crates.io. The gem ships none of their code: cargo fetches them
at install time from their own remotes. `docs/vendoring.md` names all
three, with their licences and with what would have to be carried if
Torobi were ever distributed as a compiled gem.
