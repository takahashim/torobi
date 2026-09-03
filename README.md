# Torobi(とろ火)

A Ruby training framework for Apple Silicon, built on MLX.

Torobi is for slow cooking: fine-tuning and distilling known model
architectures on a local Mac, declaratively and safely. If you are in a
hurry, rent a CUDA GPU instead; this tool does not compete on speed.

Ruby owns the language. Models and objectives are described once in a
Graph DSL and become an immutable, serializable `GraphConfig`. A Rust
engine (via MLX) owns the execution, and the boundary between them is
about a dozen calls wide.

## Requirements

An Apple Silicon Mac, Ruby 3.2 or newer, and a Rust toolchain. The
extension is built from source at install time and brings MLX with it,
which takes a while the first time.

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

## What it is not

Not an eager tensor library: there is no `Torobi::Array` to multiply, and
no Ruby runs inside a step. Not a tokenizer, not a data loader, and not a
trainer that owns your loop. Not for CUDA or Linux, by choice rather than
by limitation.

## Status

The milestones in [docs/plan.md](docs/plan.md) section 9.1 are met
through M4: a distillation of a published ModernBERT runs to the end and
leaves a record. Decoder architectures, LoRA, quantized ops and variable
length attention are not implemented.

The version is 0.0.1 and the API still moves.

## License

MIT
