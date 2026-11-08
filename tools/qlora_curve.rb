#!/usr/bin/env ruby
# frozen_string_literal: true

# Does a 4-bit base learn the way a bf16 base does? (docs/qlora.md
# section 6, "QLoRA の品質")
#
# The paper's answer is for NF4 with double quantization; MLX's affine
# 4-bit is another format, so the answer has to be measured. This runs
# the same LoRA fine-tune twice from the same checkpoint, once with the
# base in bf16 and once with it quantized as it is read, on the same
# batches in the same order from the same seed, and prints the two loss
# curves side by side with a held-out loss every so often. What decides
# it is the gap: a 4-bit curve that tracks the bf16 one within a few
# percent is a base that learns; one that does not is a format that does
# not, and `--bits 8` or `--group-size 32` is the next thing to try.
#
#   ruby tools/qlora_curve.rb <checkpoint-dir> --ids ids.json [--steps 300] [--seq 128]
#                                              [--batch 4] [--eval-every 25] [--lr 1e-4]
#                                              [--rank 8] [--on q_proj,v_proj]
#                                              [--bits 4] [--group-size 64] [--holdout 8]
#   ruby tools/qlora_curve.rb <checkpoint-dir> --synthetic
#
# `--reference 8` takes the reference run with the base at 8 bits rather
# than in bf16, for a model whose bf16 run does not fit: on Qwen2.5-0.5B
# an 8-bit base tracked a bf16 one to 0.08% of held-out loss over 300
# steps (docs/qlora.md section 9), so it stands in for it.
#
# The held-out chunks are evaluated `--batch` at a time, never all at
# once: one evaluation of 32 chunks held 32 × 128 × 152k scores in f32
# (2.5 GB) and their intermediates on top, which took down a 16 GB Mac
# (2026-11-08). `--max-gib` (default 8) stops the run, between steps,
# once the device's peak passes it, before the next step can go further.
#
# `--ids` is a JSON array of documents, each an array of token ids
# (`tools/tokenize_ids.py` makes one from text). Torobi never tokenizes.
# `--synthetic` draws sequences from a fixed second-order Markov chain
# over a few hundred ids instead: not language, but learnable, and the
# same for both runs, which is what a comparison needs when no text is
# at hand. `--holdout` documents, spread evenly over the corpus, are
# held out whole: the first chunk of each is what the held-out loss is
# over, and none of their text is trained on. (Taking the last chunks
# instead put the whole held-out set in the last article or two.)
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "torobi"
require "json"
require "optparse"

abort "the engine is not compiled (rake compile)" unless defined?(Torobi::Session)

options = { steps: 300, seq: 128, batch: 4, eval_every: 25, lr: 1e-4, rank: 8,
            on: %w[q_proj v_proj], bits: 4, group_size: 64, holdout: 8, seed: 42, max_gib: 8.0,
            reference: "bf16" }
OptionParser.new do |o|
  o.on("--ids PATH") { |v| options[:ids] = v }
  o.on("--synthetic") { options[:synthetic] = true }
  o.on("--steps N", Integer) { |v| options[:steps] = v }
  o.on("--seq N", Integer) { |v| options[:seq] = v }
  o.on("--batch N", Integer) { |v| options[:batch] = v }
  o.on("--eval-every N", Integer) { |v| options[:eval_every] = v }
  o.on("--lr X", Float) { |v| options[:lr] = v }
  o.on("--rank N", Integer) { |v| options[:rank] = v }
  o.on("--on A,B", Array) { |v| options[:on] = v }
  o.on("--bits N", Integer) { |v| options[:bits] = v }
  o.on("--group-size N", Integer) { |v| options[:group_size] = v }
  o.on("--holdout N", Integer) { |v| options[:holdout] = v }
  o.on("--seed N", Integer) { |v| options[:seed] = v }
  o.on("--max-gib X", Float) { |v| options[:max_gib] = v }
  o.on("--reference KIND", %w[bf16 8]) { |v| options[:reference] = v }
end.parse!
USAGE = "usage: qlora_curve.rb <checkpoint-dir> (--ids ids.json | --synthetic) [options]"
dir = ARGV.first or abort USAGE
abort "#{dir} is not a directory" unless File.directory?(dir)
unless options[:ids] || options[:synthetic]
  abort "say where the tokens come from: --ids ids.json or --synthetic"
end

config = Torobi::Models::Llama.from_config_file(File.join(dir, "config.json"))
seq = options[:seq]

# --- the data: chunks of seq + 1 tokens, inputs and the token after each ---

# A second-order Markov chain over `vocab` ids, fixed by its seed, so
# every run of this script sees the same sequences. Each pair of ids
# has three likely successors, which is enough structure to learn and
# little enough that the loss moves within a few hundred steps.
def synthetic_documents(seed:, documents:, length:, vocab: 300, base: 1000)
  rng = Random.new(seed)
  table = Hash.new { |h, key| h[key] = Array.new(3) { base + rng.rand(vocab) } }
  Array.new(documents) do
    doc = [base + rng.rand(vocab), base + rng.rand(vocab)]
    (length - 2).times { doc << table[[doc[-2], doc[-1]]][rng.rand(3)] }
    doc
  end
end

documents =
  if options[:synthetic]
    synthetic_documents(seed: options[:seed], documents: 64, length: seq + 1)
  else
    JSON.parse(File.read(options[:ids]))
  end
chunks_of = ->(doc) { doc.each_slice(seq + 1).select { |c| c.size == seq + 1 } }
long_enough = documents.select { |doc| doc.size >= seq + 1 }
if long_enough.size <= options[:holdout]
  abort "#{long_enough.size} documents have #{seq + 1} tokens, and #{options[:holdout]} are held " \
        "out; shorten --seq, lower --holdout or give more text"
end
stride = long_enough.size / options[:holdout]
held = (0...options[:holdout]).map { |i| i * stride }
heldout = held.map { |d| chunks_of.call(long_enough[d]).first }
trained = long_enough.each_index.reject { |d| held.include?(d) }
training = trained.flat_map { |d| chunks_of.call(long_enough[d]) }
warn "#{training.size} training chunks, #{heldout.size} held out from as many documents, " \
     "#{seq} tokens each"

def batch_of(config, chunks, seq:)
  rows = chunks.map { |c| c[0...seq] }
  targets = chunks.map { |c| c[1..seq] }
  Torobi::Models::Llama.batch(config, rows, seq:).merge(
    targets: Torobi::TensorData.from_a([rows.size, seq], targets.flatten, dtype: :i32),
    kept: Torobi::TensorData.from_a([rows.size, seq], Array.new(rows.size * seq, 1.0))
  )
end

order = Random.new(options[:seed])
batches = Array.new(options[:steps]) do
  batch_of(config, Array.new(options[:batch]) { training[order.rand(training.size)] }, seq:)
end
# The held-out chunks in batches no larger than a training step's, so an
# evaluation never needs more than a step does.
heldout_batches = heldout.each_slice(options[:batch]).map do |slice|
  [slice.size, batch_of(config, slice, seq:)]
end

# The mean loss over every held-out chunk: each batch's mean, weighted by
# how many chunks it holds (every position of a chunk is kept).
def heldout_loss(session, batches)
  total = batches.sum { |rows, _| rows }
  batches.sum { |rows, batch| session.evaluate(batch) * rows } / total
end

# Stops the run once the device has held more than `max_gib`: checked
# between steps, so it cannot stop a step that is already too large, but
# it stops the next one.
def guard!(max_gib, where)
  peak = Torobi::Memory.peak / (1024.0**3)
  return if peak <= max_gib

  abort format("stopped at %s: the device peaked at %.2f GiB, over --max-gib %.1f. " \
               "Shorten --seq or --batch, or raise --max-gib if the machine has room.",
               where, peak, max_gib)
end

# --- the two runs ---

adapter = Torobi::LoRA.new(rank: options[:rank], alpha: options[:rank] * 2, on: options[:on])
quantization = Torobi::Quantization.new(bits: options[:bits], group_size: options[:group_size])
reference = (if options[:reference] == "8"
               Torobi::Quantization.new(bits: 8,
                                        group_size: options[:group_size])
             end)

def run_of(config, seq:, adapter:, quantization:)
  graph = Torobi::Models::Llama.causal_lm(config, seq:, dtype: :bf16, quantization:, adapter:)
  Torobi::GraphConfig.new(
    models: { m: graph },
    objective: Torobi.objective(m: graph) do |g|
      logits = g.cast(g.from_model(:m, :logits), :f32)
      kept = g.from_batch(:kept, [nil, seq])
      at = g.cross_entropy(logits, g.from_batch(:targets, [nil, seq], dtype: :i32)) * kept
      g.output :loss, g.sum(at) / g.sum(kept)
    end
  )
end

curves = {}
%i[bf16 quantized].each do |base|
  run = run_of(config, seq:, adapter:, quantization: base == :quantized ? quantization : reference)
  losses = []
  evals = {}
  started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  Torobi::Session.open(run, pretrained: { m: dir }, fresh: adapter.fresh(run), seed: options[:seed],
                            optimizer: { kind: :adamw, lr: options[:lr] }) do |s|
    evals[0] = heldout_loss(s, heldout_batches)
    guard!(options[:max_gib], "#{base}, before training")
    batches.each_with_index do |batch, i|
      losses << s.step!(batch)
      step = i + 1
      guard!(options[:max_gib], "#{base}, step #{step}")
      if (step % options[:eval_every]).zero? || step == batches.size
        evals[step] = heldout_loss(s, heldout_batches)
      end
      warn format("%-9s step %4d  loss %.4f%s", base, step, losses.last,
                  evals[step] ? format("  held-out %.4f", evals[step]) : "")
    end
  end
  seconds = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
  curves[base] = { losses:, evals:, seconds:, peak_gib: Torobi::Memory.peak / (1024.0**3) }
  Torobi::Memory.reset_peak!
end

# --- side by side ---

first = options[:reference] == "8" ? "8bit" : "bf16"
second = "#{options[:bits]}bit"
puts format("%6s  %10s  %10s  %10s  %10s", "step", first, second, "#{first} held", "#{second} held")
steps = curves[:bf16][:evals].keys
steps.each do |step|
  loss = ->(base) { step.zero? ? Float::NAN : curves[base][:losses][step - 1] }
  puts format("%6d  %10.4f  %10.4f  %10.4f  %10.4f", step, loss.call(:bf16), loss.call(:quantized),
              curves[:bf16][:evals][step], curves[:quantized][:evals][step])
end

tail = [20, options[:steps]].min
last = ->(base) { curves[base][:losses].last(tail).sum / tail }
held = ->(base) { curves[base][:evals].values.last }
summary = {
  steps: options[:steps], seq:, batch: options[:batch], rank: options[:rank], on: options[:on],
  quantization: { bits: options[:bits], group_size: options[:group_size] },
  reference: options[:reference],
  train_loss_last_steps: { bf16: last.call(:bf16).round(4),
                           quantized: last.call(:quantized).round(4) },
  heldout_loss: { bf16: held.call(:bf16).round(4), quantized: held.call(:quantized).round(4) },
  heldout_gap: ((held.call(:quantized) - held.call(:bf16)) / held.call(:bf16)).round(4),
  seconds_per_step: { bf16: (curves[:bf16][:seconds] / options[:steps]).round(3),
                      quantized: (curves[:quantized][:seconds] / options[:steps]).round(3) },
  peak_gib: { bf16: curves[:bf16][:peak_gib].round(3),
              quantized: curves[:quantized][:peak_gib].round(3) }
}
puts JSON.pretty_generate(summary)
