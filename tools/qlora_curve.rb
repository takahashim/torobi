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
# `--ids` is a JSON array of documents, each an array of token ids
# (`tools/tokenize_ids.py` makes one from text). Torobi never tokenizes.
# `--synthetic` draws sequences from a fixed second-order Markov chain
# over a few hundred ids instead: not language, but learnable, and the
# same for both runs, which is what a comparison needs when no text is
# at hand. The last `--holdout` chunks are never trained on.
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "torobi"
require "json"
require "optparse"

abort "the engine is not compiled (rake compile)" unless defined?(Torobi::Session)

options = { steps: 300, seq: 128, batch: 4, eval_every: 25, lr: 1e-4, rank: 8,
            on: %w[q_proj v_proj], bits: 4, group_size: 64, holdout: 8, seed: 42 }
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
chunks = documents.flat_map { |doc| doc.each_slice(seq + 1).select { |c| c.size == seq + 1 } }
abort "no chunk of #{seq + 1} tokens in the data; shorten --seq" if chunks.size <= options[:holdout]
heldout = chunks.last(options[:holdout])
training = chunks[0...-options[:holdout]]
warn "#{training.size} training chunks, #{heldout.size} held out, #{seq} tokens each"

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
heldout_batch = batch_of(config, heldout, seq:)

# --- the two runs ---

adapter = Torobi::LoRA.new(rank: options[:rank], alpha: options[:rank] * 2, on: options[:on])
quantization = Torobi::Quantization.new(bits: options[:bits], group_size: options[:group_size])

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
  run = run_of(config, seq:, adapter:, quantization: base == :quantized ? quantization : nil)
  losses = []
  evals = {}
  started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  Torobi::Session.open(run, pretrained: { m: dir }, fresh: adapter.fresh(run), seed: options[:seed],
                            optimizer: { kind: :adamw, lr: options[:lr] }) do |s|
    evals[0] = s.evaluate(heldout_batch)
    batches.each_with_index do |batch, i|
      losses << s.step!(batch)
      step = i + 1
      if (step % options[:eval_every]).zero? || step == batches.size
        evals[step] =
          s.evaluate(heldout_batch)
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

puts "  step        bf16        4bit   bf16 held   4bit held"
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
