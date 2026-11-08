#!/usr/bin/env ruby
# frozen_string_literal: true

# Measures what a quantized base takes on this machine (docs/qlora.md
# section 1.2 and Phase 2), on a model worth measuring.
#
#   ruby tools/qlora_memory.rb <checkpoint-dir> [--seq 512] [--bits 4] [--group-size 64]
#                                               [--rank 8] [--on q_proj,v_proj] [--bf16]
#                                               [--open-only]
#
# The directory is a published checkpoint as the Hub caches it: its
# config.json, and model.safetensors or the shards with their index.
# Opens it as `causal_lm(dtype: :bf16, quantization:, adapter:)`, which
# quantizes the linears as they are read (bf16 file) or reads them packed
# (a `*-4bit` file), takes one step of batch 1 over `--seq` tokens, and
# prints what the device held at each point:
#
#   declared   what the graph's parameters take, from their shapes alone
#   opened     active and peak after Session.open: the base in memory
#   stepped    active and peak after one step: the activations on top
#
# `--bf16` leaves the base in bf16, for the number the table compares
# against; it needs the memory a bf16 model needs. `--open-only` stops
# after `opened`, for a machine where the step might not fit: the peak at
# open is the question Phase 2 asks, and it is answered before any step.
#
# What it prints is a line of JSON per point, so numbers from two
# machines can be put side by side.
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "torobi"
require "json"
require "optparse"

abort "the engine is not compiled (rake compile)" unless defined?(Torobi::Session)

options = { seq: 512, bits: 4, group_size: 64, rank: 8, on: %w[q_proj v_proj], bf16: false, open_only: false }
OptionParser.new do |o|
  o.on("--seq N", Integer) { |v| options[:seq] = v }
  o.on("--bits N", Integer) { |v| options[:bits] = v }
  o.on("--group-size N", Integer) { |v| options[:group_size] = v }
  o.on("--rank N", Integer) { |v| options[:rank] = v }
  o.on("--on A,B", Array) { |v| options[:on] = v }
  o.on("--bf16") { options[:bf16] = true }
  o.on("--open-only") { options[:open_only] = true }
end.parse!
dir = ARGV.first or abort "usage: qlora_memory.rb <checkpoint-dir> [options]"
abort "#{dir} is not a directory" unless File.directory?(dir)

config = Torobi::Models::Llama.from_config_file(File.join(dir, "config.json"))
raw = JSON.parse(File.read(File.join(dir, "config.json")))
quantization =
  if options[:bf16]
    nil
  else
    # A file already quantized says how; a bf16 file is quantized as asked.
    Torobi::Quantization.from_config(raw) ||
      Torobi::Quantization.new(bits: options[:bits], group_size: options[:group_size])
  end
adapter = Torobi::LoRA.new(rank: options[:rank], alpha: options[:rank] * 2, on: options[:on])
seq = options[:seq]

graph = Torobi::Models::Llama.causal_lm(config, seq:, dtype: :bf16, quantization:, adapter:)
run = Torobi::GraphConfig.new(
  models: { m: graph },
  objective: Torobi.objective(m: graph) do |g|
    logits = g.cast(g.from_model(:m, :logits), :f32)
    at = g.cross_entropy(logits, g.from_batch(:targets, [nil, seq], dtype: :i32))
    g.output :loss, g.mean(at)
  end
)

BYTES = { f32: 4, bf16: 2, u32: 4, i32: 4, bool: 1 }.freeze
declared = graph.parameters.sum { |spec| spec.shape.reduce(1, :*) * BYTES.fetch(spec.dtype) }
gib = ->(bytes) { (bytes / (1024.0**3)).round(3) }
say = lambda do |point, extra = {}|
  report = Torobi::Memory.report
  puts JSON.generate({ point:, active_gib: gib.call(report[:active]),
                       peak_gib: gib.call(report[:peak]) }.merge(extra))
end

puts JSON.generate(
  point: "declared", model: raw["_name_or_path"] || File.basename(dir),
  parameters: graph.parameters.sum { |spec| spec.shape.reduce(1, :*) },
  quantization: quantization&.to_h&.except(:on), adapter: { rank: adapter.rank, on: adapter.on },
  seq:, declared_gib: gib.call(declared)
)

Torobi::Memory.reset_peak!
started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
Torobi::Session.open(run, pretrained: { m: dir }, fresh: adapter.fresh(run),
                          optimizer: { kind: :adamw, lr: 1e-4 }) do |s|
  say.call("opened", seconds: (Process.clock_gettime(Process::CLOCK_MONOTONIC) - started).round(1))
  next if options[:open_only]

  rng = Random.new(1)
  ids = Array.new(seq) { rng.rand(1000...(config.vocab_size - 1000)) }
  batch = Torobi::Models::Llama.batch(config, [ids], seq:).merge(
    targets: Torobi::TensorData.from_a([1, seq], ids[1..] + [config.pad_token_id], dtype: :i32)
  )
  Torobi::Memory.reset_peak!
  started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  loss = s.step!(batch)
  say.call("stepped", loss: loss.round(4),
seconds: (Process.clock_gettime(Process::CLOCK_MONOTONIC) - started).round(1))
  loss = s.step!(batch)
  say.call("stepped again", loss: loss.round(4))
end
