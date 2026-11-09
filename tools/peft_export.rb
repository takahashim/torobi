#!/usr/bin/env ruby
# frozen_string_literal: true

# Trains an adapter for a few steps, writes it as PEFT reads one, and
# records what Torobi answers through it, for `tools/peft_check.py
# numbers` to hold PEFT's answer to (docs/qlora.md, Phase 3).
#
#   ruby tools/peft_export.rb <checkpoint-dir> <out-dir> [--steps 20] [--rank 8]
#                             [--on q_proj,v_proj] [--lr 1e-3]
#
# Held in f32, so that what is left between the two answers is the order
# of arithmetic and the comparison can be as tight as the one between
# Torobi and transformers (`Parity::PARITY`). The ids are the ones the
# bf16 reference recorded (`test/oracle/qwen2.5-0.5b.forward.json`), so
# no tokenizer is needed here.
#
# Writes `<out-dir>/adapter/` (the adapter) and `<out-dir>/torobi.json`
# (the ids and the scores at the last position of each).
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "torobi"
require "json"
require "optparse"

abort "the engine is not compiled (rake compile)" unless defined?(Torobi::Session)

options = { steps: 20, rank: 8, on: %w[q_proj v_proj], lr: 1e-3, top: 10 }
OptionParser.new do |o|
  o.on("--steps N", Integer) { |v| options[:steps] = v }
  o.on("--rank N", Integer) { |v| options[:rank] = v }
  o.on("--on A,B", Array) { |v| options[:on] = v }
  o.on("--lr X", Float) { |v| options[:lr] = v }
end.parse!
dir, out = ARGV
abort "usage: peft_export.rb <checkpoint-dir> <out-dir> [options]" unless dir && out

config = Torobi::Models::Llama.from_config_file(File.join(dir, "config.json"))
oracle = File.expand_path("../test/oracle/qwen2.5-0.5b.forward.json", __dir__)
cases = JSON.parse(File.read(oracle)).fetch("cases").map { |c| c.fetch("input_ids") }
seq = cases.map(&:size).max
adapter = Torobi::LoRA.new(rank: options[:rank], alpha: options[:rank] * 2, on: options[:on])

graph = Torobi::Models::Llama.causal_lm(config, seq:, adapter:)
run = Torobi::GraphConfig.new(
  models: { m: graph },
  objective: Torobi.objective(m: graph) do |g|
    kept = g.from_batch(:kept, [nil, seq])
    at = g.cross_entropy(g.from_model(:m, :logits),
                         g.from_batch(:targets, [nil, seq], dtype: :i32)) * kept
    g.output :loss, g.sum(at) / g.sum(kept)
  end
)

# The cases themselves are the training data: what is wanted is an
# adapter whose B is not zero, so that reading it wrong would show.
pad = config.pad_token_id
targets = cases.map { |ids| ids[1..] + Array.new(seq - ids.size + 1, pad) }
kept = cases.map { |ids| Array.new(ids.size - 1, 1.0) + Array.new(seq - ids.size + 1, 0.0) }
batch = Torobi::Models::Llama.batch(config, cases, seq:).merge(
  targets: Torobi::TensorData.from_a([cases.size, seq], targets.flatten, dtype: :i32),
  kept: Torobi::TensorData.from_a([cases.size, seq], kept.flatten)
)

recorded = Torobi::Session.open(run, pretrained: { m: dir }, fresh: adapter.fresh(run),
                                     optimizer: { kind: :adamw, lr: options[:lr] }) do |s|
  losses = Array.new(options[:steps]) { s.step!(batch) }
  warn format("loss %.4f -> %.4f over %d steps", losses.first, losses.last, losses.size)
  s.export_adapter!(File.join(out, "adapter"), adapter:, task_type: "CAUSAL_LM")

  # The scores at each case's last real token. The rows are padded on
  # the right and the model only looks back, so what a position sees is
  # the case alone, as PEFT is given it.
  logits = s.forward(Torobi::Models::Llama.batch(config, cases, seq:)).fetch("m.logits").to_a
  cases.each_with_index.map do |ids, row|
    at = ((row * seq) + ids.size - 1) * config.vocab_size
    scores = logits[at, config.vocab_size]
    top = scores.each_with_index.max_by(options[:top]) { |value, _| value }
    { "input_ids" => ids, "top_logits" => top.map { |value, id| { "id" => id, "value" => value } } }
  end
end

File.write(File.join(out, "torobi.json"),
           "#{JSON.pretty_generate({ "source" => File.basename(dir), "precision" => "f32",
                                     "adapter" => Torobi::Export.adapter_config(adapter),
                                     "cases" => recorded })}\n")
warn "wrote #{File.join(out, "adapter")} and #{File.join(out, "torobi.json")}"
