#!/usr/bin/env ruby
# frozen_string_literal: true

# M3b's other exit condition, at the size it is claimed for: the published
# ruri-v3-130m encoder with a fresh head, memorizing a handful of rows the
# teacher scored.
#
# A tiny model overfitting a tiny dataset (test/modern_bert_gradient_test.rb)
# says the loop is wired up. This says the same thing where the numbers are
# 130 million, the encoder came out of a file, and the target came out of
# another model. If the loss falls to nothing here, everything between the
# batch and the optimizer is doing its job.
#
#   ruby bench/overfit.rb <ruri-v3-130m-dir> <train.jsonl> [steps]
#
# Thirty steps is enough and the default. The loss stops falling around
# 6e-5 and the last digit of it wanders, which is the sigmoid's tail
# rather than a limit of the fit: a teacher score of 0.9997 asks for a
# logit of 8, and the run has already reproduced every row to within
# 0.012 by then.
#
# `train.jsonl` is what kohagi's dataset tool writes: `input_ids` and
# `teacher` per line (tools/dataset in the kohagi repo). Torobi never sees
# the text, which is the arrangement docs/plan.md section 15.19 settled on.
#
# It runs itself in a child process with a memory cap, rather than in the
# one you typed in. That is not ceremony: a 130M model measured in-process
# with no cap is what took the machine down (section 15.22), and Runner
# plus `Memory.limit` are what this repo grew in answer. The parent here is
# eight lines, and they are the eight lines a caller should copy.

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "torobi"
require "json"
require "tmpdir"

CAP = 8 * 1024 * 1024 * 1024

def child(run, dir, rows, steps)
  run.cap!
  config = Torobi::Models::ModernBERT.from_config_file(File.join(dir, "config.json"))
  # Long enough for the longest row and no longer: everything past it is
  # padding that every layer still attends over.
  seq = rows.map { |row| row.fetch("input_ids").size }.max
  student = Torobi::Models::ModernBERT.classifier(config, seq:, encoder_prefix: "")
  objective = Torobi.objective(student:) do |g|
    # The teacher reports a sigmoid, so the student is compared where the
    # teacher spoke rather than where its logits happen to sit.
    g.output :loss, g.mse(g.from_model(:student, :logits).sigmoid,
                          g.from_batch(:teacher, [nil, 1]))
  end
  graph = Torobi::GraphConfig.new(models: { student: }, objective:)

  # Batch fields are named once for the whole run, not per model: a model
  # reads the field its graph declared (`input_ids`), and the objective
  # reads the one it asked for (`teacher`).
  batch = Torobi::Models::ModernBERT
          .batch(config, rows.map { |row| row.fetch("input_ids") }, seq:)
          .merge(teacher: Torobi::TensorData.from_a([rows.size, 1],
                                                    rows.map { |row| row.fetch("teacher") }))

  Torobi::Session.open(graph,
                       pretrained: { student: File.join(dir, "model.safetensors") },
                       fresh: ["student.head.*", "student.classifier.*"],
                       optimizer: { kind: :adamw, lr: 2e-5 },
                       io: run.journal) do |s|
    puts format("%d parameters, %d rows, seq %d", s.parameter_paths.size, rows.size, seq)
    puts format("step %3d: loss %.8f", 0, s.evaluate(batch))
    started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    steps.times do
      s.step!(batch)
      break if run.stopping?

      puts format("step %3d: loss %.8f", s.step, s.loss) if (s.step % 10).zero?
    end
    elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
    puts format("%.2fs for %d steps (%.3fs each), peak %.0f MB",
                elapsed, s.step, elapsed / s.step, Torobi::Memory.peak / 1024.0 / 1024)

    # The loss is a number about the run; this is the claim itself. A tap
    # reads the head's output on a forward that takes no step, and the
    # sigmoid of it is what a cross-encoder reports.
    s.tap("classifier", stat: :full)
    s.evaluate(batch)
    puts "row  student  teacher"
    s.tapped.fetch("classifier").to_a.each_with_index do |logit, i|
      puts format("%3d  %7.4f  %7.4f", i, 1 / (1 + Math.exp(-logit)), rows[i].fetch("teacher"))
    end
  end
end

# The child, when the parent has said where to write.
if ENV[Torobi::Runner::DIRECTORY_VARIABLE]
  dir, train, steps = ARGV
  rows = File.readlines(train).reject { |line| line.strip.empty? }.map { |line| JSON.parse(line) }
  Torobi::Runner.child! { |run| child(run, dir, rows, Integer(steps || 30)) }
end

dir, train, steps = ARGV
unless dir && train
  abort "usage: overfit.rb <ruri-v3-130m-dir> <train.jsonl> [steps]"
end

run_dir = File.join(Dir.tmpdir, "torobi-overfit-#{Process.pid}")
runner = Torobi::Runner.new([RbConfig.ruby, __FILE__, dir, train, steps.to_s],
                            dir: run_dir, memory_limit: CAP).start
outcome = runner.wait
puts "journal: #{runner.journal_path}"
abort "the run did not finish: #{outcome.inspect}" unless outcome.finished?
