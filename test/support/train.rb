#!/usr/bin/env ruby
# frozen_string_literal: true

# A training run as a runner starts it: no arguments, no pipe, a directory
# in the environment. Used by RunnerTest, and small enough to read as the
# example it also is.
$LOAD_PATH.unshift(File.expand_path("../../lib", __dir__))
require "torobi"

DIM = 32
STEPS = Integer(ENV.fetch("STEPS", "40"))
EVERY = Integer(ENV.fetch("CHECKPOINT_EVERY", "10"))

Torobi::Runner.child! do |run|
  raise ENV["RAISE"] if ENV["RAISE"]

  model = Torobi.graph do |g|
    x = g.input :x, [nil, DIM]
    g.output :loss, g.mean(g.linear(x, DIM, name: "l"))
  end
  config = Torobi::GraphConfig.new(models: { "m" => model })
  weights = { params: { "m.l.weight" => { shape: [DIM, DIM], data: Array.new(DIM * DIM, 0.01) },
                        "m.l.bias" => { shape: [DIM], data: Array.new(DIM, 0.0) } } }
  batch = { x: { shape: [64, DIM], data: Array.new(64 * DIM, 1.0) } }

  Torobi::Session.open(config, weights: weights, io: run.journal) do |s|
    s.restore(run.checkpoint) if Torobi::Checkpoint.exist?(run.checkpoint)
    s.run(Array.new(STEPS) { batch }) do
      s.checkpoint!(run.checkpoint, at: { step: s.step }) if (s.step % EVERY).zero?
      # Stands in for MLX ending the process: no rescue sees this, which is
      # the whole reason a long run lives in a process of its own.
      Process.kill("ABRT", Process.pid) if ENV["ABORT_AT"] && s.step == Integer(ENV["ABORT_AT"])
      break if run.stopping?
    end
    # However it ended, the state it reached is on disk.
    s.checkpoint!(run.checkpoint, at: { step: s.step })
  end
end
