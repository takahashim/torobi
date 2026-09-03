#!/usr/bin/env ruby
# frozen_string_literal: true

# What the boundary costs, per batch shape.
#
# docs/plan.md refuses bare ratios: the claim "the boundary is under 0.1%"
# has to be a measurement with a shape and a machine attached. This is that
# measurement. It compares three things per shape:
#
#   marshal   Ruby's cost of serializing one batch (JSON.generate)
#   step      a full step through the boundary (marshal + parse + compute)
#   span      the same steps handed over in one call
#
# The interesting number is the share of a step that is not compute.

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "torobi"
require "benchmark"

DIM = 128
STEPS = 50

def model(dim)
  Torobi.graph do |g|
    x = g.input :x, [nil, dim]
    y = g.input :y, [nil, 1]
    g.output g.mse(g.linear(x, 1, name: "linear"), y)
  end
end

def weights(dim)
  { params: { "linear.weight" => { shape: [1, dim], data: Array.new(dim, 0.0) },
              "linear.bias" => { shape: [1], data: [0.0] } } }
end

def batch(rows, dim, rng)
  { x: { shape: [rows, dim], data: Array.new(rows * dim) { rng.rand(-1.0..1.0) } },
    y: { shape: [rows, 1], data: Array.new(rows) { rng.rand(-1.0..1.0) } } }
end

config = Torobi::GraphConfig.new(models: { "spike" => model(DIM) })
rng = Random.new(1)

# MLX initializes its device and compiles on first use; without this the
# first row measures that instead of the boundary.
Torobi::Session.open(config, weights(DIM)) { |s| 3.times { s.step!(batch(4, DIM, rng)) } }

puts "torobi boundary cost, #{RUBY_PLATFORM}, dim=#{DIM}, #{STEPS} steps each"
puts format("%8s %10s %10s %10s %10s", "rows", "marshal", "step", "span", "overhead")

[1, 8, 64, 512].each do |rows|
  batches = Array.new(STEPS) { batch(rows, DIM, rng) }

  marshal = Benchmark.realtime { batches.each { |b| JSON.generate(b) } } / STEPS

  per_step = Torobi::Session.open(config, weights(DIM)) do |s|
    s.adjust(lr: 0.01)
    s.step!(batches.first)
    Benchmark.realtime { batches.each { |b| s.step!(b) } } / STEPS
  end

  per_span = Torobi::Session.open(config, weights(DIM)) do |s|
    s.adjust(lr: 0.01)
    s.step!(batches.first)
    Benchmark.realtime { s.run(batches) } / STEPS
  end

  # What a span saves is the per-call boundary; what remains in `span` is
  # compute plus the parse the engine does either way.
  overhead = per_step - per_span
  puts format("%8d %9.3fms %9.3fms %9.3fms %9.3fms (%.1f%%)",
              rows, marshal * 1000, per_step * 1000, per_span * 1000,
              overhead * 1000, 100.0 * overhead / per_step)
end
