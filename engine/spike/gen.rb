#!/usr/bin/env ruby
# frozen_string_literal: true

# M1 spike, half one: build the linear-regression GraphConfig with the DSL
# and write deterministic synthetic data beside it. The engine and the
# verifier both read exactly these files.

require_relative "../../lib/torobi"
require "json"

N, D_IN, D_OUT = 64, 4, 2
rng = Random.new(42)

model = Torobi.graph do |g|
  x = g.input :x, [nil, D_IN]
  y = g.input :y, [nil, D_OUT]
  g.output :loss, g.mse(g.linear(x, D_OUT, name: "linear"), y)
end
config = Torobi::GraphConfig.new(models: { "spike" => model },
                                 metadata: { "purpose" => "M1 linear-regression spike" })

w_true = Array.new(D_OUT) { Array.new(D_IN) { rng.rand(-1.0..1.0) } }
b_true = Array.new(D_OUT) { rng.rand(-0.5..0.5) }
x = Array.new(N) { Array.new(D_IN) { rng.rand(-1.0..1.0) } }
y = x.map do |row|
  D_OUT.times.map do |o|
    D_IN.times.sum { |i| row[i] * w_true[o][i] } + b_true[o] + 0.01 * rng.rand(-1.0..1.0)
  end
end
w0 = Array.new(D_OUT) { Array.new(D_IN) { rng.rand(-0.1..0.1) } }
b0 = Array.new(D_OUT, 0.0)

dir = __dir__
File.write(File.join(dir, "graph.json"), config.canonical_json)
File.write(File.join(dir, "bindings.json"), JSON.generate(
  inputs: { x: { shape: [N, D_IN], data: x.flatten },
            y: { shape: [N, D_OUT], data: y.flatten } },
  params: { "spike.linear.weight" => { shape: [D_OUT, D_IN], data: w0.flatten },
            "spike.linear.bias" => { shape: [D_OUT], data: b0 } }
))
puts "graph digest=#{config.digest}"
