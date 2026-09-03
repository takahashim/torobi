#!/usr/bin/env ruby
# frozen_string_literal: true

# The graph and the data `verify.rb` holds the engine to: linear regression
# built with the DSL, and deterministic synthetic rows beside it.
#
# Written every time rather than committed. They were committed once, and
# by the time anyone looked the DSL had started naming nodes while the
# committed graph had not: the check passed, against a shape the DSL no
# longer produces.

require_relative "../../lib/torobi"
require "json"

N = 64
D_IN = 4
D_OUT = 2
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
    D_IN.times.sum { |i| row[i] * w_true[o][i] } + b_true[o] + (0.01 * rng.rand(-1.0..1.0))
  end
end
w0 = Array.new(D_OUT) { Array.new(D_IN) { rng.rand(-0.1..0.1) } }
b0 = Array.new(D_OUT, 0.0)

dir = __dir__
File.write(File.join(dir, "graph.json"), config.canonical_json)
bindings = {
  inputs: { x: { shape: [N, D_IN], data: x.flatten },
            y: { shape: [N, D_OUT], data: y.flatten } },
  params: { "spike.linear.weight" => { shape: [D_OUT, D_IN], data: w0.flatten },
            "spike.linear.bias" => { shape: [D_OUT], data: b0 } }
}
File.write(File.join(dir, "bindings.json"), JSON.generate(bindings))
puts "graph digest=#{config.digest}"
