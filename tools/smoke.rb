#!/usr/bin/env ruby
# frozen_string_literal: true

# What an installed gem has to be able to do: be required somewhere that is
# not this checkout, and take a step (docs/plan.md M1, exit condition 5).
#
# It is a small program on purpose. What is being tested is the packaging,
# not the library: that `spec.files` names everything `require "torobi"`
# reaches for, that the extension builds from what ships rather than from
# what happens to be lying in `target/`, and that the 105 MB metallib
# lands beside the bundle (MLX finds it by `dladdr`, so beside is the only
# place that works, and getting it wrong ends the process rather than
# raising).
#
# `rake smoke` builds the gem, installs it into a directory of its own and
# runs this there. Running it by hand needs the same: a GEM_HOME with
# torobi installed, and no bundler putting this checkout back on the path.

require "torobi"

loaded = $LOADED_FEATURES.grep(/torobi\.rb$/).first
if loaded.start_with?(File.expand_path("..", __dir__))
  abort "this loaded the checkout at #{loaded}, so it is not testing an install"
end
puts "torobi #{Torobi::VERSION} from #{loaded}"
puts Torobi::Native.build_info.inspect

metallib = File.expand_path("../mlx.metallib", loaded.sub(/\.rb$/, "/torobi.bundle"))
abort "no mlx.metallib beside the extension" unless File.exist?(metallib)

graph = Torobi.graph do |g|
  x = g.input :x, [nil, 2]
  y = g.input :y, [nil, 1]
  g.output :loss, g.mse(g.linear(x, 1, name: "l"), y)
end
config = Torobi::GraphConfig.new(models: { m: graph })
weights = { params: { "m.l.weight" => { shape: [1, 2], data: [0.0, 0.0] },
                      "m.l.bias" => { shape: [1], data: [0.0] } } }
batch = { x: { shape: [4, 2], data: [1.0, 2.0, 2.0, 1.0, 3.0, 0.5, 0.5, 3.0] },
          y: { shape: [4, 1], data: [3.0, 3.0, 3.5, 3.5] } }

Torobi::Session.open(config, weights:, optimizer: { kind: :sgd, lr: 0.05 }) do |s|
  before = s.evaluate(batch)
  s.step!(batch)
  after = s.evaluate(batch)
  puts format("one step: %.6f -> %.6f", before, after)
  abort "the step did not move the loss" unless after < before
end
puts "smoke: OK"
