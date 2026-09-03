#!/usr/bin/env ruby
# frozen_string_literal: true

# The engine held to exact arithmetic, through the command line.
#
# Linear regression with mse has closed-form gradients, so what the engine
# says is compared with what the maths says rather than with another
# implementation: no framework behind the answer, and nothing recorded that
# could go stale (`test/oracle/` is the other kind, recorded from kohagi).
# Then 200 SGD steps, and the loss has to collapse.
#
# Through `torobi-engine`, the command-line face of the engine, which is
# the only way to run it with no Ruby in the picture. That is worth having
# when MLX aborts: the trace is the engine's alone. It is also the only
# thing that exercises that binary, so this check is what keeps it honest.
#
#   rake engine:check

require "json"

dir = __dir__
# The workspace shares one target directory (see the root Cargo.toml).
# Debug by default: the tests build it anyway, and this is about what the
# engine answers rather than how fast.
engine = %w[debug release]
         .map { |profile| File.expand_path("../../target/#{profile}/torobi-engine", dir) }
         .find { |path| File.exist?(path) }
graph = File.join(dir, "graph.json")
bindings_path = File.join(dir, "bindings.json")
abort "run generate.rb first" unless File.exist?(bindings_path)
abort "build the engine first (cargo build -p torobi-engine)" unless engine

b = JSON.parse(File.read(bindings_path))
rows = ->(t) { t["data"].each_slice(t["shape"][1] || 1).to_a }
x = rows.(b["inputs"]["x"])
y = rows.(b["inputs"]["y"])
w = rows.(b["params"]["spike.linear.weight"])
bias = b["params"]["spike.linear.bias"]["data"]
n, d_out = y.size, y.first.size

pred = x.map { |row| w.map { |wo| row.zip(wo).sum { |a, c| a * c } + bias[w.index(wo)] } }
diff = pred.zip(y).map { |p, t| p.zip(t).map { |a, c| a - c } }
loss = diff.flatten.sum { |d| d * d } / (n * d_out)
dpred = diff.map { |row| row.map { |d| 2.0 * d / (n * d_out) } }
dw = Array.new(d_out) do |o|
  Array.new(x.first.size) { |i| n.times.sum { |r| dpred[r][o] * x[r][i] } }
end
db = Array.new(d_out) { |o| n.times.sum { |r| dpred[r][o] } }

engine_out = JSON.parse(`#{engine} grad #{graph} #{bindings_path}`)
raise "engine grad failed" unless $?.success?

check = lambda do |name, want, got|
  worst = want.zip(got).map { |a, c| (a - c).abs }.max
  puts format("%-14s max|Δ| = %.2e", name, worst)
  raise "#{name}: mismatch (#{worst})" if worst > 1e-4
end
check.("loss", [loss], [engine_out["loss"]])
check.("weight grad", dw.flatten, engine_out.dig("grads", "spike.linear.weight", "data"))
check.("bias grad", db, engine_out.dig("grads", "spike.linear.bias", "data"))

lines = `#{engine} train #{graph} #{bindings_path} 200 0.5`.lines.map { JSON.parse(_1) }
raise "engine train failed" unless $?.success?
first = lines.first.fetch("loss")
final = lines.last.fetch("final_loss")
puts format("train: loss %.4f -> %.6f over 200 steps", first, final)
raise "loss did not collapse" unless final < first * 0.05

puts "engine check: OK"
