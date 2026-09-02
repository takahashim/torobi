#!/usr/bin/env ruby
# frozen_string_literal: true

# M1 spike, half two: hold the engine against a closed-form oracle. Linear
# regression with mse has exact gradients, so this is a mathematical oracle
# with no framework behind it. Then train 200 SGD steps and require the
# loss to collapse.

require "json"

dir = __dir__
engine = File.expand_path("../target/release/torobi-engine", dir)
graph = File.join(dir, "graph.json")
bindings_path = File.join(dir, "bindings.json")
abort "run gen.rb first" unless File.exist?(bindings_path)
abort "build the engine first (cargo build --release)" unless File.exist?(engine)

b = JSON.parse(File.read(bindings_path))
rows = ->(t) { t["data"].each_slice(t["shape"][1] || 1).to_a }
x = rows.(b["inputs"]["x"])
y = rows.(b["inputs"]["y"])
w = rows.(b["params"]["linear.weight"])
bias = b["params"]["linear.bias"]["data"]
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
check.("weight grad", dw.flatten, engine_out.dig("grads", "linear.weight", "data"))
check.("bias grad", db, engine_out.dig("grads", "linear.bias", "data"))

lines = `#{engine} train #{graph} #{bindings_path} 200 0.5`.lines.map { JSON.parse(_1) }
raise "engine train failed" unless $?.success?
first = lines.first.fetch("loss")
final = lines.last.fetch("final_loss")
puts format("train: loss %.4f -> %.6f over 200 steps", first, final)
raise "loss did not collapse" unless final < first * 0.05

puts "M1 spike: OK"
