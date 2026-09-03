#!/usr/bin/env ruby
# frozen_string_literal: true

# Does the gem, built and installed, actually run?
#
# This is the M1 exit criterion the review asked to bring forward
# (docs/plan.md section 11.4): build the gem, install it into an empty
# GEM_HOME, require it from a directory that is not this checkout, and take
# one step. It answers the questions the vendoring ledger raises - is the
# metallib in the package, does it land where dladdr looks, does the
# extension build outside the workspace - by trying, not by reasoning.
#
# Slow (it compiles MLX afresh), so it is a script rather than part of the
# suite: `ruby test/installed_gem_smoke.rb`.

require "fileutils"
require "tmpdir"
require "json"

ROOT = File.expand_path("..", __dir__)
abort "run from a checkout" unless File.exist?(File.join(ROOT, "torobi.gemspec"))

def run(command, env: {}, chdir: ROOT)
  puts "  $ #{command}"
  ok = system(env, command, chdir: chdir)
  abort "  failed: #{command}" unless ok
end

Dir.mktmpdir("torobi-smoke") do |home|
  puts "1. building the gem"
  FileUtils.rm_f(Dir[File.join(ROOT, "torobi-*.gem")])
  run("gem build torobi.gemspec --quiet")
  gem = Dir[File.join(ROOT, "torobi-*.gem")].first or abort "no gem was built"
  puts "  #{File.basename(gem)}, #{(File.size(gem) / 1024.0).round} KB"

  puts "2. installing it into an empty GEM_HOME (this compiles MLX; minutes)"
  env = { "GEM_HOME" => home, "GEM_PATH" => home }
  run("gem install #{gem} --no-document --quiet", env:)

  puts "3. requiring it from outside the checkout, and taking one step"
  script = <<~RUBY
    require "torobi"
    model = Torobi.graph do |g|
      x = g.input :x, [nil, 2]
      y = g.input :y, [nil, 1]
      g.output :loss, g.mse(g.linear(x, 1, name: "l"), y)
    end
    config = Torobi::GraphConfig.new(models: { "m" => model })
    weights = { params: { "m.l.weight" => { shape: [1, 2], data: [0.0, 0.0] },
                          "m.l.bias" => { shape: [1], data: [0.0] } } }
    batch = { x: { shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0] },
              y: { shape: [2, 1], data: [1.0, 2.0] } }
    loss = Torobi::Session.open(config, weights) { |s| s.adjust(lr: 0.1).step!(batch) }
    puts JSON.generate({ loss:, metallib: File.exist?(Torobi::Preflight::METALLIB),
                         engine: Torobi::Native.build_info })
  RUBY
  Dir.mktmpdir("torobi-elsewhere") do |elsewhere|
    File.write(File.join(elsewhere, "smoke.rb"), script)
    run("ruby smoke.rb", env:, chdir: elsewhere)
  end
end

puts "installed-gem smoke: OK"
