#!/usr/bin/env ruby
# frozen_string_literal: true

# Does the precompiled platform gem, built and installed, actually load and
# run? The counterpart of `installed_gem_smoke.rb`: that one builds the
# source gem (and compiles MLX); this one builds the platform gem, which
# compiles nothing at install, and checks what a platform gem is for - the
# extension under its Ruby ABI, no metallib written beside the bundle, the
# kernels fetched into the cache and named to MLX - then takes one step.
#
# Slow only because the first step fetches the kernels (35 MB). Run from a
# checkout: `ruby test/platform_gem_smoke.rb`.

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

Dir.mktmpdir("torobi-platform-smoke") do |home|
  puts "1. building the platform gem"
  FileUtils.rm_f(Dir[File.join(ROOT, "torobi-*.gem")])
  run("bundle exec rake platform_gem")
  gem = Dir[File.join(ROOT, "torobi-*-arm64-darwin.gem")].first
  abort "no platform gem was built" unless gem
  puts "  #{File.basename(gem)}, #{(File.size(gem) / 1024.0).round} KB"

  puts "2. installing it into an empty GEM_HOME (this compiles nothing)"
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
    loss = Torobi::Session.open(config, weights: weights) { |s| s.adjust(lr: 0.1).step!(batch) }
    puts JSON.generate({ loss:, beside: File.exist?(Torobi::Preflight::METALLIB),
                         cached: File.file?(Torobi::Metallib.cached_path) })
  RUBY
  Dir.mktmpdir("torobi-elsewhere") do |elsewhere|
    File.write(File.join(elsewhere, "smoke.rb"), script)
    run("ruby smoke.rb", env:, chdir: elsewhere)
  end
end

puts "platform-gem smoke: OK"
