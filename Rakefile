# frozen_string_literal: true

require "minitest/test_task"
require "rb_sys/extensiontask"

GEMSPEC = Gem::Specification.load("torobi.gemspec")

RbSys::ExtensionTask.new("torobi", GEMSPEC) do |ext|
  ext.lib_dir = "lib/torobi"
  ext.ext_dir = "ext/torobi"
end

# MLX finds its Metal kernels through dladdr: the metallib must sit beside
# the loaded library, which for us is the extension bundle, not the engine
# binary. mlx-sys drops it in the cargo target directory; put a copy where
# the bundle can find it. See docs/vendoring.md.
task :metallib do
  source = Dir["target/*/mlx.metallib"].max_by { |path| File.mtime(path) }
  raise "no mlx.metallib in target/; run rake compile first" unless source

  destination = "lib/torobi/mlx.metallib"
  next if File.exist?(destination) && File.mtime(destination) >= File.mtime(source)

  require "fileutils"
  FileUtils.cp(source, destination)
  puts "copied #{source} -> #{destination}"
end

# The engine's own tests. Serial, and not by preference: MLX's default
# stream is one command queue, and two threads submitting to it at once
# trips a Metal assertion that aborts the process.
#
# The engine's runtime serializes everything that reaches MLX *through a
# Session*, but most of these tests are of the layers below that (plan,
# state, optimizer, tensor) and build arrays by hand. Pushing the gate down
# into every layer to satisfy a test harness would be the wrong shape: the
# gate belongs at the facade. So the harness is told to go one at a time,
# and `rust_test:facade` below is what shows the facade actually works.
desc "run the engine's Rust tests"
task :rust_test do
  sh "cargo test -p torobi-engine -- --test-threads=1"
end

namespace :rust_test do
  # The proof that the runtime is where the constraint is. Every test that
  # goes through the public Session runs at the harness's default
  # parallelism; before the runtime moved into the engine, this crashed.
  desc "run the engine's facade tests in parallel, which is the point of the runtime"
  task :facade do
    sh "cargo test -p torobi-engine --lib session::"
  end
end

Minitest::TestTask.create

task compile: [] # defined by RbSys::ExtensionTask above
task test: %i[compile metallib]
task default: %i[test rust_test rust_test:facade]
