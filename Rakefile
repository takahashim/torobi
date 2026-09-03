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

# The engine's own tests, which reach MLX directly rather than through the
# extension. Serial, and not by preference: MLX's default stream is one
# command queue, and two threads submitting to it at once trips a Metal
# assertion ("Completed handler provided after commit call") that aborts
# the process. The extension holds a mutex for the same reason; a cargo
# test binary has nothing holding one, so the harness is told not to.
desc "run the engine's Rust tests"
task :rust_test do
  sh "cargo test -p torobi-engine -- --test-threads=1"
end

Minitest::TestTask.create

task compile: [] # defined by RbSys::ExtensionTask above
task test: %i[compile metallib]
task default: %i[test rust_test]
