# frozen_string_literal: true

require "minitest/test_task"
require "shellwords"
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

# The versioned artifacts a parity test is held to (docs/plan.md section
# 12). Committed, so the comparison runs on a machine that has never
# downloaded the model; regenerated deliberately, and failing closed when
# the checkpoint is not there rather than quietly writing an empty one.
namespace :oracle do
  RURI_V3_130M = File.expand_path(
    "~/.cache/huggingface/hub/models--cl-nagoya--ruri-v3-130m/snapshots"
  )

  # kohagi is the second implementation Torobi's forward is held to
  # (docs/plan.md 9.2): candle on the CPU, the same published weights, an
  # encoder whose output people already rely on. Its `tools/reference`
  # emitter reads kohagi's public API and changes nothing about it.
  KOHAGI = File.expand_path("../kohagi", __dir__)

  desc "record kohagi's forward pass on ruri-v3-130m, for test/oracle"
  task :forward do
    dir = ENV["RURI_V3_130M"] || Dir[File.join(RURI_V3_130M, "*")].max_by { File.mtime(_1) }
    raise "no ruri-v3-130m checkpoint (set RURI_V3_130M)" unless dir && File.directory?(dir)

    manifest = File.join(ENV.fetch("KOHAGI", KOHAGI), "tools/reference/Cargo.toml")
    raise "no kohagi at #{File.dirname(File.dirname(manifest))} (set KOHAGI)" unless File.exist?(manifest)

    long = "駅前の駐輪場が不足しているため、増設を要望します。" * 30
    sh "cargo run --release --manifest-path #{manifest.shellescape} -- " \
       "--model-path #{File.join(dir, "model.safetensors").shellescape} " \
       "--tokenizer-path #{File.join(dir, "tokenizer.json").shellescape} " \
       "--out test/oracle/ruri-v3-130m.forward.json " \
       "#{"瑠璃も玻璃も照らせば光る".shellescape} #{"犬も歩けば棒に当たる".shellescape} #{long.shellescape}"
  end

  desc "record what cl-nagoya/ruri-v3-130m holds, for test/oracle"
  task :ruri do
    dir = ENV["RURI_V3_130M"] || Dir[File.join(RURI_V3_130M, "*")].max_by { File.mtime(_1) }
    raise "no ruri-v3-130m checkpoint (set RURI_V3_130M)" unless dir && File.directory?(dir)

    sh "ruby tools/inventory.rb #{dir.shellescape} test/oracle/ruri-v3-130m.json"
  end
end

desc "regenerate every oracle artifact"
task oracle: ["oracle:ruri", "oracle:forward"]

Minitest::TestTask.create

task compile: [] # defined by RbSys::ExtensionTask above
task test: %i[compile metallib]
task default: %i[test rust_test rust_test:facade]
