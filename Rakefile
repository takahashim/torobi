# frozen_string_literal: true

require "minitest/test_task"
require "shellwords"
require "rb_sys/extensiontask"

GEMSPEC = Gem::Specification.load("torobi.gemspec")

# The MLX binary every cargo build here links, fetched once and checked
# against a recorded digest (ext/torobi/mlx_prebuilt.rb). `rake compile`
# reaches it through extconf.rb; the engine's own builds reach it here,
# and both share one copy.
require_relative "ext/torobi/mlx_prebuilt"

def with_mlx
  ENV["MLX_PREBUILT_PATH"] = MlxPrebuilt.ensure!
rescue MlxPrebuilt::Refused => e
  abort "torobi: #{e.message}"
end

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
  with_mlx
  sh "cargo test -p torobi-engine -- --test-threads=1"
end

namespace :rust_test do
  # The proof that the runtime is where the constraint is. Every test that
  # goes through the public Session runs at the harness's default
  # parallelism; before the runtime moved into the engine, this crashed.
  desc "run the engine's facade tests in parallel, which is the point of the runtime"
  task :facade do
    with_mlx
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
    raise "no kohagi at #{File.dirname(manifest, 2)} (set KOHAGI)" unless File.exist?(manifest)

    long = "駅前の駐輪場が不足しているため、増設を要望します。" * 30
    sh "cargo run --release --manifest-path #{manifest.shellescape} -- " \
       "--model-path #{File.join(dir, "model.safetensors").shellescape} " \
       "--tokenizer-path #{File.join(dir, "tokenizer.json").shellescape} " \
       "--out test/oracle/ruri-v3-130m.forward.json " \
       "#{"瑠璃も玻璃も照らせば光る".shellescape} #{"犬も歩けば棒に当たる".shellescape} #{long.shellescape}"
  end

  RERANKER_310M = File.expand_path(
    "~/.cache/huggingface/hub/models--cl-nagoya--ruri-v3-reranker-310m/snapshots"
  )

  desc "record what cl-nagoya/ruri-v3-130m holds, for test/oracle"
  task :ruri do
    dir = ENV["RURI_V3_130M"] || Dir[File.join(RURI_V3_130M, "*")].max_by { File.mtime(_1) }
    raise "no ruri-v3-130m checkpoint (set RURI_V3_130M)" unless dir && File.directory?(dir)

    sh "ruby tools/inventory.rb #{dir.shellescape} test/oracle/ruri-v3-130m.json"
  end

  # The one that needs no checkpoint on disk. A safetensors file says
  # what it holds in a header at the front of it, so an inventory of a
  # published model is a config.json and a byte range rather than the
  # gigabytes underneath (tools/inventory.rb).
  desc "record what Qwen/Qwen2.5-0.5B holds, for test/oracle (over the network)"
  task :qwen2 do
    sh RbConfig.ruby, "tools/inventory.rb", "Qwen/Qwen2.5-0.5B",
       "test/oracle/qwen2.5-0.5b.json"
  end

  # The numbers, from the implementation everyone else is held to. Needs
  # the weights (about 1GB) and a Python environment, so it is `uv run`
  # with the dependencies named on the command line rather than anything
  # installed here.
  desc "record what transformers answers for Qwen2.5-0.5B, for test/oracle"
  task :qwen2_forward do
    sh "uv", "run", "--with", "transformers", "--with", "torch",
       "python", "tools/qwen2_reference.py",
       "--out", "test/oracle/qwen2.5-0.5b.forward.json"
  end

  desc "record what cl-nagoya/ruri-v3-reranker-310m holds, for test/oracle"
  task :reranker do
    dir = ENV["RURI_V3_RERANKER_310M"] ||
          Dir[File.join(RERANKER_310M, "*")].max_by { File.mtime(_1) }
    raise "no reranker checkpoint (set RURI_V3_RERANKER_310M)" unless dir && File.directory?(dir)

    sh "ruby tools/inventory.rb #{dir.shellescape} test/oracle/ruri-v3-reranker-310m.json"
  end
end

desc "regenerate every oracle artifact"
task oracle: ["oracle:ruri", "oracle:reranker", "oracle:qwen2", "oracle:forward"]

# Moving the MLX pin to another release of takahashim/mlx-prebuilt.
#
#   rake mlx:pin          # whatever that repository released last
#   rake mlx:pin[v0.4.2]  # a release by name
#
# What it automates is transcription, which is the part a person does
# badly: it asks GitHub for the release, takes the asset and the digest
# GitHub computed, writes ext/torobi/mlx_prebuilt.json, and then fetches
# through the same code an install uses, so the download is verified
# against the digest that was just recorded. If those bytes are not those
# bytes, nothing is installed and the pin is left as it was.
#
# It stops there on purpose. What makes a pin trustworthy is a test run
# against it, and that is `rake`, and then a commit.
namespace :mlx do
  desc "pin MLX to a release of the prebuilt archive (default: the latest)"
  task :pin, [:tag] do |_task, args|
    require_relative "ext/torobi/mlx_prebuilt"
    require "json"
    require "net/http"

    pin = MlxPrebuilt::PIN.dup
    tag = args[:tag]
    api = "https://api.github.com/repos/#{pin.fetch("repo")}/releases/" \
          "#{tag ? "tags/#{tag}" : "latest"}"
    release = JSON.parse(fetch_json(api))
    asset = release.fetch("assets").find { |a| a.fetch("name").end_with?(".tar.gz") }
    abort "#{release["tag_name"]} has no .tar.gz asset" unless asset
    digest = asset["digest"].to_s.delete_prefix("sha256:")
    abort "#{asset["name"]} carries no sha256 digest" if digest.empty?

    was = pin.slice("release", "asset", "digest")
    pin["release"] = release.fetch("tag_name")
    pin["asset"] = asset.fetch("name")
    pin["digest"] = digest
    write_pin(pin)

    # Through the same path an install takes, so the digest is checked
    # against the bytes rather than believed.
    MlxPrebuilt.ensure!(io: $stdout)
    said = MlxPrebuilt.manifest
    if said
      pin["mlx"] = said.fetch("mlx", pin["mlx"]).delete_prefix("v")
      pin["mlx_c"] = said.fetch("mlx-c", pin["mlx_c"]).delete_prefix("v")
      write_pin(pin)
    else
      warn "torobi: the archive has no MANIFEST.txt, so the versions are unchanged"
    end

    puts was == pin.slice("release", "asset", "digest") ? "unchanged" : "pinned:"
    %w[release asset digest mlx mlx_c].each { |key| puts "  #{key}: #{pin[key]}" }
    puts "now run `rake`, and commit ext/torobi/mlx_prebuilt.json if it passes"
  end
end

def write_pin(pin)
  File.write("ext/torobi/mlx_prebuilt.json", "#{JSON.pretty_generate(pin)}\n")
end

# One GitHub API read, following its redirects and saying what went wrong
# in its own words (a rate limit is the usual answer).
def fetch_json(url, hops: 5)
  abort "#{url} redirects too far" if hops.zero?

  uri = URI.parse(url)
  response = Net::HTTP.get_response(uri)
  case response
  when Net::HTTPRedirection then fetch_json(response["location"], hops: hops - 1)
  when Net::HTTPSuccess then response.body
  else abort "#{url} answered #{response.code} #{response.message}: #{response.body[0, 200]}"
  end
end

# The engine held to exact arithmetic, through its command line
# (engine/check). Part of the default task: it is cheap, it is the only
# thing that runs the `torobi-engine` binary, and a check nobody runs is
# not a check.
namespace :engine do
  desc "run the engine's command line against a closed-form oracle"
  task :check do
    with_mlx
    sh "cargo build -q -p torobi-engine --bin torobi-engine"
    sh RbConfig.ruby, "engine/check/generate.rb"
    sh RbConfig.ruby, "engine/check/verify.rb"
  end
end

# M1's last exit condition (docs/plan.md section 9.1): the gem installs
# somewhere clean and takes a step there.
#
# Not part of `rake default`, because it builds the extension a second
# time from what `spec.files` ships rather than from `target/`, which is
# the point of it and also a few minutes. It is what to run before
# releasing, and after touching the gemspec, extconf.rb or the metallib's
# path.
desc "build the gem, install it into a directory of its own, and take a step there"
task :smoke do
  require "tmpdir"
  Dir.mktmpdir("torobi-smoke") do |dir|
    package = File.join(dir, "torobi.gem")
    home = File.join(dir, "gems")
    sh "gem build torobi.gemspec -o #{package.shellescape}"
    # Bundler is taken out of the environment rather than pointed
    # elsewhere. It reaches a child through several variables at once
    # (RUBYOPT, RUBYLIB, BUNDLE_*), and any one of them left behind puts
    # this checkout back on the load path, leaving the install untested.
    # tools/smoke.rb refuses that outright rather than passing quietly.
    require "bundler"
    Bundler.with_unbundled_env do
      ENV["GEM_HOME"] = ENV["GEM_PATH"] = home
      sh "gem", "install", package, "--no-document"
      sh RbConfig.ruby, "tools/smoke.rb"
    end
  end
end

Minitest::TestTask.create

# The lint, and what it is for: .rubocop.yml says which rules are on and
# why the rest are not. Guarded because a checkout without the
# development dependencies should still be able to compile.
begin
  require "rubocop/rake_task"
  RuboCop::RakeTask.new
rescue LoadError
  nil
end

task compile: [] # defined by RbSys::ExtensionTask above
task test: %i[compile metallib]

# Lint first: it is five seconds, and the rest is a compile.
DEFAULT = %w[test rust_test rust_test:facade engine:check].freeze
task default: (Rake::Task.task_defined?(:rubocop) ? ["rubocop", *DEFAULT] : DEFAULT)
