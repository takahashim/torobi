#!/usr/bin/env ruby
# frozen_string_literal: true

# What a checkpoint costs at the size M3b works at, and how much of that is
# the read-back.
#
# `checkpoint::write` lays the files out somewhere that does not count,
# reads the whole thing back, and only then lets it take the name. The
# read-back is not free: at 130M parameters it is half a gigabyte of
# parameters plus twice that in AdamW moments, parsed and loaded again.
# docs/plan.md section 15.12 left it as a decision to make once there was a
# number rather than a feeling. This is the number.
#
#   ruby bench/checkpoint.rb <ruri-v3-130m-dir>
#
# The read-back is measured through `restore`, which reads the same files
# and checks the same things before committing them. It is the closer of
# the two available bounds, and it is an upper one: restore also installs
# what it read.
#
# Runs itself in a child process with a cap, for the reason bench/overfit.rb
# gives at more length.

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "torobi"
require "json"
require "tmpdir"

CAP = 8 * 1024 * 1024 * 1024
SEQ = 32

def timed
  started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  yield
  Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
end

def megabytes(dir)
  Dir.glob(File.join(dir, "*")).sum { |path| File.size(path) } / 1024.0 / 1024
end

def child(run, dir)
  run.cap!
  config = Torobi::Models::ModernBERT.from_config_file(File.join(dir, "config.json"))
  encoder = Torobi::Models::ModernBERT.graph(config, seq: SEQ)
  graph = Torobi::GraphConfig.new(
    models: { m: encoder },
    objective: Torobi.objective(m: encoder) { |g|
      g.output :loss, g.mean(g.from_model(:m, :hidden).square)
    }
  )
  ids = [Array.new(SEQ) { |i| (i * 137) + 11 }]
  batch = Torobi::Models::ModernBERT.batch(config, ids, seq: SEQ)

  Torobi::Session.open(graph, pretrained: { m: File.join(dir, "model.safetensors") },
                       optimizer: { kind: :adamw, lr: 1e-5 }) do |s|
    puts "#{s.parameter_paths.size} parameters"
    # One step first: AdamW's moments do not exist until a step has been
    # taken, and they are two thirds of what a checkpoint holds.
    s.step!(batch)

    to = File.join(run.dir, "c")
    first = timed { s.checkpoint!(to) }
    # The second one has a previous checkpoint to displace, which is the
    # case a run is in from step two onwards.
    again = timed { s.checkpoint!(to) }
    back = timed { s.restore(to) }

    puts format("checkpoint!: %.3fs (first), %.3fs (over a previous)", first, again)
    puts format("restore:     %.3fs", back)
    puts format("on disk:     %.0f MB", megabytes(to))
    puts format("the read-back is at most %.0f%% of a checkpoint", 100.0 * back / again)
    puts format("a step is %.3fs, so a checkpoint every 200 steps costs %.2f%% of the run",
                (step = timed { s.step!(batch) }), 100.0 * again / (200 * step))
  end
end

if ENV[Torobi::Runner::DIRECTORY_VARIABLE]
  Torobi::Runner.child! { |run| child(run, ARGV.fetch(0)) }
end

dir = ARGV.fetch(0) { abort "usage: checkpoint.rb <ruri-v3-130m-dir>" }
run_dir = File.join(Dir.tmpdir, "torobi-checkpoint-#{Process.pid}")
runner = Torobi::Runner.new([RbConfig.ruby, __FILE__, dir], dir: run_dir,
                            memory_limit: CAP).start
outcome = runner.wait
abort "the run did not finish: #{outcome.inspect}" unless outcome.finished?
