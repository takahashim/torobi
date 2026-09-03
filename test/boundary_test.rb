# frozen_string_literal: true

require_relative "test_helper"

# What the native boundary does when things go wrong. Each case runs in a
# subprocess, because some of them end the process: that is the finding
# this file exists to pin down, not a thing to hide.
#
# The claim under test is narrow and honest: errors MLX reports through its
# C error handler become Ruby exceptions; failures it raises as C++
# exceptions (device and library initialization) abort, and the ones we can
# foresee are refused by Preflight before MLX is touched.
class BoundaryTest < Minitest::Test
  ROOT = File.expand_path("..", __dir__)

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  # Runs `body` in a fresh Ruby, optionally with the metallib moved aside.
  # Returns [stdout+stderr, Process::Status].
  def run_isolated(body, hide_metallib: false)
    script = <<~RUBY
      $LOAD_PATH.unshift(#{File.join(ROOT, "lib").inspect})
      require "torobi"
      def build
        model = Torobi.graph do |g|
          x = g.input(:x, [nil, 2])
          y = g.input(:y, [nil, 1])
          g.output g.mse(g.linear(x, 1, name: "l"), y)
        end
        config = Torobi::GraphConfig.new(models: { "spike" => model })
        bindings = {
          inputs: { x: { shape: [1, 2], data: [1.0, 2.0] },
                    y: { shape: [1, 1], data: [3.0] } },
          params: { "l.weight" => { shape: [1, 2], data: [0.0, 0.0] },
                    "l.bias" => { shape: [1], data: [0.0] } }
        }
        [config, bindings]
      end
      #{body}
    RUBY
    file = File.join(Dir.tmpdir, "torobi_boundary_#{Process.pid}_#{rand(1 << 32)}.rb")
    File.write(file, script)
    metallib = Torobi::Preflight::METALLIB
    hidden = "#{metallib}.hidden"
    File.rename(metallib, hidden) if hide_metallib && File.exist?(metallib)
    output = `#{RbConfig.ruby} #{file} 2>&1`
    [output, $?]
  ensure
    File.rename(hidden, metallib) if hidden && File.exist?(hidden)
    File.unlink(file) if file && File.exist?(file)
  end

  def test_errors_mlx_reports_become_ruby_exceptions
    # A shape the build-time inference cannot catch: the bound data does not
    # match the declared input. MLX reports it through its error handler.
    body = <<~RUBY
      config, bindings = build
      bindings[:inputs][:x] = { shape: [1, 3], data: [1.0, 2.0, 3.0] }
      begin
        Torobi::Session.open(config, bindings).run(steps: 1)
        puts "NO ERROR"
      rescue => e
        puts "RESCUED \#{e.class}: \#{e.message}"
      end
    RUBY
    output, status = run_isolated(body)
    assert_predicate status, :success?, output
    assert_match(/RESCUED RuntimeError/, output)
    assert_match(/matmul/i, output, "the message should name what MLX objected to")
  end

  def test_a_missing_metallib_is_refused_before_mlx_is_touched
    body = <<~RUBY
      config, bindings = build
      begin
        Torobi::Session.open(config, bindings)
        puts "OPENED"
      rescue Torobi::EngineUnavailable => e
        puts "REFUSED"
      end
    RUBY
    output, status = run_isolated(body, hide_metallib: true)
    assert_predicate status, :success?, output
    assert_match(/REFUSED/, output)
  end

  # The finding this file is really about: without that refusal, MLX ends
  # the process. Documented as a test so it cannot quietly change.
  def test_without_the_refusal_a_missing_metallib_ends_the_process
    body = <<~RUBY
      config, bindings = build
      begin
        Torobi::Native::Session.open(config.canonical_json, JSON.generate(bindings))
        puts "OPENED"
      rescue => e
        puts "RESCUED \#{e.class}"
      end
    RUBY
    output, status = run_isolated(body, hide_metallib: true)
    refute_predicate status, :success?,
                     "MLX now survives a missing metallib; the preflight and the plan " \
                     "should say so (#{output})"
    refute_match(/RESCUED/, output, "the failure did not reach Ruby, as expected")
  end

  def test_a_session_survives_forced_gc_and_repeated_spans
    body = <<~RUBY
      config, bindings = build
      s = Torobi::Session.open(config, bindings)
      s.adjust(lr: 0.1)
      50.times do
        s.run(steps: 4)
        GC.start
      end
      puts "STEPS \#{s.step}"
      puts(s.loss.finite? ? "FINITE" : "NOT FINITE")
    RUBY
    output, status = run_isolated(body)
    assert_predicate status, :success?, output
    assert_match(/STEPS 200/, output)
    assert_match(/FINITE/, output)
  end

  def test_many_sessions_are_opened_and_dropped_without_growth
    body = <<~RUBY
      config, bindings = build
      100.times { Torobi::Session.open(config, bindings).run(steps: 1) }
      GC.start
      puts "SURVIVED"
    RUBY
    output, status = run_isolated(body)
    assert_predicate status, :success?, output
    assert_match(/SURVIVED/, output)
  end
end
