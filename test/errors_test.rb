# frozen_string_literal: true

require_relative "test_helper"
require "benchmark"

# The error contract of docs/plan.md section 5A.4: three classes, three
# moments, three things that survive. This holds the implementation to it.
class ErrorsTest < Minitest::Test
  def test_every_class_is_one_family
    [Torobi::ConfigError, Torobi::StepError, Torobi::EngineUnavailable].each do |klass|
      assert_operator klass, :<, Torobi::Error, "#{klass} should be a Torobi::Error"
      assert_operator klass, :<, StandardError
    end
  end

  # A description that is wrong is found while building it, with no engine
  # anywhere near.
  def test_config_errors_come_from_building
    assert_raises(Torobi::ConfigError) do
      Torobi.graph { |g| g.output :out, g.matmul(g.input(:a, [2, 3]), g.input(:b, [4, 5])) }
    end
    assert_raises(Torobi::ConfigError) { Torobi::GraphConfig.new(models: {}) }
    assert_raises(Torobi::ConfigError) do
      Torobi.graph { |g| g.output :out, g.input(:x, [2]).softmax(axis: "middle") }
    end
  end

  def test_step_errors_leave_the_session_usable
    skip "extension not compiled" unless defined?(Torobi::Session)

    model = Torobi.graph do |g|
      x = g.input :x, [nil, 2]
      y = g.input :y, [nil, 1]
      g.output :loss, g.mse(g.linear(x, 1, name: "l"), y)
    end
    config = Torobi::GraphConfig.new(models: { "m" => model })
    weights = { params: { "m.l.weight" => { shape: [1, 2], data: [0.0, 0.0] },
                          "m.l.bias" => { shape: [1], data: [0.0] } } }
    good = { x: { shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0] },
             y: { shape: [2, 1], data: [1.0, 2.0] } }

    Torobi::Session.open(config, weights) do |s|
      # 2 rows of x against 3 of y: neither the build-time inference nor
      # the bind check can settle a symbolic dimension, so MLX does.
      mismatched = good.merge(y: { shape: [3, 1], data: [1.0, 2.0, 3.0] })
      assert_raises(Torobi::StepError) { s.step!(mismatched) }
      assert_equal 0, s.step, "a refused step is not a step"

      # The session took the next one, which is what StepError promises.
      s.step!(good)
      assert_equal 1, s.step
      assert_predicate s.loss, :finite?
    end
  end

  # The hierarchy is a contract about what survives a failure, so a caller
  # can act on the class rather than on the message
  # (notes/SESSION_CONCURRENCY_SPEC.md section 6).
  def test_busy_is_a_step_error_and_the_rest_are_not
    # "the engine refused, the session is still mine" is true of Busy.
    assert_operator Torobi::Busy, :<, Torobi::StepError

    # It is false of these: the session is gone, so a `rescue StepError`
    # that retried would be wrong.
    [Torobi::SessionPoisoned, Torobi::SessionClosed].each do |klass|
      assert_operator klass, :<, Torobi::Error
      refute_operator klass, :<, Torobi::StepError
    end
  end

  def test_engine_unavailable_is_raised_before_the_engine_is_touched
    skip "extension not compiled" unless defined?(Torobi::Session)
    # Preflight's contract; the abort it prevents is pinned in
    # BoundaryTest, which must run in a subprocess to survive it.
    assert_respond_to Torobi::Preflight, :check!
    assert_operator Torobi::EngineUnavailable, :<, Torobi::Error
  end

  # A review reached a machine where the metallib was present and MLX still
  # ended the process on device initialization. Preflight therefore asks in
  # a subprocess, where an abort is an exit status rather than the end of
  # this one. Once per process: the answer is memoized.
  def test_the_device_is_probed_where_an_abort_is_survivable
    skip "extension not compiled" unless defined?(Torobi::Session)
    Torobi::Preflight.forget_probe!
    assert Torobi::Preflight.probe!, "MLX should start on this machine"

    first = Benchmark.realtime { Torobi::Preflight.check! }
    second = Benchmark.realtime { Torobi::Preflight.check! }
    assert_operator second, :<, first, "the probe should be asked once, not per session"
  end
end
