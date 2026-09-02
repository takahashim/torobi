# frozen_string_literal: true

require_relative "test_helper"

# The engine half of M1: the same linear-regression graph the spike uses,
# but driven in-process through the narrow session API.
class SessionTest < Minitest::Test
  N = 32

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
    @config = Torobi::GraphConfig.new(models: { "spike" => model })
    @bindings = bindings
  end

  def model
    Torobi.graph do |g|
      x = g.input :x, [nil, 2]
      y = g.input :y, [nil, 1]
      g.output g.mse(g.linear(x, 1, name: "linear"), y)
    end
  end

  # y = 3*x0 - 2*x1 + 1, exactly; a fit this easy makes convergence a
  # meaningful assertion rather than a hopeful one.
  def bindings
    rng = Random.new(7)
    xs = Array.new(N) { [rng.rand(-1.0..1.0), rng.rand(-1.0..1.0)] }
    ys = xs.map { |a, b| [(3 * a) - (2 * b) + 1] }
    {
      inputs: { x: { shape: [N, 2], data: xs.flatten },
                y: { shape: [N, 1], data: ys.flatten } },
      params: { "linear.weight" => { shape: [1, 2], data: [0.0, 0.0] },
                "linear.bias" => { shape: [1], data: [0.0] } }
    }
  end

  def test_a_session_trains_the_graph_it_was_opened_with
    Torobi::Session.open(@config, @bindings) do |s|
      assert_equal 0, s.step
      assert_equal %w[linear.weight linear.bias], s.parameter_paths

      s.adjust(lr: 0.5)
      first = s.run(steps: 1)
      last = s.run(steps: 99)

      assert_equal 100, s.step
      assert_operator last, :<, first * 0.01, "the loss should collapse on an exact fit"
      assert_in_delta 0.0, last, 1e-4

      # It recovered the coefficients it was given.
      weight = s.fetch("linear.weight")
      assert_equal [1, 2], weight[:shape]
      assert_in_delta 3.0, weight[:data][0], 1e-2
      assert_in_delta(-2.0, weight[:data][1], 1e-2)
      assert_in_delta 1.0, s.fetch("linear.bias")[:data][0], 1e-2
    end
  end

  def test_gradients_come_back_by_name_as_copies
    Torobi::Session.open(@config, @bindings) do |s|
      grads = s.gradients
      assert_equal %w[linear.weight linear.bias], grads.keys
      assert_equal [1, 2], grads["linear.weight"][:shape]
      # At w = 0, the gradient of the bias is -2 * mean(y).
      mean_y = @bindings[:inputs][:y][:data].sum / N
      assert_in_delta(-2 * mean_y, grads["linear.bias"][:data][0], 1e-5)
    end
  end

  def test_knobs_are_read_back_and_take_effect
    Torobi::Session.open(@config, @bindings) do |s|
      s.adjust(lr: 0.25)
      assert_in_delta 0.25, s.lr
      slow = s.run(steps: 5)

      fast = Torobi::Session.open(@config, @bindings) { |t| t.adjust(lr: 0.5).run(steps: 5) }
      assert_operator fast, :<, slow, "a larger lr should get further in the same steps"
    end
  end

  # The point of releasing the GVL: another Ruby thread runs while a span
  # is in flight.
  def test_other_threads_proceed_while_a_span_runs
    Torobi::Session.open(@config, @bindings) do |s|
      ticks = 0
      ticker = Thread.new do
        loop do
          ticks += 1
          sleep 0.001
        end
      end
      s.adjust(lr: 0.5)
      s.run(steps: 400)
      ticker.kill
      assert_operator ticks, :>, 1, "the ticker thread did not run during the span"
    end
  end

  def test_mistakes_are_named
    e = assert_raises(RuntimeError) do
      Torobi::Session.open(@config, @bindings.merge(params: {}))
    end
    assert_match(/missing parameter "linear.weight"/, e.message)

    Torobi::Session.open(@config, @bindings) do |s|
      assert_raises(RuntimeError) { s.fetch("nope") }
      assert_raises(ArgumentError) { s.run(steps: 0) }
    end
  end
end
