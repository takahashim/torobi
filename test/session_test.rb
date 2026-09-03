# frozen_string_literal: true

require_relative "test_helper"

# The engine half of M1: the same linear-regression graph the spike uses,
# driven in process through the narrow session API, with every step given
# its own batch.
class SessionTest < Minitest::Test
  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
    @config = Torobi::GraphConfig.new(models: { "spike" => model })
    @weights = { params: { "spike.linear.weight" => { shape: [1, 2], data: [0.0, 0.0] },
                           "spike.linear.bias" => { shape: [1], data: [0.0] } } }
  end

  def model
    Torobi.graph do |g|
      x = g.input :x, [nil, 2]
      y = g.input :y, [nil, 1]
      g.output :loss, g.mse(g.linear(x, 1, name: "linear"), y)
    end
  end

  # y = 3*x0 - 2*x1 + 1, exactly; a fit this easy makes convergence a
  # meaningful assertion rather than a hopeful one.
  def batch(rows, seed: 7)
    rng = Random.new(seed)
    xs = Array.new(rows) { [rng.rand(-1.0..1.0), rng.rand(-1.0..1.0)] }
    ys = xs.map { |a, b| [(3 * a) - (2 * b) + 1] }
    { x: { shape: [rows, 2], data: xs.flatten },
      y: { shape: [rows, 1], data: ys.flatten } }
  end

  # Different data every step, which is what training actually looks like
  # and what the fixed-bindings design could not express.
  def batches(count, rows: 8)
    Array.new(count) { |i| batch(rows, seed: i) }
  end

  def test_a_span_takes_one_batch_per_step
    Torobi::Session.open(@config, weights: @weights) do |s|
      assert_equal 0, s.step
      assert_equal %w[spike.linear.weight spike.linear.bias], s.parameter_paths
      assert_equal %w[x y], s.input_names.sort

      s.adjust(lr: 0.5)
      first = s.step!(batch(8, seed: 0))
      last = s.run(batches(99))

      assert_equal 100, s.step
      assert_operator last, :<, first * 0.05, "the loss should fall across the span"

      # It recovered the coefficients behind every batch.
      weight = s.fetch("spike.linear.weight")
      assert_equal [1, 2], weight.shape
      assert_in_delta 3.0, weight.to_a[0], 5e-2
      assert_in_delta(-2.0, weight.to_a[1], 5e-2)
      assert_in_delta 1.0, s.fetch("spike.linear.bias").to_a[0], 5e-2
    end
  end

  # The symbolic batch dimension is real: batches of different sizes go
  # through the same graph.
  def test_batches_may_differ_in_size_from_step_to_step
    Torobi::Session.open(@config, weights: @weights) do |s|
      s.adjust(lr: 0.3)
      [1, 4, 16, 3].each { |rows| s.step!(batch(rows)) }
      assert_equal 4, s.step
      assert_predicate s.loss, :finite?
    end
  end

  def test_gradients_are_for_the_batch_they_are_given
    Torobi::Session.open(@config, weights: @weights) do |s|
      b = batch(8, seed: 1)
      grads = s.gradients(b)
      assert_equal %w[spike.linear.weight spike.linear.bias], grads.keys
      assert_equal [1, 2], grads["spike.linear.weight"].shape
      # At w = 0, the gradient of the bias is -2 * mean(y).
      ys = b[:y][:data]
      assert_in_delta(-2 * ys.sum / ys.size, grads["spike.linear.bias"].to_a[0], 1e-5)

      # A different batch, a different gradient; and asking did not train.
      refute_equal grads["spike.linear.bias"].to_a, s.gradients(batch(8, seed: 2))["spike.linear.bias"].to_a
      assert_equal 0, s.step
    end
  end

  def test_knobs_are_read_back_and_take_effect
    slow = Torobi::Session.open(@config, weights: @weights) do |s|
      s.adjust(lr: 0.05)
      assert_in_delta 0.05, s.lr
      s.run(batches(5))
    end
    fast = Torobi::Session.open(@config, weights: @weights) { |s| s.adjust(lr: 0.5).run(batches(5)) }
    assert_operator fast, :<, slow, "a larger lr should get further in the same steps"
  end

  # The point of releasing the GVL: another Ruby thread runs while a span
  # is in flight.
  def test_other_threads_proceed_while_a_span_runs
    Torobi::Session.open(@config, weights: @weights) do |s|
      ticks = 0
      ticker = Thread.new do
        loop do
          ticks += 1
          sleep 0.001
        end
      end
      s.adjust(lr: 0.5)
      s.run(batches(400, rows: 4))
      ticker.kill
      assert_operator ticks, :>, 1, "the ticker thread did not run during the span"
    end
  end

  # Watching a run is not serving it. The numbers a watcher reads come
  # from the last committed step, and what the run is made of was settled
  # when it opened, so neither waits for the engine.
  def test_a_watcher_reads_the_run_while_a_span_is_in_flight
    Torobi::Session.open(@config, weights: @weights) do |s|
      watched = %i[step loss parameter_paths input_names node_names]
      refused = Hash.new(0)
      answered = Hash.new(0)

      span = Thread.new { s.run(batches(400, rows: 4)) }
      while span.alive?
        watched.each do |name|
          s.public_send(name)
          answered[name] += 1
        rescue Torobi::Busy
          refused[name] += 1
        end
      end
      span.join

      watched.each do |name|
        assert_equal 0, refused[name], "#{name} should not wait for a step"
        assert_operator answered[name], :>, 0
      end
    end
  end

  # The other half of that line. What is trained is state, because
  # freezing moves it, so asking for it goes to the engine and is refused
  # when there is no engine to ask. The names are facts about the run and
  # outlive it.
  def test_what_a_run_is_made_of_outlives_it_and_what_it_is_doing_does_not
    s = Torobi::Session.open(@config, weights: @weights)
    paths = s.parameter_paths
    s.close

    assert_equal paths, s.parameter_paths
    assert_raises(Torobi::SessionClosed) { s.trainable }
  end

  # Where the parameters come from is a keyword, not a position: model
  # import adds loading from a file, and the GraphConfig's declared
  # initializers are a natural third. Saying which one is meant should not
  # be a matter of what type the second argument happens to be.
  def test_a_session_says_which_parameters_it_wants
    e = assert_raises(ArgumentError) { Torobi::Session.open(@config) }

    assert_match(/needs its parameters/, e.message)
    assert_match(/weights:/, e.message)
  end

  def test_mistakes_are_named
    e = assert_raises(Torobi::StepError) do
      Torobi::Session.open(@config, weights: { params: {} })
    end
    assert_match(/missing parameter "spike.linear.weight"/, e.message)

    Torobi::Session.open(@config, weights: @weights) do |s|
      assert_raises(Torobi::StepError) { s.fetch("nope") }
      assert_raises(ArgumentError) { s.run([]) }

      # A batch that omits an input, names one the graph does not have, or
      # contradicts the declared shape.
      e = assert_raises(Torobi::StepError) { s.step!({ x: batch(2)[:x] }) }
      assert_match(/missing input "y"/, e.message)

      e = assert_raises(Torobi::StepError) { s.step!(batch(2).merge(z: batch(2)[:x])) }
      assert_match(/no input named "z"/, e.message)

      wrong = batch(2)
      wrong[:x] = { shape: [2, 3], data: [0.0] * 6 }
      e = assert_raises(Torobi::StepError) { s.step!(wrong) }
      assert_match(/dimension 1 is 3, declared 2/, e.message)

      short = batch(2)
      short[:x] = { shape: [2, 2], data: [0.0] }
      e = assert_raises(Torobi::StepError) { s.step!(short) }
      assert_match(/1 values for shape/, e.message)
    end
  end
end
