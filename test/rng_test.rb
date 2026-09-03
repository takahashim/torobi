# frozen_string_literal: true

require_relative "test_helper"
require "tmpdir"

# The RNG is state, not a global (docs/plan.md section 11.1). Dropout is
# its first consumer, and the tests below are about what that buys: the
# same seed draws the same masks, a different seed draws different ones,
# and a resumed run draws what a continuous one would.
class RngTest < Minitest::Test
  ROWS = 32
  DIM = 8

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  # Dropout sits between the parameter and the loss, so the masks a step
  # draws change what it learns.
  def config(p: 0.5)
    model = Torobi.graph do |g|
      x = g.input :x, [nil, DIM]
      y = g.input :y, [nil, 1]
      h = g.dropout(g.linear(x, DIM, name: "hidden"), p)
      g.output :loss, g.mse(g.linear(h, 1, name: "out"), y)
    end
    Torobi::GraphConfig.new(models: { "m" => model })
  end

  def weights
    rng = Random.new(1)
    { params: {
      "m.hidden.weight" => { shape: [DIM, DIM], data: Array.new(DIM * DIM) do
        rng.rand(-0.3..0.3)
      end },
      "m.hidden.bias" => { shape: [DIM], data: Array.new(DIM, 0.0) },
      "m.out.weight" => { shape: [1, DIM], data: Array.new(DIM) { rng.rand(-0.3..0.3) } },
      "m.out.bias" => { shape: [1], data: [0.0] }
    } }
  end

  def batches(count, seed: 2)
    rng = Random.new(seed)
    Array.new(count) do
      xs = Array.new(ROWS * DIM) { rng.rand(-1.0..1.0) }
      ys = Array.new(ROWS) { rng.rand(-1.0..1.0) }
      { x: { shape: [ROWS, DIM], data: xs }, y: { shape: [ROWS, 1], data: ys } }
    end
  end

  def losses(seed:, steps: 6, all: nil)
    all ||= batches(steps)
    Torobi::Session.open(config, weights: weights, optimizer: { kind: :sgd, lr: 0.05 }) do |s|
      s.adjust(seed:)
      all.map { |b| s.step!(b) }
    end
  end

  def test_the_same_seed_draws_the_same_masks
    all = batches(6)

    assert_equal losses(seed: 7, all:), losses(seed: 7, all:)
  end

  def test_a_different_seed_draws_different_ones
    all = batches(6)

    refute_equal losses(seed: 7, all:), losses(seed: 8, all:),
                 "dropout should depend on the seed"
  end

  def test_the_seed_is_readable_state
    Torobi::Session.open(config, weights: weights) do |s|
      assert_equal 0, s.seed, "a session starts from a stated seed, not from chance"
      s.adjust(seed: 42)

      assert_equal 42, s.seed
    end
  end

  # The point of holding the key rather than a global: stopping and
  # resuming reaches the same place, with dropout on.
  def test_a_resumed_run_draws_what_a_continuous_one_would
    Dir.mktmpdir("torobi-rng") do |dir|
      all = batches(8)
      optimizer = { kind: :adamw, lr: 0.02 }
      path = File.join(dir, "half")

      continuous = Torobi::Session.open(config, weights: weights, optimizer:) do |s|
        s.adjust(seed: 5)
        s.run(all)
        s.parameter_paths.to_h { |p| [p, s.fetch(p).to_a] }
      end

      Torobi::Session.open(config, weights: weights, optimizer:) do |s|
        s.adjust(seed: 5)
        s.run(all.first(4))
        s.checkpoint!(path)
      end
      resumed = Torobi::Session.open(config, weights: weights, optimizer:) do |s|
        s.restore(path)

        assert_equal 5, s.seed, "the seed came back with the checkpoint"
        s.run(all.drop(4))
        s.parameter_paths.to_h { |p| [p, s.fetch(p).to_a] }
      end

      continuous.each do |path_name, values|
        values.each_with_index do |value, i|
          assert_in_delta value, resumed.fetch(path_name)[i], 1e-5,
                          "#{path_name}[#{i}] after resuming through dropout"
        end
      end
    end
  end

  # Two dropouts in one graph must not draw the same mask, or they would be
  # one dropout applied twice. The loss below is the mean square of their
  # difference, which is zero exactly when the masks agree.
  def test_two_dropouts_in_one_step_draw_differently
    model = Torobi.graph do |g|
      x = g.input :x, [nil, DIM]
      # A trainable parameter, so the session has something to differentiate.
      h = g.linear(x, DIM, name: "l")
      g.output :loss, g.mean((g.dropout(h, 0.5) - g.dropout(h, 0.5)).square)
    end
    conf = Torobi::GraphConfig.new(models: { "m" => model })
    w = { params: {
      "m.l.weight" => { shape: [DIM, DIM], data: Array.new(DIM * DIM, 0.5) },
      "m.l.bias" => { shape: [DIM], data: Array.new(DIM, 1.0) }
    } }
    batch = { x: { shape: [ROWS, DIM], data: Array.new(ROWS * DIM, 1.0) } }

    loss = Torobi::Session.open(conf, weights: w, optimizer: { kind: :sgd, lr: 0.0 }) do |s|
      s.adjust(seed: 3)
      s.step!(batch)
    end

    assert_operator loss, :>, 0.0, "the two dropouts drew the same mask"
  end

  def test_a_rate_outside_zero_to_one_is_refused
    conf = config(p: 1.5)
    e = assert_raises(Torobi::StepError) do
      Torobi::Session.open(conf, weights: weights) { |s| s.step!(batches(1).first) }
    end
    assert_match(/p must be in 0\.\.1/, e.message)
  end

  def test_a_rate_of_zero_is_the_identity
    all = batches(4)
    without = Torobi::Session.open(config(p: 0.0), weights: weights,
                                   optimizer: { kind: :sgd, lr: 0.05 }) do |s|
      s.adjust(seed: 1)
      all.map { |b| s.step!(b) }
    end
    other_seed = Torobi::Session.open(config(p: 0.0), weights: weights,
                                      optimizer: { kind: :sgd, lr: 0.05 }) do |s|
      s.adjust(seed: 99)
      all.map { |b| s.step!(b) }
    end

    assert_equal without, other_seed, "with p = 0 the seed cannot matter"
  end
end
