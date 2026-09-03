# frozen_string_literal: true

require_relative "test_helper"

# One transformer block, forward and gradient (docs/plan.md M3a).
#
# The oracle is finite differences, and it is chosen rather than settled
# for. A Python MLX oracle compares Torobi against another implementation
# of the same idea; central differences compare the engine's gradient
# against its own forward, which is the thing autodiff is supposed to
# agree with. It needs no second toolchain, it fails closed by
# construction, and it catches exactly the mistake this milestone is about:
# an op whose forward is right and whose backward is not.
#
# What it does not catch is a forward that is wrong in the same way as the
# gradient. That is the Python MLX oracle's job, and it is M3b's.
class BlockTest < Minitest::Test
  DIM = 8
  HEADS = 2
  SEQ = 4
  ROWS = 2

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  # attention over one head group, then a GeGLU, each with a residual and a
  # norm. The shapes are small on purpose: finite differences cost one
  # forward per parameter element, twice.
  def block
    Torobi.graph do |g|
      x = g.input :x, [nil, SEQ, DIM]
      target = g.input :y, [nil, SEQ, DIM]

      normed = g.layer_norm(x, name: "attn_norm", bias: true)
      q = g.linear(normed, DIM, name: "q", bias: false)
      k = g.linear(normed, DIM, name: "k", bias: false)
      v = g.linear(normed, DIM, name: "v", bias: false)
      attended = g.sdpa(q, k, v)
      x = x + g.linear(attended, DIM, name: "o", bias: false)

      x = x + g.geglu(g.layer_norm(x, name: "mlp_norm"), DIM * 2, name: "mlp")
      g.output :loss, g.mse(x, target)
    end
  end

  def config = Torobi::GraphConfig.new(models: { "m" => block })

  # Deterministic and small, so a difference is the engine's rather than
  # the data's.
  def weights
    rng = Random.new(7)
    params = config.parameters.to_h do |parameter|
      shape = parameter.spec.shape
      data = Array.new(shape.reduce(1, :*)) { rng.rand(-0.4..0.4) }
      [parameter.qualified_path, { shape:, data: }]
    end
    { params: }
  end

  def batch
    rng = Random.new(11)
    size = ROWS * SEQ * DIM
    { x: { shape: [ROWS, SEQ, DIM], data: Array.new(size) { rng.rand(-1.0..1.0) } },
      y: { shape: [ROWS, SEQ, DIM], data: Array.new(size) { rng.rand(-1.0..1.0) } } }
  end

  def test_a_block_runs_forward_and_reports_every_parameter
    Torobi::Session.open(config, weights: weights) do |s|
      loss = s.evaluate(batch)

      assert_predicate loss, :finite?, "the block should produce a number"
      # Everything the block declares is there, and everything is trained.
      assert_equal s.parameter_paths.sort, s.trainable.sort
      assert_includes s.parameter_paths, "m.attn_norm.weight"
      assert_includes s.parameter_paths, "m.mlp.wi.weight"
    end
  end

  # The claim: for every parameter, autodiff agrees with the forward it is
  # differentiating. Central differences, at a step chosen so that
  # truncation and f32 rounding are both small.
  def test_every_gradient_agrees_with_central_differences
    @differences_batch = batch
    b = @differences_batch
    Torobi::Session.open(config, weights: weights) do |s|
      analytic = s.gradients(b)
      # Without this the comparison could pass on gradients that are all
      # zero, which would agree with anything.
      biggest = analytic.values.flat_map { |g| g[:data] }.map(&:abs).max

      assert_operator biggest, :>, 1e-2, "the block should have gradients worth checking"
      assert(analytic.all? { |_, g| g[:data].any? { |v| v.abs > 1e-6 } },
             "every parameter should be reached by the backward pass")

      worst = { path: nil, at: nil, delta: 0.0 }

      s.parameter_paths.each do |path|
        held = s.fetch(path)
        # A few positions per parameter: every one would be thousands of
        # forwards, and a wrong backward is wrong everywhere.
        positions = sample(held[:data].size)
        positions.each do |i|
          numeric = central_difference(s, path, held, i)
          delta = (analytic.fetch(path)[:data][i] - numeric).abs
          worst = { path:, at: i, delta: } if delta > worst[:delta]
        end
        s.put(path, held)
      end

      # Measured at 5.4e-6 across the block, so this leaves an order of
      # magnitude and still catches a backward that is wrong by anything
      # like the size of the gradients themselves.
      assert_operator worst[:delta], :<, 1e-4,
                      "#{worst[:path]}[#{worst[:at]}] disagrees with the forward"
    end
  end

  # A block trains: the loss falls, and it falls on parameters that moved.
  def test_a_block_trains
    b = batch
    Torobi::Session.open(config, weights: weights,
                         optimizer: { kind: :adamw, lr: 0.02 }) do |s|
      before = s.evaluate(b)
      before_weight = s.fetch("m.q.weight")[:data]
      s.repeat(b, steps: 30)
      after = s.evaluate(b)

      assert_operator after, :<, before * 0.9, "the block should learn its own batch"
      refute_equal before_weight, s.fetch("m.q.weight")[:data]
    end
  end

  private

  # Up to four positions, spread across the parameter.
  def sample(size)
    return (0...size).to_a if size <= 4

    Array.new(4) { |i| (i * (size - 1)) / 3 }
  end

  # (loss(w + h) - loss(w - h)) / 2h, with dropout off (evaluate draws no
  # randomness, which is what makes this comparable at all).
  def central_difference(session, path, held, index, step: 1e-2)
    moved = ->(delta) do
      data = held[:data].dup
      data[index] += delta
      session.put(path, { shape: held[:shape], data: })
      session.evaluate(@differences_batch)
    end
    (moved.call(step) - moved.call(-step)) / (2 * step)
  end
end
