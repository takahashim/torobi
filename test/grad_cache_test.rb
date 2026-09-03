# frozen_string_literal: true

require_relative "test_helper"

# A gradient cache reaches where the batch it stands for would have
# (docs/plan.md section 15.37).
#
# The claim is exact rather than approximate: encoding the parts, taking
# the loss over all of the representations, and back-propagating each part
# with that loss's gradient by its own rows is the chain rule written out.
# So the test is an equality, held to f32 rounding, against the same batch
# trained in one piece.
class GradCacheTest < Minitest::Test
  DIM = 4
  WIDTH = 3
  PAIRS = 3
  ROWS = 2 * PAIRS
  SCALE = 5.0

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  # The encoder: rows in, one vector each. Its own loss is a product with
  # a seed the caller supplies, which is how a cached backward is asked
  # for; with a zero seed it is a forward that says nothing.
  def encoder
    @encoder ||= Torobi.graph do |g|
      x = g.input :x, [nil, DIM]
      seed = g.input :seed, [nil, nil]
      h = g.linear(x, WIDTH, name: "l", bias: false).tanh
      e = g.name("e", h / g.sum(h.square, axes: [-1], keepdims: true).sqrt)
      g.output :loss, g.sum(e * seed)
    end
  end

  # The loss over representations that were computed elsewhere. Queries
  # first, then their documents, which is the order the parts arrive in.
  def scores
    @scores ||= Torobi.graph do |g|
      v = g.input :vectors, [ROWS, WIDTH]
      pair = v.reshape(shape: [2, -1, WIDTH])
      half = ->(i) { pair.slice(axis: 0, start: i, length: 1).reshape(shape: [-1, WIDTH]) }
      q = half.call(0)
      d = half.call(1)
      matched = g.sum(q * d, axes: [-1]) * SCALE
      all = g.matmul(q, d.transpose(axes: [1, 0])) * SCALE
      g.output :loss, g.mean(g.sum(all.exp, axes: [-1]).log - matched)
    end
  end

  # The same computation in one graph, which is what a machine that could
  # hold the whole batch would run.
  def whole
    @whole ||= Torobi.graph do |g|
      x = g.input :x, [ROWS, DIM]
      h = g.linear(x, WIDTH, name: "l", bias: false).tanh
      e = h / g.sum(h.square, axes: [-1], keepdims: true).sqrt
      pair = e.reshape(shape: [2, -1, WIDTH])
      half = ->(i) { pair.slice(axis: 0, start: i, length: 1).reshape(shape: [-1, WIDTH]) }
      q = half.call(0)
      d = half.call(1)
      matched = g.sum(q * d, axes: [-1]) * SCALE
      all = g.matmul(q, d.transpose(axes: [1, 0])) * SCALE
      g.output :loss, g.mean(g.sum(all.exp, axes: [-1]).log - matched)
    end
  end

  def weights
    rng = Random.new(4)
    { params: { "m.l.weight" => { shape: [WIDTH, DIM],
                                  data: Array.new(WIDTH * DIM) { rng.rand(-0.5..0.5) } } } }
  end

  # Rows that are not symmetric, so a gradient that lost track of which
  # row was which would show.
  def rows
    rng = Random.new(9)
    Array.new(ROWS) { Array.new(DIM) { rng.rand(-1.0..1.0) } }
  end

  def batches(rows, per_part)
    rows.each_slice(per_part).map { |slice| { x: Torobi::TensorData.nested(slice) } }
  end

  def open_encoder(graph, &)
    Torobi::Session.open(Torobi::GraphConfig.new(models: { m: graph }),
                         weights:, optimizer: { kind: :sgd, lr: 0.5 }, &)
  end

  # A session opened to be read rather than trained: it holds no
  # parameters, and `train: []` is how that is said rather than found out.
  def loss_over(graph, &)
    Torobi::Session.open(Torobi::GraphConfig.new(models: { m: graph }, train: []),
                         weights: { params: {} }, &)
  end

  def cache_step(parts)
    open_encoder(encoder) do |session|
      loss_over(scores) do |loss|
        cache = Torobi::GradCache.new(session, loss:, tap: "m.e",
                                      into: :vectors, seed: :seed)
        [cache.step(parts), session.fetch("m.l.weight").to_a, cache]
      end
    end
  end

  def test_a_cached_step_lands_where_the_whole_batch_would
    data = rows
    direct, direct_weight = open_encoder(whole) do |s|
      [s.step!({ x: Torobi::TensorData.nested(data) }), s.fetch("m.l.weight").to_a]
    end

    cached, cached_weight, cache = cache_step(batches(data, 2))

    assert_equal 3, cache.parts
    assert_equal ROWS, cache.rows
    assert_in_delta direct, cached, 1e-5, "the loss is the same loss"
    direct_weight.zip(cached_weight).each do |want, got|
      assert_in_delta want, got, 1e-5, "the step is the same step"
    end
  end

  # However the rows are divided. The parts are only how much is held at
  # once; what the loss sees is all of them.
  def test_the_answer_does_not_depend_on_how_the_batch_was_split
    data = rows
    _, in_one = cache_step(batches(data, ROWS))
    _, in_threes = cache_step(batches(data, 2))
    _, one_at_a_time = cache_step(batches(data, 1))

    in_one.zip(in_threes, one_at_a_time).each do |whole_batch, threes, singles|
      assert_in_delta whole_batch, threes, 1e-5
      assert_in_delta whole_batch, singles, 1e-5
    end
  end

  def test_a_step_with_no_parts_is_refused
    open_encoder(encoder) do |session|
      loss_over(scores) do |loss|
        cache = Torobi::GradCache.new(session, loss:, tap: "m.e", into: :vectors, seed: :seed)

        assert_raises(ArgumentError) { cache.step([]) }
      end
    end
  end
end
