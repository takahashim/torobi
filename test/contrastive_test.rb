# frozen_string_literal: true

require_relative "test_helper"

# The four pieces a sentence-embedding fine-tune is made of, together:
# a ModernBERT encoder, a mean over the tokens a row actually has, an
# in-batch contrastive loss, and a gradient cache that lets the batch be
# larger than the machine can hold at once (docs/plan.md section 15.37).
#
# Each has its own test. What this one asks is whether they compose,
# because they only meet in a real recipe: the pooling has to survive
# being taped and back-propagated through, the loss has to read across
# rows that were encoded in separate passes, and the cache has to hand
# each part a seed for its own rows and no others.
#
# The claim is exact. A cached step is the chain rule written out, so it
# must land where the same batch trained in one piece lands, to f32
# rounding, and not merely close enough to look trained.
class ContrastiveTest < Minitest::Test
  SEQ = 6
  PAIRS = 3
  ROWS = 2 * PAIRS
  # sentence-transformers' default temperature of 0.05, as its reciprocal.
  SCALE = 20.0

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  def config
    @config ||= Torobi::Models::ModernBERT.from_hash(
      "vocab_size" => 12, "hidden_size" => 8, "intermediate_size" => 16,
      "num_hidden_layers" => 3, "num_attention_heads" => 2, "local_attention" => 4,
      "global_attn_every_n_layers" => 3
    )
  end

  def width = config.hidden_size

  # Queries first, then the document each one matches, which is the order
  # the loss reads them in and the order the parts arrive in. Different
  # lengths, so the pooling is doing something.
  def queries = [[1, 5, 2], [1, 7, 9, 4, 2], [1, 3, 2]]
  def documents = [[1, 5, 8, 2], [1, 7, 9, 2], [1, 3, 6, 10, 11, 2]]
  def rows = queries + documents

  def embedder(rows: nil)
    Torobi::Models::ModernBERT.embedder(config, seq: SEQ, pooling: :mean, rows:)
  end

  # Multiple negatives ranking loss: every other document in the batch is
  # a negative for this query, and the only supervision is which pairs
  # were put in together.
  def mnrl(g, e)
    pair = e.reshape(shape: [2, -1, width])
    half = ->(i) { pair.slice(axis: 0, start: i, length: 1).reshape(shape: [-1, width]) }
    q = half.call(0)
    d = half.call(1)
    matched = g.sum(q * d, axes: [-1]) * SCALE
    all = g.matmul(q, d.transpose(axes: [1, 0])) * SCALE
    g.sum(all.exp, axes: [-1]).log - matched
  end

  # What a machine that could hold the whole batch would run: one graph,
  # one step, no cache.
  def whole
    @whole ||= begin
      model = embedder(rows: ROWS)
      Torobi::GraphConfig.new(
        models: { m: model },
        objective: Torobi.objective(m: model) { |g|
          g.output :loss, g.mean(mnrl(g, g.from_model(:m, :embedding)))
        }
      )
    end
  end

  # The same encoder, told to back-propagate a seed instead. This is the
  # only thing the recipe adds to the model: the loss it is really being
  # trained on is not in this graph at all.
  def seeded
    @seeded ||= begin
      model = embedder
      Torobi::GraphConfig.new(
        models: { m: model },
        objective: Torobi.objective(m: model) { |g|
          g.output :loss, g.sum(g.from_model(:m, :embedding) * g.from_batch(:seed, [nil, nil]))
        }
      )
    end
  end

  # The loss over representations computed elsewhere. It holds no
  # parameters and never steps: it is opened to be differentiated by its
  # input.
  def over_vectors
    @over_vectors ||= begin
      graph = Torobi.graph do |g|
        g.output :loss, g.mean(mnrl(g, g.input(:vectors, [ROWS, width])))
      end
      Torobi::GraphConfig.new(models: { m: graph }, train: [])
    end
  end

  def weights
    @weights ||= begin
      rng = Random.new(11)
      params = embedder.parameters.to_h do |spec|
        [
          "m.#{spec.path}",
          { shape: spec.shape,
            data: Array.new(spec.shape.reduce(1, :*)) { rng.rand(-0.4..0.4) } }
        ]
      end
      { params: }
    end
  end

  def parts(per_part)
    rows.each_slice(per_part).map do |slice|
      Torobi::Models::ModernBERT.batch(config, slice, seq: SEQ, pooling: :mean)
    end
  end

  def open_encoder(graph_config, &)
    Torobi::Session.open(graph_config, weights:,
                                       optimizer: { kind: :sgd, lr: 0.1 }, &)
  end

  def parameters_of(session)
    weights.fetch(:params).keys.sort.flat_map { |path| session.fetch(path).to_a }
  end

  # One cached step, and what the parameters were left at.
  def cached_step(per_part)
    open_encoder(seeded) do |session|
      Torobi::Session.open(over_vectors, weights: { params: {} }) do |loss|
        cache = Torobi::GradCache.new(session, loss:, tap: "m.embedding",
                                      into: :vectors, seed: :seed)
        [cache.step(parts(per_part)), parameters_of(session), cache]
      end
    end
  end

  def test_the_four_pieces_together_land_where_the_whole_batch_would
    direct, direct_parameters = open_encoder(whole) do |s|
      batch = Torobi::Models::ModernBERT.batch(config, rows, seq: SEQ, pooling: :mean)
      [s.step!(batch), parameters_of(s)]
    end

    loss, cached_parameters, cache = cached_step(2)

    assert_equal 3, cache.parts
    assert_equal ROWS, cache.rows
    assert_in_delta direct, loss, 1e-5, "the loss is the loss over the whole batch"
    apart = direct_parameters.zip(cached_parameters).map { |want, got| (want - got).abs }.max

    assert_operator apart, :<, 1e-5, "the step is the step the whole batch would have taken"
  end

  # However the rows are divided. How much is held at once is a fact about
  # the machine, and the answer must not be one.
  def test_the_answer_does_not_depend_on_how_the_batch_was_split
    _, in_one = cached_step(ROWS)
    _, in_threes = cached_step(3)
    _, one_at_a_time = cached_step(1)

    [in_threes, one_at_a_time].each do |other|
      apart = in_one.zip(other).map { |a, b| (a - b).abs }.max

      assert_operator apart, :<, 1e-5
    end
  end

  # And that the whole thing trains: the matching document is what the
  # loss is asking for, and a few steps should make it more likely.
  def test_the_loss_falls_over_a_few_cached_steps
    losses = open_encoder(seeded) do |session|
      Torobi::Session.open(over_vectors, weights: { params: {} }) do |loss|
        cache = Torobi::GradCache.new(session, loss:, tap: "m.embedding",
                                      into: :vectors, seed: :seed)
        Array.new(5) { cache.step(parts(2)) }
      end
    end

    # Not monotonically: plain SGD at this rate steps past the minimum
    # and comes back, which is a fact about the optimizer and not about
    # the pieces being tested. What is claimed is the fall.
    assert_operator losses.last, :<, losses.first / 2,
                    "five steps of a contrastive loss should separate the pairs"
    assert(losses.all? { |loss| loss.finite? && loss >= 0.0 },
           "a cross-entropy over the batch is finite and not negative")
  end
end
