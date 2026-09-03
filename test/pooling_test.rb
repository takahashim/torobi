# frozen_string_literal: true

require_relative "test_helper"

# Pooling a sequence into one vector, with padding weighed out of it.
#
# The claim worth testing is not that a mean was taken but that the
# padding took no part in it. A mean over the whole sequence is a
# plausible-looking vector that moves when a row's neighbours get longer,
# because padding to the batch's longest row is a fact about the batch and
# not about the text; a model trained that way learns the batching.
class PoolingTest < Minitest::Test
  SEQ = 6
  LONG = [3, 8, 5, 9, 2, 7].freeze
  SHORT = [4, 6, 1].freeze

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  # Small, and with a local layer whose window cuts something off at this
  # sequence length: pooling is downstream of attention, so a mask that
  # leaked would show up in the vector.
  def config
    @config ||= Torobi::Models::ModernBERT.from_hash(
      "vocab_size" => 12, "hidden_size" => 8, "intermediate_size" => 16,
      "num_hidden_layers" => 3, "num_attention_heads" => 2, "local_attention" => 4,
      "global_attn_every_n_layers" => 3
    )
  end

  def embedder(seq: SEQ, pooling: :mean, normalize: false)
    Torobi::Models::ModernBERT.embedder(config, seq:, pooling:, normalize:)
  end

  # The same numbers for every graph here, so two sequence lengths are two
  # views of one model rather than two models.
  def weights(graph)
    rng = Random.new(7)
    params = graph.parameters.to_h do |spec|
      ["m.#{spec.path}",
       { shape: spec.shape, data: Array.new(spec.shape.reduce(1, :*)) { rng.rand(-0.4..0.4) } }]
    end
    { params: }
  end

  # A run opened to be read rather than trained. It still needs a scalar
  # to call the loss, and what that scalar is does not matter: nothing
  # here differentiates it.
  def read_only(graph)
    Torobi::GraphConfig.new(
      models: { m: graph }, train: [],
      objective: Torobi.objective(m: graph) { |g|
        g.output :loss, g.mean(g.from_model(:m, :embedding))
      }
    )
  end

  def embed(rows, seq: SEQ, pooling: :mean)
    graph = embedder(seq:, pooling:)
    Torobi::Session.open(read_only(graph), weights: weights(graph)) do |s|
      s.tap("m.embedding", stat: :full)
      s.evaluate(Torobi::Models::ModernBERT.batch(config, rows, seq:, pooling:))
      s.tapped.fetch("m.embedding").to_a.each_slice(config.hidden_size).to_a
    end
  end

  # The mean is over the tokens the row has, and the padded positions are
  # not among them. Read off the hidden states the same run produced, so
  # this compares the pooling against its own input rather than against a
  # second implementation of the encoder.
  def test_the_mean_is_over_the_tokens_a_row_actually_has
    graph = embedder
    hidden, pooled = Torobi::Session.open(read_only(graph), weights: weights(graph)) do |s|
      s.tap("m.hidden", stat: :full).tap("m.embedding", stat: :full)
      s.evaluate(Torobi::Models::ModernBERT.batch(config, [LONG, SHORT], seq: SEQ,
                                                  pooling: :mean))
      [s.tapped.fetch("m.hidden").to_a, s.tapped.fetch("m.embedding").to_a]
    end

    dim = config.hidden_size
    states = hidden.each_slice(dim).each_slice(SEQ).to_a
    got = pooled.each_slice(dim).to_a
    [LONG, SHORT].each_with_index do |row, r|
      real = states[r].first(row.size)
      want = Array.new(dim) { |j| real.sum { |state| state[j] } / row.size }
      want.zip(got[r]).each_with_index do |(w, g), j|
        assert_in_delta w, g, 1e-6, "row #{r} dimension #{j}"
      end
    end
  end

  # The consequence, and the reason to care: a row's vector is the same
  # whatever it was batched with. The graph built for six positions and
  # the one built for three are the same weights, and the three real
  # tokens attend to the same three either way, so a pooling that counted
  # the padding is the only thing that could make these differ.
  def test_a_row_is_the_same_vector_however_far_it_was_padded
    padded = embed([SHORT], seq: SEQ).first
    exact = embed([SHORT], seq: SHORT.size).first

    padded.zip(exact).each_with_index do |(with_padding, without), j|
      assert_in_delta without, with_padding, 1e-5, "dimension #{j}"
    end
  end

  # And that it is not trivially true: the unmasked mean this replaced
  # would have divided six states by six, and the difference is large
  # rather than a rounding.
  def test_counting_the_padding_would_have_given_another_answer
    exact = embed([SHORT], seq: SHORT.size).first
    graph = embedder
    over_everything = Torobi::Session.open(read_only(graph), weights: weights(graph)) do |s|
      s.tap("m.hidden", stat: :full)
      s.evaluate(Torobi::Models::ModernBERT.batch(config, [SHORT], seq: SEQ, pooling: :mean))
      states = s.tapped.fetch("m.hidden").to_a.each_slice(config.hidden_size).to_a
      Array.new(config.hidden_size) { |j| states.sum { |state| state[j] } / SEQ }
    end

    apart = exact.zip(over_everything).map { |a, b| (a - b).abs }.max

    assert_operator apart, :>, 1e-3, "the padding would have to move the vector to be worth masking"
  end

  # A pooling that reads no weights asks for none: the boundary refuses a
  # field the graph does not read, so a batch carrying one would not run.
  def test_cls_pooling_carries_no_token_weights
    batch = Torobi::Models::ModernBERT.batch(config, [LONG, SHORT], seq: SEQ, pooling: :cls)

    refute_includes batch.keys, :tokens
    cls = embed([LONG, SHORT], pooling: :cls)

    assert_equal 2, cls.size
    assert_equal config.hidden_size, cls.first.size
  end

  def test_a_row_with_no_tokens_has_no_mean
    e = assert_raises(Torobi::ConfigError) do
      Torobi::Models::ModernBERT.batch(config, [LONG, []], seq: SEQ, pooling: :mean)
    end

    assert_match(/row 1 has no tokens/, e.message)
  end
end
