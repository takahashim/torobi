# frozen_string_literal: true

require_relative "test_helper"

# The four things a decoder needs that an encoder did not
# (docs/plan.md section 15.48).
#
# None of them is a decoder. They are the gaps that stood between the
# vocabulary this had and the one a language model is written in: the
# largest of a row, the loss over a vocabulary, attention whose key heads
# are fewer than its query heads, and one parameter read twice.
class DecoderOpsTest < Minitest::Test
  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  # A run opened to be read, which is what most of this file wants.
  def reading(graph, weights: { params: {} }, &)
    Torobi::Session.open(Torobi::GraphConfig.new(models: { m: graph }, train: []),
                         weights:, &)
  end

  def flat(shape, data) = Torobi::TensorData.from_a(shape, data)

  # --- the largest of a row ---

  def test_max_reduces_like_the_others
    graph = Torobi.graph do |g|
      x = g.input :x, [nil, 3]
      g.output :rows, g.max(x, axes: [-1])
      g.output :all, g.max(x)
      g.output :kept, g.max(x, axes: [-1], keepdims: true)
    end

    produced = reading(graph) do |s|
      s.forward({ x: flat([2, 3], [1.0, -5.0, 2.0, 0.5, 0.25, 0.125]) })
    end

    assert_equal [2.0, 0.5], produced["m.rows"].to_a
    assert_equal [2.0], produced["m.all"].to_a
    assert_equal [2, 1], produced["m.kept"].shape
  end

  # --- the loss over a vocabulary ---

  # Three classes, two positions, worked out by hand: the loss at a
  # position is log(sum(exp(logits))) minus the logit the target names.
  def test_cross_entropy_is_the_loss_at_each_position
    graph = Torobi.graph do |g|
      logits = g.input :logits, [nil, 3]
      targets = g.input :targets, [nil], dtype: :i32
      g.output :loss, g.cross_entropy(logits, targets)
    end

    got = reading(graph) do |s|
      s.forward({ logits: flat([2, 3], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]),
                  targets: Torobi::TensorData.from_a([2], [2, 0], dtype: :i32) })
    end.fetch("m.loss")

    want = [Math.log([1.0, 2.0, 3.0].sum { |v| Math.exp(v) }) - 3.0, Math.log(3.0)]

    assert_equal [2], got.shape, "one number per position, and no reduction"
    got.to_a.zip(want).each { |a, b| assert_in_delta b, a, 1e-6 }
  end

  # The reason it is one op rather than a handful. A vocabulary's logits
  # are not small, and `log(sum(exp(x)))` written out overflows f32 well
  # before they are; taking the largest out first is what this op is.
  def test_cross_entropy_survives_logits_that_would_overflow
    graph = Torobi.graph do |g|
      logits = g.input :logits, [nil, 2]
      targets = g.input :targets, [nil], dtype: :i32
      g.output :loss, g.cross_entropy(logits, targets)
      # The same thing written out, which is what it would have been.
      g.output :naive, g.sum(logits.exp, axes: [-1]).log
    end

    produced = reading(graph) do |s|
      s.forward({ logits: flat([1, 2], [1.0e4, 0.0]),
                  targets: Torobi::TensorData.from_a([1], [0], dtype: :i32) })
    end

    assert_in_delta 0.0, produced["m.loss"].to_a.first, 1e-6
    assert_predicate produced["m.naive"].to_a.first, :infinite?,
                     "the written-out form is what this op exists instead of"
  end

  # The gradient a language model is trained by: the softmax, minus one
  # at the class that should have been picked.
  def test_the_gradient_of_a_cross_entropy_is_softmax_minus_one
    graph = Torobi.graph do |g|
      targets = g.input :targets, [nil], dtype: :i32
      # The logits are the parameter, so the gradient this asks for is
      # the loss's own by its scores, with nothing in between.
      logits = g.param("logits", [1, 3], init: { "type" => "zeros" })
      g.output :loss, g.sum(g.cross_entropy(logits, targets))
    end

    got = Torobi::Session.open(
      Torobi::GraphConfig.new(models: { m: graph }),
      weights: { params: { "m.logits" => { shape: [1, 3], data: [1.0, 2.0, 3.0] } } }
    ) do |s|
      s.gradients(targets: Torobi::TensorData.from_a([1], [2], dtype: :i32))
    end.fetch("m.logits").to_a

    total = [1.0, 2.0, 3.0].sum { |v| Math.exp(v) }
    want = [1.0, 2.0, 3.0].map { |v| Math.exp(v) / total }
    want[2] -= 1.0

    got.zip(want).each_with_index { |(a, b), i| assert_in_delta b, a, 1e-6, "class #{i}" }
  end

  # --- grouped-query attention ---

  # Four query heads over two key heads is the arrangement every decoder
  # since Llama 2 has, and the claim is that it means what tiling the keys
  # would have meant: the first two query heads share the first key head.
  def test_grouped_query_attention_is_the_tiled_one_without_the_tiling
    grouped = attention(kv_heads: 2)
    tiled = attention(kv_heads: 4)
    q = Array.new(4 * 2 * 3) { |i| ((i % 7) - 3) * 0.1 }
    two = Array.new(2 * 2 * 3) { |i| ((i % 5) - 2) * 0.2 }
    # Query heads 0 and 1 read key head 0, and 2 and 3 read key head 1:
    # each key head repeated in place, not the pair repeated.
    four = two.each_slice(2 * 3).flat_map { |head| [head, head] }.flatten

    got = reading(grouped) do |s|
      s.forward({ q: flat([1, 4, 2, 3], q), k: flat([1, 2, 2, 3], two),
                  v: flat([1, 2, 2, 3], two) })
    end.fetch("m.attended")
    want = reading(tiled) do |s|
      s.forward({ q: flat([1, 4, 2, 3], q), k: flat([1, 4, 2, 3], four),
                  v: flat([1, 4, 2, 3], four) })
    end.fetch("m.attended")

    assert_equal [1, 4, 2, 3], got.shape
    got.to_a.zip(want.to_a).each_with_index do |(a, b), i|
      assert_in_delta b, a, 1e-6, "element #{i}"
    end
  end

  def attention(kv_heads:)
    Torobi.graph do |g|
      q = g.input :q, [nil, 4, 2, 3]
      k = g.input :k, [nil, kv_heads, 2, 3]
      v = g.input :v, [nil, kv_heads, 2, 3]
      g.output :attended, g.sdpa(q, k, v)
    end
  end

  def test_query_heads_that_do_not_divide_into_the_key_heads_are_refused
    e = assert_raises(Torobi::ConfigError) { attention(kv_heads: 3) }

    assert_match(/4 query heads do not divide into 3 key heads/, e.message)
  end

  # --- the causal mask ---

  # Every position attends to itself and what came before it, and to
  # nothing after. With equal queries and keys the scores are equal, so
  # each position is the average of what it may see, and the averages say
  # exactly how far each one saw.
  def test_a_causal_attention_sees_backwards_only
    graph = Torobi.graph do |g|
      q = g.input :q, [nil, 1, 3, 2]
      v = g.input :v, [nil, 1, 3, 2]
      g.output :attended, g.sdpa(q, q, v, causal: true)
    end

    got = reading(graph) do |s|
      s.forward({ q: flat([1, 1, 3, 2], Array.new(6, 0.0)),
                  v: flat([1, 1, 3, 2], [1.0, 0.0, 0.0, 3.0, 0.0, 0.0]) })
    end.fetch("m.attended").to_a

    want = [1.0, 0.0,                      # only itself
            0.5, 1.5,                      # itself and the one before
            1.0 / 3, 1.0]                  # all three

    got.zip(want).each_with_index { |(a, b), i| assert_in_delta b, a, 1e-6, "element #{i}" }
  end

  def test_a_causal_attention_does_not_take_another_mask_as_well
    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph do |g|
        q = g.input :q, [nil, 1, 3, 2]
        mask = g.input :mask, [nil, 1, 3, 3]
        g.sdpa(q, q, q, mask:, causal: true)
      end
    end

    assert_match(/causal: is a mask/, e.message)
  end

  # --- one parameter, read twice ---

  # Weight tying. The output projection is the embedding table
  # transposed, which is one parameter used at both ends of the model.
  #
  # The claim is exact: the gradient the tied table gets is the gradient
  # the embedding would have got plus the gradient the output projection
  # would have got, which is what "the same parameter, read twice" means
  # and is not the same as two parameters that happen to start equal.
  VOCAB = 4
  WIDE = 2

  def tied
    Torobi.graph do |g|
      h = language_model(g)
      logits = g.matmul(h, g.parameter("embed.weight").transpose(axes: [1, 0]))
      g.output :loss, g.mean(g.cross_entropy(logits, g.input(:targets, [nil, 2], dtype: :i32)))
    end
  end

  def untied
    Torobi.graph do |g|
      h = language_model(g)
      # The same numbers, transposed, as a parameter of its own.
      logits = g.matmul(h, g.param("head", [WIDE, VOCAB], init: { "type" => "zeros" }))
      g.output :loss, g.mean(g.cross_entropy(logits, g.input(:targets, [nil, 2], dtype: :i32)))
    end
  end

  def language_model(g)
    g.embedding(g.input(:ids, [nil, 2], dtype: :i32), vocab: VOCAB, dim: WIDE, name: "embed")
  end

  def table = Array.new(VOCAB * WIDE) { |i| ((i % 3) + 1) * 0.1 }

  def gradient_of(graph, params)
    Torobi::Session.open(Torobi::GraphConfig.new(models: { m: graph }), weights: { params: }) do |s|
      s.gradients(ids: Torobi::TensorData.from_a([1, 2], [0, 1], dtype: :i32),
                  targets: Torobi::TensorData.from_a([1, 2], [1, 2], dtype: :i32))
    end
  end

  def test_a_tied_embedding_is_one_parameter_that_gets_both_gradients
    assert_equal ["embed.weight"], tied.parameters.map(&:path), "one parameter, not two"

    embedding = { shape: [VOCAB, WIDE], data: table }
    both = gradient_of(tied, { "m.embed.weight" => embedding })
    apart = gradient_of(untied, { "m.embed.weight" => embedding,
                                  "m.head" => { shape: [WIDE, VOCAB],
                                                data: transposed(table) } })

    assert_equal ["m.embed.weight"], both.keys
    want = both.fetch("m.embed.weight").to_a
    got = apart.fetch("m.embed.weight").to_a
               .zip(transposed(apart.fetch("m.head").to_a, from: [WIDE, VOCAB]))
               .map(&:sum)

    want.zip(got).each_with_index do |(a, b), i|
      assert_in_delta a, b, 1e-6, "element #{i}"
    end
    refute(want.all? { |v| v.abs < 1e-9 }, "and the gradient is not simply zero")
  end

  # [rows, columns] to [columns, rows], flat both ways.
  def transposed(flat, from: [VOCAB, WIDE])
    rows, columns = from
    Array.new(columns * rows) { |i| flat[((i % rows) * columns) + (i / rows)] }
  end

  def test_a_shared_parameter_that_was_never_declared_is_refused
    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph { |g| g.parameter("nothing.weight") }
    end

    assert_match(/no parameter "nothing.weight" is declared yet/, e.message)
  end
end
