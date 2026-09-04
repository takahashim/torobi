# frozen_string_literal: true

require_relative "test_helper"

# One model, applied to two sets of rows at two lengths
# (docs/plan.md 15.63).
#
# Two things had to be added and they only pay off together. A reshape
# can keep a dimension it does not know (`0`), which is what lets a graph
# be built for no particular sequence: the head split was the only place
# a concrete one was needed. And `g.sharing` makes a second application
# read the first's parameters rather than declare its own.
#
# What is claimed here is that the sharing is real. Not "two towers were
# built" but: there is one set of weights, one gradient reaches it from
# both sides, and a side embeds the same numbers it would have on its
# own.
class SharingTest < Minitest::Test
  SEQ = 6
  ROWS = [[3, 8, 5, 9, 2, 7], [4, 6, 1]].freeze
  SHORT = [[4, 6, 1], [7, 2]].freeze

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

  def towers(sides = { queries: nil, documents: nil })
    Torobi::Models::ModernBERT.towers(config, sides, normalize: false)
  end

  def alone(seq: nil)
    Torobi::Models::ModernBERT.embedder(config, seq:, pooling: :mean, normalize: false)
  end

  # The same numbers whichever graph asks for them, so a difference in an
  # embedding is a difference in what the graph did.
  def weights(graph)
    rng = Random.new(11)
    params = graph.parameters.to_h do |spec|
      ["m.#{spec.path}",
       { shape: spec.shape,
         data: Array.new(spec.shape.reduce(1, :*)) { rng.rand(-0.4..0.4) } }]
    end
    { params: }
  end

  def read_only(graph) = Torobi::GraphConfig.new(models: { m: graph }, train: [])

  def batch(rows, seq:, fields: "")
    Torobi::Models::ModernBERT.batch(config, rows, seq:, pooling: :mean, fields:)
  end

  def vectors(produced, name)
    produced.fetch(name).to_a.each_slice(config.hidden_size).to_a
  end

  def close(a, b, margin: 2e-5, message: nil)
    a.flatten.zip(b.flatten).each { |x, y| assert_in_delta x, y, margin, message }
  end

  # --- one set of weights ---

  def test_a_second_tower_declares_no_parameters_of_its_own
    two = towers
    one = alone

    assert_equal one.parameters.map(&:path), two.parameters.map(&:path)
    assert_equal 0, two.parameters.map(&:path).tally.count { |_, n| n > 1 },
                 "a shared parameter is declared once and read twice"
  end

  # The sides are told apart by their names, not by having their own
  # weights: two values computed from different rows are two values.
  def test_each_side_gets_its_own_inputs_and_output
    two = towers

    assert_equal %w[documents.embedding queries.embedding], two.outputs.keys.sort
    assert_includes two.inputs.map(&:name), "queries.input_ids"
    assert_includes two.inputs.map(&:name), "documents.input_ids"
  end

  def test_declaring_a_shared_parameter_differently_is_refused
    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph do |g|
        x = g.input(:x, [nil, 4])
        g.sharing { g.linear(x, 4, name: "w") }
        g.sharing("again") { g.linear(x, 5, name: "w") }
      end
    end

    assert_match(/"w.weight" is declared already/, e.message)
    assert_match(/shape \[4, 4\] vs \[5, 4\]/, e.message)
    assert_match(/have to be the same model/, e.message)
  end

  # Without sharing it is two parameters, which is what it has always
  # been: the block is what changes the meaning, not the repetition.
  def test_the_same_path_twice_outside_a_sharing_block_is_still_two
    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph do |g|
        x = g.input(:x, [nil, 4])
        g.linear(x, 4, name: "w")
        g.linear(x, 4, name: "w")
      end
    end

    assert_match(/named "w"/, e.message)
  end

  # --- the numbers ---

  # A side of a shared graph is the model it would have been on its own.
  def test_a_tower_embeds_what_the_model_embeds_alone
    two = towers
    one = alone

    shared = Torobi::Session.open(read_only(two), weights: weights(two)) do |s|
      vectors(s.forward(batch(ROWS, seq: SEQ, fields: "queries.")
                        .merge(batch(SHORT, seq: 3, fields: "documents."))),
              "m.queries.embedding")
    end
    apart = Torobi::Session.open(read_only(one), weights: weights(one)) do |s|
      vectors(s.forward(batch(ROWS, seq: SEQ)), "m.embedding")
    end

    close(shared, apart)
  end

  # The point of the exercise: the two sides are at different lengths in
  # one pass, and the short side is not padded out to the long one.
  def test_the_sides_are_embedded_at_their_own_lengths
    two = towers

    short, long = Torobi::Session.open(read_only(two), weights: weights(two)) do |s|
      produced = s.forward(batch(ROWS, seq: SEQ, fields: "queries.")
                           .merge(batch(SHORT, seq: 3, fields: "documents.")))
      [vectors(produced, "m.documents.embedding"), vectors(produced, "m.queries.embedding")]
    end

    assert_equal 2, short.size
    assert_equal 2, long.size
    # And the short side is what it is at any other padding: a graph
    # built for no length still weighs the padding out of the mean.
    padded = Torobi::Session.open(read_only(two), weights: weights(two)) do |s|
      produced = s.forward(batch(ROWS, seq: SEQ, fields: "queries.")
                           .merge(batch(SHORT, seq: SEQ, fields: "documents.")))
      vectors(produced, "m.documents.embedding")
    end

    close(short, padded, message: "padding took part in the mean")
  end

  # --- one gradient ---

  # The sharpest claim. Both sides read the same weights, so a loss that
  # reads both sends both gradients to one place: with the same rows on
  # each side that is exactly twice what one side sends. Two parameter
  # sets that merely held equal numbers would give one.
  def test_a_gradient_from_both_sides_lands_on_the_one_parameter
    doubled = gradient_of(towers, %w[queries documents])
    single = gradient_of(alone, %w[])

    assert_equal single.keys.sort, doubled.keys.sort
    single.each do |path, value|
      close(doubled.fetch(path), value.map { |v| v * 2 }, margin: 1e-4,
            message: "#{path} did not receive both sides")
    end
  end

  # The sum of every embedding a graph produces, differentiated. `sides`
  # names the outputs to add up; empty means the lone model's own.
  def gradient_of(graph, sides)
    objective = Torobi.objective(m: graph) do |g|
      outputs = sides.empty? ? [:embedding] : sides.map { |side| :"#{side}.embedding" }
      total = outputs
              .map { |name| g.sum(g.from_model(:m, name)) }
              .reduce { |a, b| a + b }
      g.output :loss, total
    end
    config = Torobi::GraphConfig.new(models: { m: graph }, objective:, train: [:m])
    fields = if sides.empty?
               batch(ROWS, seq: SEQ)
             else
               sides.map { |side| batch(ROWS, seq: SEQ, fields: "#{side}.") }.reduce(:merge)
             end
    Torobi::Session.open(config, weights: weights(graph)) do |s|
      s.gradients(fields).transform_values(&:to_a)
    end
  end
end
