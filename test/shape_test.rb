# frozen_string_literal: true

require_relative "test_helper"

# Shape inference is where a mistake in a model description is caught, at
# build time and with the node named (docs/plan.md, M0). Each rule refuses
# in its own words, and this holds every refusal to a test: a rule that
# started accepting what it used to refuse would otherwise pass unnoticed.
class ShapeTest < Minitest::Test
  # Each case is built inside a graph, because that is where a caller meets
  # it; the fragment is what the refusal must say. The shapes are the
  # smallest that reach each branch.
  REFUSALS = {
    "matmul needs rank >= 2" => ->(g) { g.matmul(g.input(:a, [3]), g.input(:b, [3])) },
    "rank mismatch" => ->(g) { g.matmul(g.input(:a, [2, 3, 4]), g.input(:b, [5, 6, 4, 7])) },
    "mixed dtypes" => ->(g) { g.input(:a, [2]) * g.input(:b, [2], dtype: :bf16) },
    "axis 5 is out of range for rank 2" => ->(g) { g.sum(g.input(:a, [2, 3]), axes: [5]) },
    "at most one -1" => ->(g) { g.input(:a, [6]).reshape(shape: [-2]) },
    "only one dimension may be -1" => ->(g) { g.input(:a, [6]).reshape(shape: [-1, -1]) },
    "must name a -1 or a 0" => ->(g) { g.input(:a, [nil, 6]).reshape(shape: [2, 3]) },
    "wants 4" => ->(g) { g.input(:a, [nil, 6]).reshape(shape: [-1, 4]) },
    "does not divide into" => ->(g) { g.input(:a, [5]).reshape(shape: [-1, 2]) },
    "is not a permutation" => ->(g) { g.input(:a, [2, 3]).transpose(axes: [0, 0]) },
    "is out of 0...4" => ->(g) { g.input(:a, [4]).slice(axis: 0, start: 3, length: 2) },
    "duplicate reduction axes" => ->(g) { g.mean(g.input(:a, [2, 3]), axes: [0, 0]) },
    "take table needs rank >= 2" => lambda { |g|
      g.emit("take", inputs: [g.input(:t, [4]), g.input(:i, [1], dtype: :i32)])
    },
    "attention takes" => ->(g) { g.sdpa(g.input(:q, [2]), g.input(:k, [2]), g.input(:v, [2])) },
    "logits need a class axis" =>
      ->(g) { g.cross_entropy(g.input(:l, [3]), g.input(:t, [1], dtype: :i32)) },
    "targets are class indices, so i32" =>
      ->(g) { g.cross_entropy(g.input(:l, [2, 3]), g.input(:t, [2])) }
  }.freeze

  def test_every_shape_refusal_says_what_it_refused
    REFUSALS.each do |fragment, build|
      e = assert_raises(Torobi::ConfigError, "expected a refusal for #{fragment}") do
        Torobi.graph { |g| g.output :out, build.call(g) }
      end
      assert_match(/#{Regexp.escape(fragment)}/, e.message)
    end
  end

  # The rules are named once, in `Shape::RULES`, and `ops.yml` names the
  # same set; an emission asking for one that is not there is the two files
  # disagreeing, which is worth hearing about rather than a NoMethodError.
  def test_a_rule_that_does_not_exist_is_refused_by_name
    e = assert_raises(Torobi::ConfigError) do
      Torobi::Shape.infer(:improvise, inputs: [], params: [], attrs: {}, where: "t")
    end

    assert_match(/no shape rule :improvise/, e.message)
  end
end
