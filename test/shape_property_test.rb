# frozen_string_literal: true

require_relative "test_helper"

# Shape inference is an algebra, and these are its laws: what holds for
# every shape, not for the examples the other tests happen to write. They
# are pure Ruby, so they cost nothing and reach shapes a model description
# only sometimes uses.
class ShapePropertyTest < Minitest::Test
  # Concrete, symbolic and singleton, rank one to three, chosen so every
  # pair here meets.
  SHAPES = [[1], [3], [nil], [2, 1], [1, 3], [2, 3], [nil, 3], [4, 1]].freeze

  def builder = Torobi::DSL::Builder.new

  def input(g, name, shape) = g.input(name, shape)

  # The shape an add gives, or `:refused` when the two do not meet.
  def shape_or_refusal
    yield(builder).shape
  rescue Torobi::ConfigError
    :refused
  end

  # Broadcasting is symmetric: which side a 1 is on cannot change the
  # answer, and whether the pair meets at all is symmetric too.
  def test_broadcasting_is_commutative
    SHAPES.repeated_permutation(2) do |a, b|
      left = shape_or_refusal { |g| input(g, :a, a) + input(g, :b, b) }
      right = shape_or_refusal { |g| input(g, :b, b) + input(g, :a, a) }

      assert_equal left, right, "#{a.inspect} with #{b.inspect}"
    end
  end

  # And associative, so a chain of adds does not depend on where it is
  # parenthesized.
  def test_broadcasting_is_associative
    SHAPES.repeated_permutation(3) do |a, b, c|
      left = shape_or_refusal { |g| (input(g, :a, a) + input(g, :b, b)) + input(g, :c, c) }
      right = shape_or_refusal { |g| input(g, :a, a) + (input(g, :b, b) + input(g, :c, c)) }
      next if left == :refused || right == :refused # a triple that does not meet

      assert_equal left, right, "#{a.inspect}, #{b.inspect}, #{c.inspect}"
    end
  end

  def test_a_reshape_to_one_dimension_and_back_is_the_original
    [[6], [2, 3], [1, 6], [3, 2]].each do |shape|
      g = builder
      flat = input(g, :x, shape).reshape(shape: [-1])

      assert_equal [shape.inject(:*)], flat.shape
      assert_equal shape, flat.reshape(shape: shape).shape
    end
  end

  def test_splitting_heads_and_merging_them_back_is_the_original_shape
    [[1, 4, 8], [2, 5, 12]].each do |batch, seq, width|
      g = builder
      x = input(g, :x, [batch, seq, width])

      assert_equal [batch, seq, width], x.split_heads(4).merge_heads.shape
    end
  end

  def test_a_reduction_keeps_the_rank_or_drops_it
    g = builder
    x = input(g, :x, [2, 3, 4])

    assert_equal [], g.mean(x).shape
    assert_equal [1, 1, 1], g.mean(x, keepdims: true).shape
    assert_equal [2, 4], g.mean(x, axes: [1]).shape
    assert_equal [2, 1, 4], g.mean(x, axes: [1], keepdims: true).shape
  end

  def test_a_singleton_dimension_stretches_to_the_other_side
    g = builder

    assert_equal [3], (input(g, :a, [1]) + input(g, :b, [3])).shape
    assert_equal [2, 3], (input(g, :a, [2, 1]) + input(g, :b, [1, 3])).shape
    assert_equal [4, 3], (input(g, :a, [nil, 3]) + input(g, :b, [4, 1])).shape
  end

  def test_a_symbolic_dimension_takes_the_concrete_one_it_meets
    g = builder

    assert_equal [5, 3], (input(g, :a, [nil, 3]) + input(g, :b, [5, 3])).shape
    assert_equal [5, 3], (input(g, :a, [nil, 3]) + input(g, :b, [5, 1])).shape
  end

  def test_matmul_contracts_the_inner_dimension_and_keeps_the_batch
    g = builder

    assert_equal [2, 5], g.matmul(input(g, :a, [2, 3]), input(g, :b, [3, 5])).shape
    assert_equal [7, 2, 5], g.matmul(input(g, :a, [7, 2, 3]), input(g, :b, [3, 5])).shape
    assert_equal [7, 2, 5], g.matmul(input(g, :a, [7, 2, 3]), input(g, :b, [7, 3, 5])).shape
  end
end
