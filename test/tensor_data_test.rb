# frozen_string_literal: true

require_relative "test_helper"

# TensorData is what crosses the boundary: bytes, a shape and a dtype. Its
# constructors are several ways to write the same value, so what holds is
# that they agree, and that a value whose bytes do not fill its shape is
# refused here rather than sent.
class TensorDataTest < Minitest::Test
  def setup
    skip "extension not compiled" unless defined?(Torobi::TensorData)
  end

  def test_runs_write_the_same_bytes_as_a_flat_array
    runs = Torobi::TensorData.runs([2, 3], [[3, 0.0], [3, -1.0]])
    flat = Torobi::TensorData.from_a([2, 3], [0.0, 0.0, 0.0, -1.0, -1.0, -1.0])

    assert_equal flat, runs
    assert_equal flat.bytes, runs.bytes
  end

  def test_filled_is_a_run_of_one_value
    assert_equal Torobi::TensorData.from_a([2, 2], [7.0] * 4),
                 Torobi::TensorData.filled([2, 2], 7.0)
  end

  def test_nested_takes_its_shape_from_the_nesting
    t = Torobi::TensorData.nested([[1.0, 2.0], [3.0, 4.0]])

    assert_equal [2, 2], t.shape
    assert_equal [1.0, 2.0, 3.0, 4.0], t.to_a
  end

  def test_the_bytes_are_the_shape_times_the_width
    t = Torobi::TensorData.from_a([2, 3, 4], Array.new(24, 1.0))

    assert_equal 24, t.size
    assert_equal 4, t.value_size
    assert_equal 96, t.bytesize
  end

  def test_i32_crosses_as_four_bytes_a_value
    t = Torobi::TensorData.from_a([3], [1, -2, 300], dtype: :i32)

    assert_equal 12, t.bytesize
    assert_equal [1, -2, 300], t.to_a
  end

  def test_already_packed_bytes_pass_through
    packed = [1.5, -2.5].pack("f*")
    t = Torobi::TensorData.from({ shape: [2], data: packed })

    assert_equal packed, t.bytes
    assert_equal [1.5, -2.5], t.to_a
  end

  # A value is its shape, its dtype and its bytes, and nothing else is
  # equal to it.
  def test_equality_is_shape_dtype_and_bytes
    a = Torobi::TensorData.from_a([2], [1.0, 2.0])
    b = Torobi::TensorData.from_a([2], [1.0, 2.0])
    c = Torobi::TensorData.from_a([2], [1.0, 2.5])

    assert_equal a, b
    assert_equal a.hash, b.hash
    refute_equal a, c
    refute_equal a, nil
    refute_equal a, 3
  end

  def test_a_dtype_that_does_not_cross_is_refused
    e = assert_raises(ArgumentError) { Torobi::TensorData.format(:f64) }

    assert_match(/does not cross the boundary/, e.message)
  end

  def test_a_value_that_is_not_a_tensor_or_a_hash_is_refused
    e = assert_raises(ArgumentError) { Torobi::TensorData.from(42) }

    assert_match(/a tensor is a TensorData/, e.message)
  end

  def test_runs_that_do_not_cover_the_shape_are_refused
    e = assert_raises(ArgumentError) { Torobi::TensorData.runs([2], [[1, 0.0]]) }

    assert_match(/holds 2 values, the runs give 1/, e.message)
  end
end
