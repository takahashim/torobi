# frozen_string_literal: true

require_relative "test_helper"

# The laws the device is held to, on the smallest graph that shows them: a
# softmax is a distribution over its axis, and broadcasting adds a row to
# every row. The finite differences elsewhere check gradients; these check
# that the values mean what the op says.
class MathPropertyTest < Minitest::Test
  ROWS = [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]].freeze

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  def config
    model = Torobi.graph do |g|
      x = g.input :x, [nil, 4]
      y = g.input :y, [nil, 4]
      g.output :row_sums, g.sum(x.softmax(axis: -1), axes: [-1], keepdims: true)
      g.output :added, x + y
    end
    Torobi::GraphConfig.new(models: { "m" => model }, train: [])
  end

  def forward(batch)
    Torobi::Session.open(config, weights: { params: {} }) { |s| s.forward(batch) }
  end

  def test_a_softmax_is_a_distribution_over_its_axis
    sums = forward({ x: Torobi::TensorData.nested(ROWS),
                     y: Torobi::TensorData.nested([[0.0, 0.0, 0.0, 0.0]]) })["m.row_sums"]

    assert_equal [ROWS.size, 1], sums.shape
    sums.to_a.each { |sum| assert_in_delta 1.0, sum, 1e-6 }
  end

  def test_broadcasting_adds_a_row_to_every_row
    row = [0.5, 0.5, 0.5, 0.5]
    added = forward({ x: Torobi::TensorData.nested(ROWS),
                      y: Torobi::TensorData.nested([row]) })["m.added"]

    assert_equal [ROWS.size, 4], added.shape
    expected = ROWS.flat_map { |r| r.zip(row).map { |a, b| a + b } }

    added.to_a.zip(expected).each { |got, want| assert_in_delta want, got, 1e-6 }
  end
end
