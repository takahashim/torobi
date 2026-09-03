# frozen_string_literal: true

require_relative "test_helper"

# The loss differentiated by what it was given, rather than by what it
# holds (docs/plan.md section 15.36).
#
# A gradient cache is defined in terms of this: encode the parts without
# gradients, work out the loss over all of the representations at once,
# then re-run each part with the loss's gradient by those representations
# as its seed. The middle step is this one.
class FieldGradientsTest < Minitest::Test
  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  # loss = mean((a - b)^2) over four numbers.
  def config
    model = Torobi.graph do |g|
      a = g.input :a, [nil, 2]
      b = g.input :b, [nil, 2]
      # A parameter the loss does not use, so it is visible that nothing
      # differentiates it here.
      unused = g.param("unused", [1], init: { "type" => "zeros" })
      g.output :loss, g.mse(a, b) + g.sum(unused * 0.0)
    end
    Torobi::GraphConfig.new(models: { m: model })
  end

  def weights = { params: { "m.unused" => { shape: [1], data: [0.0] } } }

  def batch
    { a: Torobi::TensorData.nested([[1.0, 2.0]]),
      b: Torobi::TensorData.nested([[0.0, 0.5]]) }
  end

  def test_the_gradient_by_a_field_is_the_arithmetic
    Torobi::Session.open(config, weights:) do |s|
      grads = s.field_gradients(batch, of: %i[a b])

      # d/da mean((a - b)^2) = 2(a - b)/n, with n = 2.
      assert_equal %w[a b], grads.keys
      assert_equal [1, 2], grads["a"].shape
      grads["a"].to_a.zip([1.0, 1.5]).each { |got, want| assert_in_delta want, got, 1e-6 }
      # And by b it is the same with the sign turned round.
      grads["b"].to_a.zip([-1.0, -1.5]).each { |got, want| assert_in_delta want, got, 1e-6 }
    end
  end

  def test_it_moves_nothing
    Torobi::Session.open(config, weights:) do |s|
      before = s.fetch("m.unused").to_a
      s.field_gradients(batch, of: [:a])

      assert_equal 0, s.step
      assert_equal before, s.fetch("m.unused").to_a
    end
  end

  def test_a_field_the_batch_does_not_have_is_refused
    Torobi::Session.open(config, weights:) do |s|
      e = assert_raises(Torobi::StepError) { s.field_gradients(batch, of: [:elsewhere]) }

      assert_match(/elsewhere/, e.message)
      assert_match(/"a"/, e.message)
    end
  end

  # The property a gradient cache rests on: what the loss says about a
  # representation does not depend on how many other things were in the
  # batch when it was worked out. Two rows on their own and the same two
  # rows inside four give the same gradient for those two, once the mean's
  # weighting is accounted for.
  def test_the_gradient_by_a_field_is_local_to_its_rows
    pair = { a: Torobi::TensorData.nested([[1.0, 2.0], [0.0, 1.0]]),
             b: Torobi::TensorData.nested([[0.0, 0.5], [1.0, 1.0]]) }
    four = { a: Torobi::TensorData.nested([[1.0, 2.0], [0.0, 1.0], [3.0, 3.0], [1.0, 1.0]]),
             b: Torobi::TensorData.nested([[0.0, 0.5], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0]]) }

    Torobi::Session.open(config, weights:) do |s|
      of_pair = s.field_gradients(pair, of: [:a])["a"].to_a
      of_four = s.field_gradients(four, of: [:a])["a"].to_a.first(4)

      # The loss is a mean over twice as many numbers, so each row's share
      # is half. Nothing else about the other rows reaches these.
      of_pair.zip(of_four).each { |alone, among| assert_in_delta alone / 2, among, 1e-6 }
    end
  end
end
