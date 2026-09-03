# frozen_string_literal: true

require_relative "test_helper"

# A batch too large to hold, trained as several that fit.
#
# The gradients of a sum are the sum of the gradients, so accumulating the
# parts and stepping once reaches where one step over the whole would have.
# What the engine will not do is guess the weighting: a loss that is a mean
# over its rows means each part carries the share of the rows it holds, and
# only the caller knows what the mean was over.
#
# It is also the half of a gradient cache that Torobi did not have: the
# rest (a forward whose values come back through a tap, and a backward
# seeded with a cotangent the caller worked out) is expressible already.
class AccumulateTest < Minitest::Test
  DIM = 4

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  def config
    model = Torobi.graph do |g|
      x = g.input :x, [nil, DIM]
      y = g.input :y, [nil, 1]
      g.output :loss, g.mse(g.linear(x, 1, name: "l"), y)
    end
    Torobi::GraphConfig.new(models: { m: model })
  end

  def weights
    { params: { "m.l.weight" => { shape: [1, DIM], data: [0.1, -0.2, 0.3, 0.0] },
                "m.l.bias" => { shape: [1], data: [0.05] } } }
  end

  def batch
    { x: Torobi::TensorData.nested([[1.0, 2.0, 0.5, -1.0], [0.0, 1.0, 1.0, 2.0]]),
      y: Torobi::TensorData.nested([[1.0], [2.0]]) }
  end

  def open(&) = Torobi::Session.open(config, weights:, optimizer: { kind: :sgd, lr: 0.1 }, &)

  # Plain SGD, so the size of the gradient is the size of the step: two
  # parts of the same batch move twice as far as one step over it. Under
  # AdamW they would not, because it normalizes, which is exactly why this
  # is checked here rather than there.
  def test_the_gradients_of_the_parts_are_summed
    from = weights.fetch(:params).fetch("m.l.weight").fetch(:data)

    once = open { |s| s.step!(batch) && s.fetch("m.l.weight").to_a }
    twice = open do |s|
      2.times { s.accumulate(batch) }
      s.apply!
      s.fetch("m.l.weight").to_a
    end

    from.each_index do |i|
      assert_in_delta 2 * (once[i] - from[i]), twice[i] - from[i], 1e-6
    end
  end

  def test_accumulating_moves_nothing_until_it_is_applied
    open do |s|
      before = s.fetch("m.l.weight").to_a
      loss = s.accumulate(batch)

      assert_predicate loss, :finite?
      assert_equal 1, s.accumulated
      assert_equal 0, s.step
      assert_equal before, s.fetch("m.l.weight").to_a

      assert_in_delta loss, s.apply!, 1e-6, "one part's loss is its own mean"
      assert_equal 1, s.step
      assert_equal 0, s.accumulated
      refute_equal before, s.fetch("m.l.weight").to_a
    end
  end

  def test_what_is_waiting_can_be_thrown_away
    open do |s|
      before = s.fetch("m.l.weight").to_a
      2.times { s.accumulate(batch) }

      assert_equal 2, s.discard
      assert_equal 0, s.accumulated
      assert_equal before, s.fetch("m.l.weight").to_a
      assert_raises(Torobi::StepError) { s.apply! }
    end
  end

  # A step from no gradients is not a step of zero; it is a caller that has
  # lost track of where it is.
  def test_applying_nothing_is_refused
    open do |s|
      e = assert_raises(Torobi::StepError) { s.apply! }

      assert_match(/nothing has been accumulated/, e.message)
    end
  end

  # Freezing moves what a gradient is for, and a checkpoint does not hold
  # what is waiting. Both say so while the caller can still choose.
  def test_freezing_and_checkpointing_refuse_while_parts_are_waiting
    Dir.mktmpdir("torobi-accumulate") do |dir|
      open do |s|
        s.accumulate(batch)

        assert_match(/accumulated/, assert_raises(Torobi::StepError) { s.freeze!("m.*") }.message)
        assert_match(/accumulated/,
                     assert_raises(Torobi::StepError) { s.checkpoint!(File.join(dir, "c")) }.message)

        # And both work once it is settled.
        s.apply!
        s.checkpoint!(File.join(dir, "c"))
        assert_equal ["m.l.weight"], s.freeze!("m.l.weight")
      end
    end
  end

  # The record follows the same rule as everything else in the window: what
  # happened is in the journal, and what was read is an observation.
  def test_the_journal_holds_the_parts_and_the_step
    io = StringIO.new
    Torobi::Session.open(config, weights:, optimizer: { kind: :sgd, lr: 0.1 }, io:) do |s|
      2.times { s.accumulate(batch) }
      s.apply!
    end
    entries = io.string.lines.drop(1).map { |line| JSON.parse(line) }
    observed = entries.select { |e| e["kind"] == "observe" }
    span = entries.find { |e| e["kind"] == "span" }

    assert_equal [1, 2], observed.map { |e| e["accumulated"] }
    assert_equal 2, span.fetch("parts")
    assert_equal 1, span.fetch("steps")
  end
end
