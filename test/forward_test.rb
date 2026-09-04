# frozen_string_literal: true

require_relative "test_helper"

# Asking a model what it produces (docs/plan.md section 15.47).
#
# The other half of a fine-tune. `evaluate` answers what the loss is,
# which is the question training asks; this answers what the model says,
# which is the question everything else asks: an embedding to index, a
# score to rank with, logits to look at.
#
# It is the same pass `evaluate` runs, stopping before the objective, so
# what is claimed here is mostly what it does *not* do: not move the run,
# not draw randomness, not ask for the labels it is not being marked
# against.
class ForwardTest < Minitest::Test
  DIM = 2

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  # Two outputs, because one is the case where nothing has to be named.
  def model
    @model ||= Torobi.graph do |g|
      x = g.input :x, [nil, DIM]
      hidden = g.linear(x, 3, name: "l")
      g.output :hidden, hidden
      g.output :logits, g.linear(hidden, 1, name: "head")
    end
  end

  # A trained run: the loss is against labels that arrive in the batch,
  # which is what makes "the batch a forward needs" a different set.
  def trained
    Torobi::GraphConfig.new(
      models: { m: model },
      objective: Torobi.objective(m: model) do |g|
        g.output :loss, g.mse(g.from_model(:m, :logits), g.from_batch(:y, [nil, 1]))
      end
    )
  end

  def weights
    { params: { "m.l.weight" => { shape: [3, DIM], data: [1.0, 0.0, 0.0, 1.0, 1.0, 1.0] },
                "m.l.bias" => { shape: [3], data: [0.0, 0.0, 0.5] },
                "m.head.weight" => { shape: [1, 3], data: [1.0, 1.0, 1.0] },
                "m.head.bias" => { shape: [1], data: [0.0] } } }
  end

  def x = { x: Torobi::TensorData.nested([[1.0, 2.0], [3.0, 4.0]]) }
  def labelled = x.merge(y: Torobi::TensorData.nested([[1.0], [2.0]]))

  def open_run(config = trained, &) = Torobi::Session.open(config, weights:, &)

  def test_a_forward_answers_every_output_by_the_name_its_model_gives_it
    produced = open_run { |s| s.forward(x) }

    assert_equal %w[m.hidden m.logits], produced.keys.sort
    assert_instance_of Torobi::TensorData, produced["m.hidden"]
    assert_equal [2, 3], produced["m.hidden"].shape
    # [[1, 2], [3, 4]] through the first layer, then summed by the head.
    assert_equal [1.0, 2.0, 3.5, 3.0, 4.0, 7.5], produced["m.hidden"].to_a
    assert_equal [6.5, 14.5], produced["m.logits"].to_a
  end

  def test_a_forward_can_be_asked_for_one_output
    produced = open_run { |s| s.forward(x, outputs: ["m.logits"]) }

    assert_equal ["m.logits"], produced.keys
  end

  # What there is to ask for, settled at open like the other names.
  def test_the_run_says_which_outputs_it_has
    names = open_run(&:output_names)

    assert_equal %w[m.hidden m.logits], names.sort
  end

  def test_an_output_no_model_declares_is_refused_with_what_there_is
    e = assert_raises(Torobi::StepError) { open_run { |s| s.forward(x, outputs: ["m.pooled"]) } }

    assert_match(/no output is named/, e.message)
    assert_match(/m.hidden/, e.message)
  end

  # `forward(x: ids)` is the batch written the way it reads, and in Ruby
  # 3 it is keywords, so the method is called with no batch at all. The
  # count Ruby reports for that says nothing about the braces that fix
  # it, and this has now cost an afternoon twice.
  def test_a_batch_written_as_keywords_is_told_about_the_braces
    e = assert_raises(ArgumentError) { open_run { |s| s.forward(x: 1) } }

    assert_match(/forward\(\{x: \.\.\.\}\)/, e.message)
    assert_match(/keywords/, e.message)
  end

  # The objective's fields are the training run's, not the model's. A
  # model being asked what it thinks is not being marked against anything.
  def test_a_forward_needs_only_what_the_models_read
    produced = open_run { |s| s.forward(x) }

    assert_equal [6.5, 14.5], produced["m.logits"].to_a
    # And the same run still refuses a step without them, so this is a
    # narrower demand rather than a lost check.
    e = assert_raises(Torobi::StepError) { open_run { |s| s.step!(x) } }

    assert_match(/missing input "y"/, e.message)
  end

  def test_a_forward_moves_nothing
    step, loss, weight = open_run do |s|
      s.step!(labelled)
      before = [s.step, s.loss, s.fetch("m.l.weight").to_a]
      s.forward(x)
      after = [s.step, s.loss, s.fetch("m.l.weight").to_a]

      assert_equal before, after, "a forward is not a step"
      after
    end

    assert_equal 1, step
    refute_nil loss
    refute_empty weight
  end

  # The random ops stand aside, as they do in `evaluate`: the answer is
  # the model's rather than one sample of it, and two forwards agree.
  def test_a_forward_draws_no_randomness
    dropped = Torobi.graph do |g|
      x = g.input :x, [nil, DIM]
      g.output :hidden, g.dropout(g.linear(x, 3, name: "l"), 0.5)
    end
    config = Torobi::GraphConfig.new(models: { m: dropped }, train: [])
    w = { params: weights.fetch(:params).slice("m.l.weight", "m.l.bias") }

    first, second = Torobi::Session.open(config, weights: w) do |s|
      [s.forward(x)["m.hidden"].to_a, s.forward(x)["m.hidden"].to_a]
    end

    assert_equal first, second
    assert_equal [1.0, 2.0, 3.5, 3.0, 4.0, 7.5], first, "dropout stood aside, so this is the model"
  end

  # A run opened to be read: nothing is trained, so nothing has to be a
  # loss, and the outputs can be whatever shape the model produces.
  def test_a_run_that_trains_nothing_needs_no_loss
    config = Torobi::GraphConfig.new(models: { m: model }, train: [])

    refute_predicate config, :loss?

    Torobi::Session.open(config, weights:) do |s|
      refute_predicate s, :loss?
      assert_equal [6.5, 14.5], s.forward(x)["m.logits"].to_a

      # And everything that would need one says so, rather than reaching
      # the engine and failing about a shape.
      %i[step! evaluate accumulate gradients].each do |what|
        e = assert_raises(Torobi::ConfigError) { s.public_send(what, x) }

        assert_match(/needs a loss/, e.message, "#{what} should say what is missing")
        assert_match(/forward/, e.message)
      end
    end
  end

  # A config that trains something still has to have one, and the refusal
  # says how to open a run that does not.
  def test_a_config_that_trains_still_needs_its_loss
    e = assert_raises(Torobi::ConfigError) { Torobi::GraphConfig.new(models: { m: model }) }

    assert_match(/must declare exactly one output/, e.message)
  end

  # A tap watches what a pass did; a forward asks for a value by name.
  # Both work on the same pass.
  def test_the_taps_report_a_forward
    config = Torobi::GraphConfig.new(models: { m: model }, train: [])

    Torobi::Session.open(config, weights:) do |s|
      s.tap("m.head", stat: :full)
      s.forward(x, outputs: ["m.hidden"])

      assert_equal [6.5, 14.5], s.tapped.fetch("m.head").to_a
    end
  end
end
