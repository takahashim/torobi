# frozen_string_literal: true

require_relative "test_helper"
require "tmpdir"

# What an external review found representable and should not be. Each case
# here ended in the engine guessing, or in a panic that no Ruby could
# rescue; each is now refused where it is written.
class ContractTest < Minitest::Test
  def model
    Torobi.graph do |g|
      x = g.input :x, [nil, 2]
      g.output :logits, g.linear(x, 1, name: "l")
    end
  end

  def scalar_model
    Torobi.graph do |g|
      x = g.input :x, [nil, 2]
      g.output :loss, g.mean(g.linear(x, 1, name: "l"))
    end
  end

  # An objective with parameters of its own reached the interpreter with an
  # empty parameter slice and indexed past its end, aborting the process.
  def test_an_objective_owns_no_parameters
    m = model
    objective = Torobi.objective(m: m) do |g|
      g.output :loss, g.mean(g.linear(g.from_model(:m, :logits), 1, name: "extra"))
    end
    e = assert_raises(Torobi::ConfigError) do
      Torobi::GraphConfig.new(models: { "m" => m }, objective:)
    end
    assert_match(/objective has no parameters of its own/, e.message)
    assert_match(/extra\.weight/, e.message)
  end

  # The engine differentiates one value, so there is one output, and its
  # name is not a matter of luck: two outputs used to mean "whichever sorts
  # first".
  def test_an_objective_declares_exactly_one_output_named_loss
    m = model
    two = Torobi.objective(m: m) do |g|
      s = g.from_model(:m, :logits)
      g.output(:zzz, g.mean(s))
      g.output(:aaa, g.mean(s * 2.0))
    end
    e = assert_raises(Torobi::ConfigError) { Torobi::GraphConfig.new(models: { "m" => m }, objective: two) }
    assert_match(/exactly one output, named "loss"/, e.message)

    misnamed = Torobi.objective(m: m) { |g| g.output(:cost, g.mean(g.from_model(:m, :logits))) }
    e = assert_raises(Torobi::ConfigError) do
      Torobi::GraphConfig.new(models: { "m" => m }, objective: misnamed)
    end
    assert_match(/declares "cost"/, e.message)
  end

  def test_the_loss_is_a_scalar
    m = model
    vector = Torobi.objective(m: m) { |g| g.output(:loss, g.from_model(:m, :logits)) }
    e = assert_raises(Torobi::ConfigError) do
      Torobi::GraphConfig.new(models: { "m" => m }, objective: vector)
    end
    assert_match(/loss must be a scalar/, e.message)
    assert_match(/reduce it/, e.message)
  end

  # Without an objective the model's own output is the loss, so the same
  # demands apply, and more than one model needs an objective to choose.
  def test_a_config_without_an_objective_still_names_one_scalar_loss
    e = assert_raises(Torobi::ConfigError) { Torobi::GraphConfig.new(models: { "m" => model }) }
    assert_match(/must be an f32 scalar/, e.message)

    e = assert_raises(Torobi::ConfigError) do
      Torobi::GraphConfig.new(models: { "a" => scalar_model, "b" => scalar_model })
    end
    assert_match(/an objective says which loss to train/, e.message)
  end

  # A checkpoint missing its optimizer state used to restore "successfully"
  # and panic on the next step, with nothing for Ruby to rescue.
  def test_a_checkpoint_without_the_slots_its_optimizer_needs_is_refused
    skip "extension not compiled" unless defined?(Torobi::Session)
    config = Torobi::GraphConfig.new(models: { "m" => scalar_model })
    weights = { params: { "m.l.weight" => { shape: [1, 2], data: [0.1, 0.2] },
                          "m.l.bias" => { shape: [1], data: [0.0] } } }
    batch = { x: { shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0] } }
    adamw = { kind: :adamw, lr: 0.01 }

    Dir.mktmpdir("torobi-contract") do |dir|
      path = File.join(dir, "c")
      Torobi::Session.open(config, weights: weights, optimizer: adamw) do |s|
        s.step!(batch)
        s.checkpoint!(path)
      end
      File.unlink(File.join(path, "optimizer.safetensors"))

      e = assert_raises(Torobi::StepError) do
        Torobi::Session.open(config, weights: weights, optimizer: adamw) { |s| s.restore(path) }
      end
      assert_match(/no optimizer state, and adamw needs it/, e.message)

      # And an SGD checkpoint is not an AdamW one, in either direction.
      sgd_path = File.join(dir, "sgd")
      Torobi::Session.open(config, weights: weights, optimizer: { kind: :sgd, lr: 0.1 }) do |s|
        s.step!(batch)
        s.checkpoint!(sgd_path)
      end
      e = assert_raises(Torobi::StepError) do
        Torobi::Session.open(config, weights: weights, optimizer: adamw) { |s| s.restore(sgd_path) }
      end
      assert_match(/different optimizer/, e.message)
    end
  end

  # A refused restore must leave the session as it was, not half replaced.
  def test_a_refused_restore_changes_nothing
    skip "extension not compiled" unless defined?(Torobi::Session)
    config = Torobi::GraphConfig.new(models: { "m" => scalar_model })
    weights = { params: { "m.l.weight" => { shape: [1, 2], data: [0.1, 0.2] },
                          "m.l.bias" => { shape: [1], data: [0.0] } } }
    batch = { x: { shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0] } }

    Dir.mktmpdir("torobi-contract") do |dir|
      path = File.join(dir, "c")
      Torobi::Session.open(config, weights: weights, optimizer: { kind: :sgd, lr: 0.1 }) do |s|
        s.step!(batch)
        s.checkpoint!(path)
      end
      File.unlink(File.join(path, "random.safetensors"))

      Torobi::Session.open(config, weights: weights, optimizer: { kind: :adamw, lr: 0.01 }) do |s|
        s.step!(batch)
        before = { step: s.step, weight: s.fetch("m.l.weight").to_a, seed: s.seed }
        assert_raises(Torobi::StepError) { s.restore(path) }
        assert_equal before[:step], s.step
        assert_equal before[:weight], s.fetch("m.l.weight").to_a
        assert_equal before[:seed], s.seed
        # And it still runs.
        s.step!(batch)
        assert_equal 2, s.step
      end
    end
  end

  # A failed step must leave the session where it was, including the
  # optimizer's slots and the RNG.
  def test_a_failed_step_changes_nothing
    skip "extension not compiled" unless defined?(Torobi::Session)
    config = Torobi::GraphConfig.new(models: { "m" => scalar_model })
    weights = { params: { "m.l.weight" => { shape: [1, 2], data: [0.1, 0.2] },
                          "m.l.bias" => { shape: [1], data: [0.0] } } }
    good = { x: { shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0] } }

    Torobi::Session.open(config, weights: weights, optimizer: { kind: :adamw, lr: 0.05 }) do |s|
      s.step!(good)
      before = s.fetch("m.l.weight").to_a

      assert_raises(Torobi::StepError) do
        s.step!({ x: { shape: [2, 3], data: [0.0] * 6 } })
      end
      assert_equal 1, s.step
      assert_equal before, s.fetch("m.l.weight").to_a

      # The next step continues the same trajectory: run the same batch
      # twice from a fresh session and compare.
      after_failure = s.step!(good)
      clean = Torobi::Session.open(config, weights: weights, optimizer: { kind: :adamw, lr: 0.05 }) do |t|
        t.step!(good)
        t.step!(good)
      end
      assert_in_delta clean, after_failure, 1e-6,
                      "the failed step should not have moved the optimizer"
    end
  end

  # Replacing a checkpoint must never leave neither the old nor the new.
  def test_replacing_a_checkpoint_keeps_one_of_them
    skip "extension not compiled" unless defined?(Torobi::Session)
    config = Torobi::GraphConfig.new(models: { "m" => scalar_model })
    weights = { params: { "m.l.weight" => { shape: [1, 2], data: [0.1, 0.2] },
                          "m.l.bias" => { shape: [1], data: [0.0] } } }
    batch = { x: { shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0] } }

    Dir.mktmpdir("torobi-contract") do |dir|
      path = File.join(dir, "c")
      Torobi::Session.open(config, weights: weights, optimizer: { kind: :sgd, lr: 0.1 }) do |s|
        s.step!(batch)
        s.checkpoint!(path)
        s.step!(batch)
        s.checkpoint!(path)
      end
      assert_equal 2, JSON.parse(File.read(File.join(path, "manifest.json"))).fetch("step")
      refute_path_exists "#{path}.writing"
      refute_path_exists "#{path}.replaced"
    end
  end
end
