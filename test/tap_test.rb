# frozen_string_literal: true

require_relative "test_helper"

# Observation taps (docs/plan.md 8.3, capability B+): a named value inside
# the graph, read out for the window without changing anything. This is the
# answer to a closed engine being opaque - you can see in, but only look.
class TapTest < Minitest::Test
  DIM = 4
  ROWS = 8

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  # first: x @ 0.1 + 0 over DIM inputs of 1.0 = 0.4 everywhere.
  # second: that @ 0.1 summed = 0.16.
  def config
    model = Torobi.graph do |g|
      x = g.input :x, [nil, DIM]
      h = g.linear(x, DIM, name: "first")
      g.output :loss, g.mean(g.linear(h, 1, name: "second"))
    end
    Torobi::GraphConfig.new(models: { "m" => model })
  end

  def weights
    { params: {
      "m.first.weight" => { shape: [DIM, DIM], data: Array.new(DIM * DIM, 0.1) },
      "m.first.bias" => { shape: [DIM], data: Array.new(DIM, 0.0) },
      "m.second.weight" => { shape: [1, DIM], data: Array.new(DIM, 0.1) },
      "m.second.bias" => { shape: [1], data: [0.0] }
    } }
  end

  def batch = { x: { shape: [ROWS, DIM], data: Array.new(ROWS * DIM, 1.0) } }

  def test_the_names_a_tap_can_ask_for_are_the_ones_the_dsl_gave
    Torobi::Session.open(config, weights) do |s|
      assert_equal %w[first second], s.node_names
    end
  end

  # Every statistic is checked against the value we know is there: 0.4 in
  # every one of ROWS x DIM places after the first layer.
  def test_a_tap_reduces_on_the_device
    Torobi::Session.open(config, weights, optimizer: { kind: :sgd, lr: 0.0 }) do |s|
      s.tap("first", stat: :mean)
      s.step!(batch)
      assert_in_delta 0.4, s.tapped.fetch("first"), 1e-6

      s.tap("first", stat: :norm)
      s.step!(batch)
      assert_in_delta Math.sqrt(ROWS * DIM * 0.4**2), s.tapped.fetch("first"), 1e-5

      s.tap("first", stat: :extent)
      s.step!(batch)
      extent = s.tapped.fetch("first")
      assert_equal [2], extent.fetch(:shape)
      assert_in_delta 0.4, extent.fetch(:data).first, 1e-6
      assert_in_delta 0.4, extent.fetch(:data).last, 1e-6
    end
  end

  def test_a_full_tap_brings_the_tensor_back
    Torobi::Session.open(config, weights, optimizer: { kind: :sgd, lr: 0.0 }) do |s|
      s.tap("second", stat: :full)
      s.step!(batch)
      value = s.tapped.fetch("second")
      assert_equal [ROWS, 1], value.fetch(:shape)
      assert_equal ROWS, value.fetch(:data).size
      assert(value.fetch(:data).all? { |v| (v - 0.16).abs < 1e-6 })
    end
  end

  def test_several_taps_at_once_and_untapping
    Torobi::Session.open(config, weights, optimizer: { kind: :sgd, lr: 0.0 }) do |s|
      s.tap("first").tap("second", stat: :mean)
      assert_equal %w[first second], s.taps
      s.step!(batch)
      assert_equal %w[first second], s.tapped.keys

      assert s.untap("first")
      refute s.untap("first"), "untapping twice reports that there was nothing to untap"
      s.step!(batch)
      assert_equal %w[second], s.tapped.keys
    end
  end

  # A tap watches, it does not change: the same run with and without one
  # reaches the same place.
  def test_watching_does_not_change_what_is_learned
    without = Torobi::Session.open(config, weights, optimizer: { kind: :sgd, lr: 0.05 }) do |s|
      s.run([batch] * 5)
      s.fetch("m.second.weight")[:data]
    end
    with = Torobi::Session.open(config, weights, optimizer: { kind: :sgd, lr: 0.05 }) do |s|
      s.tap("first", stat: :norm)
      s.run([batch] * 5)
      s.fetch("m.second.weight")[:data]
    end
    assert_equal without, with
  end

  def test_asking_for_what_is_not_there_is_refused_by_name
    Torobi::Session.open(config, weights) do |s|
      e = assert_raises(Torobi::StepError) { s.tap("third") }
      assert_match(/no value is named "third"/, e.message)
      assert_match(/first/, e.message, "the refusal should say what there is")

      e = assert_raises(Torobi::StepError) { s.tap("first", stat: :median) }
      assert_match(/is not a statistic/, e.message)
    end
  end

  # A hook is the natural reader of a tap: it fires in the window, where a
  # tap's value is waiting.
  def test_a_hook_reads_what_the_tap_saw
    seen = []
    Torobi::Session.open(config, weights, optimizer: { kind: :sgd, lr: 0.05 }) do |s|
      s.tap("first", stat: :norm)
      s.on(:step) { |e| seen << e.session.tapped.fetch("first") }
      s.run([batch] * 3)
    end
    assert_equal 3, seen.size
    assert(seen.all? { |v| v.is_a?(Float) && v.positive? })
    refute_equal seen.first, seen.last, "the activation should move as it trains"
  end

  # Naming is per graph and stable under scopes, which is what makes
  # "layers.3.attn" mean one thing.
  def test_names_are_scoped_and_unique
    graph = Torobi.graph do |g|
      h = g.input :h, [nil, DIM]
      2.times { |i| g.scope("layers.#{i}") { h = g.linear(h, DIM, name: "ff") } }
      g.output :loss, g.mean(h)
    end
    assert_equal %w[layers.0.ff layers.1.ff], graph.node_names

    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph do |g|
        x = g.input :x, [nil, DIM]
        a = g.name("twice", g.mean(x))
        g.output :loss, g.name("twice", a * 2.0)
      end
    end
    assert_match(/two values are named "twice"/, e.message)
  end
end
