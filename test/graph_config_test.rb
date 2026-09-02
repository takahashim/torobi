# frozen_string_literal: true

require_relative "test_helper"

class GraphConfigTest < Minitest::Test
  def test_the_same_definition_always_yields_the_same_digest
    a = Torobi::TestGraphs.config(metadata: { "b" => 1, "a" => { "y" => 2, "x" => 1 } })
    b = Torobi::TestGraphs.config(extra_metadata_order: true,
                                  metadata: { "a" => { "x" => 1, "y" => 2 }, "b" => 1 })
    assert_equal a.canonical_json, b.canonical_json
    assert_equal a.digest, b.digest
    assert_match(/\A[0-9a-f]{64}\z/, a.digest)
  end

  def test_model_order_does_not_reach_the_digest
    g = Torobi::TestGraphs.linear_graph
    ab = Torobi::GraphConfig.new(models: { "a" => g, "b" => g })
    ba = Torobi::GraphConfig.new(models: { "b" => g, "a" => g })
    assert_equal ab.digest, ba.digest
  end

  def test_the_config_round_trips_through_its_canonical_json
    config = Torobi::TestGraphs.config(metadata: { "note" => "roundtrip" })
    revived = Torobi::GraphConfig.from_h(JSON.parse(config.canonical_json))
    assert_equal config.digest, revived.digest
    assert_equal config, revived
  end

  def test_everything_reachable_is_frozen
    config = Torobi::TestGraphs.config
    graph = config.models.fetch("student")
    assert_predicate config.models, :frozen?
    assert_predicate graph.nodes, :frozen?
    assert_predicate graph.nodes[0].inputs, :frozen?
    assert_predicate graph.parameters[0].initializer, :frozen?
    assert_raises(FrozenError) { graph.nodes[0].inputs << "node:0" }
    assert_raises(FrozenError) { config.models["another"] = graph }
  end

  def test_an_objective_graph_is_carried_beside_the_models
    g = Torobi::TestGraphs.linear_graph
    with = Torobi::GraphConfig.new(models: { "student" => g }, objective: g)
    without = Torobi::GraphConfig.new(models: { "student" => g })
    refute_equal with.digest, without.digest
    assert_nil without.to_h.fetch("objective")
  end

  def test_wrong_shapes_of_the_bundle_are_rejected
    g = Torobi::TestGraphs.linear_graph
    assert_raises(Torobi::ConfigError) { Torobi::GraphConfig.new(models: {}) }
    assert_raises(Torobi::ConfigError) { Torobi::GraphConfig.new(models: { "m" => :not_a_graph }) }
    assert_raises(Torobi::ConfigError) do
      Torobi::GraphConfig.new(models: { "m" => g }, objective: :nope)
    end
    assert_raises(Torobi::ConfigError) do
      Torobi::GraphConfig.new(models: { "m" => g }, metadata: { "k" => Object.new })
    end
  end
end
