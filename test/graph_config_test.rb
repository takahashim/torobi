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

  # Two models need an objective to say which loss is trained, so this one
  # reads the first and ignores the second.
  def two_models
    g = Torobi::TestGraphs.linear_graph
    objective = Torobi.objective("a" => g, "b" => g) do |o|
      o.output :loss, o.from_model("a", :loss)
    end
    [g, objective]
  end

  def test_model_order_does_not_reach_the_digest
    g, objective = two_models
    ab = Torobi::GraphConfig.new(models: { "a" => g, "b" => g }, objective:, train: ["a"])
    ba = Torobi::GraphConfig.new(models: { "b" => g, "a" => g }, objective:, train: ["a"])
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
    objective = Torobi.objective("student" => g) do |o|
      o.output :loss, o.from_model("student", :loss) * 2.0
    end
    with = Torobi::GraphConfig.new(models: { "student" => g }, objective:)
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
    # An objective must be a scalar named "loss", with no parameters of its
    # own: the engine differentiates one value and passes it none.
    assert_raises(Torobi::ConfigError) do
      Torobi::GraphConfig.new(models: { "m" => g }, objective: g)
    end
    assert_raises(Torobi::ConfigError) do
      Torobi::GraphConfig.new(models: { "m" => g }, metadata: { "k" => Object.new })
    end
  end
end
