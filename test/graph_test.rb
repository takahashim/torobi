# frozen_string_literal: true

require_relative "test_helper"

class GraphTest < Minitest::Test
  IR = Torobi::IR

  def test_a_well_formed_graph_is_representable
    graph = Torobi::TestGraphs.linear_graph
    assert_equal 3, graph.nodes.size
    assert_equal({ "loss" => "node:2" }, graph.outputs)
  end

  def test_ids_must_be_consecutive_from_zero
    e = assert_raises(Torobi::ConfigError) do
      IR::Graph.new(
        inputs: [IR::InputSpec.new(id: 1, name: "x", shape: [2], dtype: :f32)],
        parameters: [], nodes: [], outputs: { "y" => "input:0" }
      )
    end
    assert_match(/consecutive from 0/, e.message)
  end

  def test_forward_references_are_rejected
    e = assert_raises(Torobi::ConfigError) do
      IR::Graph.new(
        inputs: [IR::InputSpec.new(id: 0, name: "x", shape: [2], dtype: :f32)],
        parameters: [],
        nodes: [
          IR::NodeSpec.new(id: 0, op: "add", inputs: ["node:1", "input:0"]),
          IR::NodeSpec.new(id: 1, op: "abs", inputs: ["input:0"])
        ],
        outputs: { "y" => "node:0" }
      )
    end
    assert_match(/forward references/, e.message)
    assert_match(/node 0 \(add\)/, e.message)
  end

  def test_unknown_references_are_rejected_by_name
    inputs = [IR::InputSpec.new(id: 0, name: "x", shape: [2], dtype: :f32)]
    e = assert_raises(Torobi::ConfigError) do
      IR::Graph.new(inputs:, parameters: [],
                    nodes: [IR::NodeSpec.new(id: 0, op: "abs", inputs: ["input:7"])],
                    outputs: { "y" => "node:0" })
    end
    assert_match(/unknown input:7/, e.message)

    e = assert_raises(Torobi::ConfigError) do
      IR::Graph.new(inputs:, parameters: [],
                    nodes: [IR::NodeSpec.new(id: 0, op: "abs", inputs: ["input:0"],
                                             parameters: [3])],
                    outputs: { "y" => "node:0" })
    end
    assert_match(/unknown parameter 3/, e.message)

    e = assert_raises(Torobi::ConfigError) do
      IR::Graph.new(inputs:, parameters: [], nodes: [], outputs: { "y" => "node:0" })
    end
    assert_match(/output "y" references unknown node:0/, e.message)
  end

  def test_duplicate_names_and_paths_are_rejected
    e = assert_raises(Torobi::ConfigError) do
      IR::Graph.new(
        inputs: [IR::InputSpec.new(id: 0, name: "x", shape: [2], dtype: :f32),
                 IR::InputSpec.new(id: 1, name: "x", shape: [2], dtype: :f32)],
        parameters: [], nodes: [], outputs: { "y" => "input:0" }
      )
    end
    assert_match(/duplicate input name "x"/, e.message)
  end

  def test_dead_nodes_are_rejected_by_name
    e = assert_raises(Torobi::ConfigError) do
      IR::Graph.new(
        inputs: [IR::InputSpec.new(id: 0, name: "x", shape: [2], dtype: :f32)],
        parameters: [],
        nodes: [IR::NodeSpec.new(id: 0, op: "abs", inputs: ["input:0"]),
                IR::NodeSpec.new(id: 1, op: "neg", inputs: ["input:0"])],
        outputs: { "y" => "node:1" }
      )
    end
    assert_match(/unreachable from any output: node 0 \(abs\)/, e.message)
  end

  def test_a_graph_needs_at_least_one_output
    e = assert_raises(Torobi::ConfigError) do
      IR::Graph.new(inputs: [], parameters: [], nodes: [], outputs: {})
    end
    assert_match(/at least one named output/, e.message)
  end

  def test_spec_level_mistakes_are_rejected_where_they_are_made
    assert_raises(Torobi::ConfigError) do
      IR::InputSpec.new(id: 0, name: "x", shape: [0], dtype: :f32)      # zero dim
    end
    assert_raises(Torobi::ConfigError) do
      IR::InputSpec.new(id: 0, name: "x", shape: [2], dtype: :f64)      # unknown dtype
    end
    assert_raises(Torobi::ConfigError) do
      IR::ParameterSpec.new(id: 0, path: "w", shape: [nil, 2], dtype: :f32,
                            initializer: { "type" => "zeros" })         # symbolic param shape
    end
    assert_raises(Torobi::ConfigError) do
      IR::ParameterSpec.new(id: 0, path: "w", shape: [2], dtype: :f32,
                            initializer: "zeros")                       # initializer not a Hash
    end
    e = assert_raises(Torobi::ConfigError) do
      IR::NodeSpec.new(id: 0, op: "x", inputs: [], attributes: { "o" => Object.new })
    end
    assert_match(/not JSON-serializable/, e.message)
  end
end
