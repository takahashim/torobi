# frozen_string_literal: true

require_relative "test_helper"

class GraphTest < Minitest::Test
  IR = Torobi::IR

  # What shape inference would have given each node below; these tests
  # are about structure, so every node is the same small vector.
  VALUE = { shape: [2], dtype: :f32 }.freeze

  def test_a_well_formed_graph_is_representable
    graph = Torobi::TestGraphs.linear_graph

    assert_equal 3, graph.nodes.size
    assert_equal({ "loss" => IR::Ref.node(2) }, graph.outputs)
    assert_equal({ "loss" => "node:2" }, graph.to_h.fetch("outputs"), "and written as text")
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
          IR::NodeSpec.new(id: 0, op: "add", inputs: ["node:1", "input:0"], **VALUE),
          IR::NodeSpec.new(id: 1, op: "abs", inputs: ["input:0"], **VALUE)
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
                    nodes: [IR::NodeSpec.new(id: 0, op: "abs", inputs: ["input:7"], **VALUE)],
                    outputs: { "y" => "node:0" })
    end
    assert_match(/unknown input:7/, e.message)

    e = assert_raises(Torobi::ConfigError) do
      IR::Graph.new(inputs:, parameters: [],
                    nodes: [IR::NodeSpec.new(id: 0, op: "abs", inputs: ["input:0"],
                                             parameters: [3], **VALUE)],
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
        nodes: [IR::NodeSpec.new(id: 0, op: "abs", inputs: ["input:0"], **VALUE),
                IR::NodeSpec.new(id: 1, op: "neg", inputs: ["input:0"], **VALUE)],
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
      IR::NodeSpec.new(id: 0, op: "x", inputs: [], attributes: { "o" => Object.new }, **VALUE)
    end
    assert_match(/not JSON-serializable/, e.message)
  end

  # A reference is refused by the node that holds it when it cannot be
  # read at all; whether it points anywhere is the graph's to say.
  def test_a_malformed_reference_is_refused_by_its_node
    e = assert_raises(Torobi::ConfigError) do
      IR::NodeSpec.new(id: 0, op: "abs", inputs: ["garbage"], **VALUE)
    end
    assert_match(/"garbage" is not a reference/, e.message)
  end

  def test_a_node_says_what_shape_inference_gave_it
    assert_raises(ArgumentError) { IR::NodeSpec.new(id: 0, op: "abs", inputs: ["input:0"]) }
  end

  def test_a_source_round_trips_and_says_what_it_is
    batch = IR::Source.batch("x")
    read = IR::Source.model_output("student", "logits")

    assert_equal batch, IR::Source.from_h(batch.to_h, where: "t")
    assert_equal read, IR::Source.from_h(read.to_h, where: "t")
    e = assert_raises(Torobi::ConfigError) { IR::Source.from_h({ "model" => "m" }, where: "t") }
    assert_match(/\At: a source is/, e.message)
    assert_raises(Torobi::ConfigError) { IR::Source.batch("") }
  end

  # The spec refusals that had no test: a hash key that is not a string
  # (the canonical form would not be total), an input source of the wrong
  # kind, a parameter that is neither trainable nor frozen, an initializer
  # with no "type", and a spec of the wrong class in a list.
  def test_the_rest_of_the_spec_refusals_are_made_where_they_are_written
    e = assert_raises(Torobi::ConfigError) { IR::Json.primitive!({ 1 => "x" }, where: "t") }
    assert_match(/hash keys must be strings/, e.message)

    e = assert_raises(Torobi::ConfigError) do
      IR::InputSpec.new(id: 0, name: "x", shape: [2], dtype: :f32, source: "batch")
    end
    assert_match(/source is a String, expected an IR::Source/, e.message)

    e = assert_raises(Torobi::ConfigError) do
      IR::ParameterSpec.new(id: 0, path: "w", shape: [2], dtype: :f32,
                            initializer: { "type" => "zeros" }, trainable: :yes)
    end
    assert_match(/trainable must be true or false/, e.message)

    e = assert_raises(Torobi::ConfigError) do
      IR::ParameterSpec.new(id: 0, path: "w", shape: [2], dtype: :f32,
                            initializer: { "kind" => "zeros" })
    end
    assert_match(/must have a "type" key/, e.message)

    e = assert_raises(Torobi::ConfigError) do
      IR::Graph.new(inputs: [Object.new], parameters: [], nodes: [], outputs: { "y" => "input:0" })
    end
    assert_match(/input at position 0 is a Object, expected Torobi::IR::InputSpec/, e.message)
  end
end
