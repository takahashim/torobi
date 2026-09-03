# frozen_string_literal: true

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "torobi"
require "minitest/autorun"
require "tmpdir"

module Torobi
  # Builders for the tests: a tiny linear model (x @ w + b) and an mse
  # objective, assembled by hand at the IR level. The DSL will produce the
  # same values later; these tests pin the value semantics underneath it.
  module TestGraphs
    module_function

    def linear_graph(extra_metadata_order: false)
      inputs = [IR::InputSpec.new(id: 0, name: "x", shape: [nil, 4], dtype: :f32)]
      parameters = [
        IR::ParameterSpec.new(id: 0, path: "linear.weight", shape: [4, 2], dtype: :f32,
                              initializer: init(extra_metadata_order)),
        IR::ParameterSpec.new(id: 1, path: "linear.bias", shape: [2], dtype: :f32,
                              initializer: { "type" => "zeros" }, trainable: false)
      ]
      nodes = [
        IR::NodeSpec.new(id: 0, op: "matmul", inputs: [IR::Ref.input(0)], parameters: [0]),
        IR::NodeSpec.new(id: 1, op: "add", inputs: [IR::Ref.node(0)], parameters: [1]),
        # A config without an objective takes the model's single output as
        # the loss, so it has to be a scalar (GraphConfig::LOSS).
        IR::NodeSpec.new(id: 2, op: "mean", inputs: [IR::Ref.node(1)],
                         attributes: { "axes" => nil, "keepdims" => false },
                         shape: [], dtype: :f32)
      ]
      IR::Graph.new(inputs:, parameters:, nodes:, outputs: { "loss" => IR::Ref.node(2) })
    end

    # The same initializer built in two different key orders.
    def init(reversed)
      if reversed
        { "fan_mode" => "fan_in", "type" => "kaiming_uniform", "gain" => 1.0 }
      else
        { "type" => "kaiming_uniform", "gain" => 1.0, "fan_mode" => "fan_in" }
      end
    end

    def config(extra_metadata_order: false, **)
      Torobi::GraphConfig.new(models: { "student" => linear_graph(extra_metadata_order:) },
**)
    end
  end
end
