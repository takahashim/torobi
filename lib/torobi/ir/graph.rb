# frozen_string_literal: true

module Torobi
  module IR
    # One computation graph: inputs, parameters, nodes in topological order,
    # and its named outputs. Construction validates the structure
    # ([`Graph::Check`]), so an invalid graph is never representable.
    #
    # Outputs are named because that is what an objective graph refers to
    # (docs/plan.md section 5A.3): {"logits" => "node:12"}.
    class Graph < Data.define(:inputs, :parameters, :nodes, :outputs)
      def initialize(inputs:, parameters:, nodes:, outputs:)
        outputs = Check.new(inputs:, parameters:, nodes:).call(outputs)
        super(inputs: Freeze.deep(inputs.dup), parameters: Freeze.deep(parameters.dup),
              nodes: Freeze.deep(nodes.dup), outputs:)
      end

      def to_h
        {
          "inputs" => inputs.map(&:to_h),
          "parameters" => parameters.map(&:to_h),
          "nodes" => nodes.map(&:to_h),
          "outputs" => outputs.transform_values(&:to_s)
        }
      end

      # The input or node +ref+ (a Ref, or one written out) points at.
      def resolve(ref) = Ref.parse(ref).resolve(inputs:, nodes:, where: "this graph")

      # The shape and dtype behind one named output, for whoever consumes it.
      def output_signature(name)
        ref = outputs.fetch(name.to_s) do
          raise ConfigError,
                "this graph has no output #{name.to_s.inspect}; it has " \
                "#{outputs.keys.map(&:inspect).join(", ")}"
        end
        spec = resolve(ref)
        [spec.shape, spec.dtype]
      end

      # Every name a tap could ask for.
      def node_names = nodes.filter_map(&:name)

      def self.from_h(h)
        new(inputs: h.fetch("inputs").map { |x| InputSpec.from_h(x) },
            parameters: h.fetch("parameters").map { |x| ParameterSpec.from_h(x) },
            nodes: h.fetch("nodes").map { |x| NodeSpec.from_h(x) },
            outputs: h.fetch("outputs"))
      end
    end
  end
end
