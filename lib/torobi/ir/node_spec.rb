# frozen_string_literal: true

module Torobi
  module IR
    # One operation: which op, what it reads (references to inputs and
    # earlier nodes), which parameters it uses, its attributes, and the
    # shape and dtype shape inference gave it. A node is only ever made
    # after its shape is known, so neither is optional.
    #
    # `name` is a stable path under the scopes the node was built in
    # ("layers.3.attn.sdpa"), which is how a tap asks for it (docs/plan.md
    # section 6.4). Names are unique within a graph, and nil for nodes the
    # DSL did not name.
    class NodeSpec < Data.define(:id, :op, :name, :inputs, :parameters, :attributes,
                                 :shape, :dtype)
      def initialize(id:, op:, inputs:, shape:, dtype:, name: nil, parameters: [],
                     attributes: {})
        op = op.to_s
        raise ConfigError, "node #{id}: op must not be empty" if op.empty?

        name = name&.to_s
        raise ConfigError, "node #{id}: name must not be empty" if name && name.empty?

        where = "node #{id} (#{op})"
        # Parsed here so a malformed reference is refused by the node that
        # holds it; whether it points anywhere is the graph's to say.
        inputs.each { |ref| Ref.parse(ref) }
        inputs = Freeze.deep(inputs.map { |ref| -ref.to_s })
        parameters = Freeze.deep(parameters.map { |pid| Integer(pid) })
        attributes = Freeze.deep(Json.primitive!(attributes.transform_keys(&:to_s),
                                                 where: "#{where} attributes"))
        shape = Dimensions.check!(shape, where:, symbolic: true)
        Dtype.check!(dtype, where:)
        super(id: Integer(id), op: -op, name: name && -name, inputs:, parameters:,
              attributes:, shape:, dtype:)
      end

      def to_h
        {
          "id" => id, "op" => op, "name" => name, "inputs" => inputs,
          "parameters" => parameters, "attributes" => Json.canonical(attributes),
          "shape" => shape, "dtype" => dtype.to_s
        }
      end

      def self.from_h(h)
        new(id: h.fetch("id"), op: h.fetch("op"), name: h["name"], inputs: h.fetch("inputs"),
            parameters: h.fetch("parameters"), attributes: h.fetch("attributes"),
            shape: h.fetch("shape"), dtype: h.fetch("dtype").to_sym)
      end
    end
  end
end
