# frozen_string_literal: true

module Torobi
  module IR
    # References between IR values, serialized as short strings so the JSON
    # stays readable: "input:0" is the first InputSpec, "node:3" the fourth
    # NodeSpec. Parameters are referenced by their bare id in NodeSpec, not
    # through Ref, since they live in a namespace of their own.
    module Ref
      PATTERN = /\A(input|node):(0|[1-9][0-9]*)\z/

      module_function

      def input(id) = "input:#{Integer(id)}"
      def node(id) = "node:#{Integer(id)}"

      # Returns [:input | :node, id], or raises ConfigError.
      def parse(ref)
        m = PATTERN.match(ref.to_s) or
          raise ConfigError, "#{ref.inspect} is not a reference; expected \"input:N\" or \"node:N\""
        [m[1].to_sym, Integer(m[2])]
      end

      # The spec +ref+ points at, among +inputs+ and +nodes+, or a
      # ConfigError saying +where+ asked for something that is not there.
      def resolve(ref, inputs:, nodes:, where:)
        kind, id = parse(ref)
        specs = kind == :input ? inputs : nodes
        return specs[id] if id < specs.size

        raise ConfigError, "#{where} references unknown #{kind}:#{id}"
      end
    end
  end
end
