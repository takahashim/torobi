# frozen_string_literal: true

module Torobi
  module IR
    # A reference from one IR value to another: the `id`th input, or the
    # `id`th node. Written as a short string so the JSON stays readable
    # ("input:0", "node:3"), and read back from it once, where it enters,
    # so everything past that asks the value rather than the text.
    #
    # Parameters are referenced by their bare id in NodeSpec, not through
    # Ref, since they live in a namespace of their own.
    class Ref < Data.define(:kind, :id)
      KINDS = %i[input node].freeze
      PATTERN = /\A(input|node):(0|[1-9][0-9]*)\z/

      def self.input(id) = new(kind: :input, id:)
      def self.node(id) = new(kind: :node, id:)

      # The reference `ref` is, or is written as. A Ref passes through.
      def self.parse(ref)
        return ref if ref.is_a?(Ref)

        m = PATTERN.match(ref.to_s) or
          raise ConfigError, "#{ref.inspect} is not a reference; expected \"input:N\" or \"node:N\""
        new(kind: m[1].to_sym, id: Integer(m[2]))
      end

      def initialize(kind:, id:)
        raise ConfigError, "a reference is to an input or a node, not #{kind.inspect}" unless
          KINDS.include?(kind)

        super(kind:, id: Integer(id))
      end

      def input? = kind == :input
      def node? = kind == :node

      # The spec this points at, among `inputs` and `nodes`, or a
      # ConfigError saying `where` asked for something that is not there.
      def resolve(inputs:, nodes:, where:)
        specs = input? ? inputs : nodes
        return specs[id] if id < specs.size

        raise ConfigError, "#{where} references unknown #{self}"
      end

      def to_s = "#{kind}:#{id}"

      def inspect = to_s
    end
  end
end
