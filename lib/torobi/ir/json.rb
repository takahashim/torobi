# frozen_string_literal: true

module Torobi
  module IR
    # What may appear inside attributes, initializers and metadata: JSON
    # primitives only, so the canonical serialization is total and the
    # digest never depends on Ruby object identity.
    module Json
      module_function

      def primitive!(value, where:)
        case value
        when nil, true, false, Integer, Float, String
          value
        when Array
          value.each { |v| primitive!(v, where:) }
          value
        when Hash
          value.each do |k, v|
            unless k.is_a?(String)
              raise ConfigError, "#{where}: hash keys must be strings, got #{k.inspect}"
            end

            primitive!(v, where:)
          end
          value
        else
          raise ConfigError,
                "#{where}: #{value.inspect} is not JSON-serializable data " \
                "(allowed: nil, booleans, numbers, strings, arrays, hashes)"
        end
      end

      # The order-normalized form the canonical JSON is built from: hash keys
      # sorted at every depth, so insertion order never reaches the digest.
      def canonical(value)
        case value
        when Hash then value.sort.to_h { |k, v| [k, canonical(v)] }
        when Array then value.map { |v| canonical(v) }
        else value
        end
      end
    end
  end
end
