# frozen_string_literal: true

module Torobi
  module IR
    # What a shape in the IR may be: an array of positive integers, where a
    # value that flows through the graph may leave a dimension nil
    # (symbolic: batch, sequence) and storage may not. The one check the
    # specs share, as Dtype is the one for dtypes.
    module Dimensions
      module_function

      # Returns a frozen copy of +shape+, or raises ConfigError.
      def check!(shape, where:, symbolic:)
        return Freeze.deep(shape.dup) if valid?(shape, symbolic:)

        allowed = symbolic ? "positive integers or nil (symbolic)" : "positive integers"
        raise ConfigError, "#{where}: shape must be an array of #{allowed}, got #{shape.inspect}"
      end

      def valid?(shape, symbolic:)
        shape.is_a?(Array) &&
          shape.all? { |d| (symbolic && d.nil?) || (d.is_a?(Integer) && d.positive?) }
      end
    end
  end
end
