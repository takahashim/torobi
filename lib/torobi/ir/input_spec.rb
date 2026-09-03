# frozen_string_literal: true

module Torobi
  module IR
    # One graph input: a name, where its data comes from ([`Source`]), a
    # shape whose nil dimensions are symbolic (batch, sequence), and a dtype.
    class InputSpec < Data.define(:id, :name, :source, :shape, :dtype)
      def initialize(id:, name:, shape:, dtype:, source: nil)
        name = name.to_s
        raise ConfigError, "input #{id}: name must not be empty" if name.empty?

        shape = check_shape(id, shape)
        Dtype.check!(dtype, where: "input #{name.inspect}")
        source = Source.check!(source || Source.batch(name), where: "input #{name.inspect}")
        super(id: Integer(id), name: -name, source:, shape:, dtype:)
      end

      def from_batch? = Source.batch?(source)
      def from_model? = Source.model?(source)

      def to_h
        { "id" => id, "name" => name, "source" => source, "shape" => shape,
          "dtype" => dtype.to_s }
      end

      def self.from_h(h)
        new(id: h.fetch("id"), name: h.fetch("name"), source: h["source"],
            shape: h.fetch("shape"), dtype: h.fetch("dtype").to_sym)
      end

      private

      def check_shape(id, shape)
        unless shape.is_a?(Array) && shape.all? { |d| d.nil? || (d.is_a?(Integer) && d.positive?) }
          raise ConfigError,
                "input #{id}: shape must be an array of positive integers or nil " \
                "(symbolic), got #{shape.inspect}"
        end
        Freeze.deep(shape.dup)
      end
    end
  end
end
