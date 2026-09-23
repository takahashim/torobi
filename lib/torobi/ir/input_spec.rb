# frozen_string_literal: true

module Torobi
  module IR
    # One graph input: a name, where its data comes from ([`Source`]), a
    # shape whose nil dimensions are symbolic (batch, sequence), and a dtype.
    class InputSpec < Data.define(:id, :name, :source, :shape, :dtype)
      def initialize(id:, name:, shape:, dtype:, source: nil)
        name = name.to_s
        raise ConfigError, "input #{id}: name must not be empty" if name.empty?

        where = "input #{name.inspect}"
        shape = Dimensions.check!(shape, where:, symbolic: true)
        Dtype.check!(dtype, where:)
        source ||= Source.batch(name)
        unless source.is_a?(Source::Batch) || source.is_a?(Source::ModelOutput)
          raise ConfigError, "#{where}: source is a #{source.class}, expected an IR::Source"
        end

        super(id: Integer(id), name: -name, source:, shape:, dtype:)
      end

      def from_batch? = source.is_a?(Source::Batch)
      def from_model? = source.is_a?(Source::ModelOutput)

      def to_h
        { "id" => id, "name" => name, "source" => source.to_h, "shape" => shape,
          "dtype" => dtype.to_s }
      end

      def self.from_h(h)
        where = "input #{h["name"].inspect}"
        new(id: h.fetch("id"), name: h.fetch("name"),
            source: h["source"]&.then { |s| Source.from_h(s, where:) },
            shape: h.fetch("shape"), dtype: h.fetch("dtype").to_sym)
      end
    end
  end
end
