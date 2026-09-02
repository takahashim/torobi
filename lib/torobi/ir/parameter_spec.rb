# frozen_string_literal: true

module Torobi
  module IR
    # One learnable (or frozen) tensor: a stable path such as
    # "layers.3.attn.wqkv.weight", a concrete shape, and an initializer
    # declaration. The initializer is data, not code: either a random
    # scheme, or {"type" => "pretrained", ...} naming a tensor in a
    # checkpoint. Parameter order and paths are what checkpoints are
    # validated against, so both must be stable.
    class ParameterSpec < Data.define(:id, :path, :shape, :dtype, :initializer, :trainable)
      def initialize(id:, path:, shape:, dtype:, initializer:, trainable: true)
        path = path.to_s
        raise ConfigError, "parameter #{id}: path must not be empty" if path.empty?

        shape = check_shape(path, shape)
        Dtype.check!(dtype, where: "parameter #{path.inspect}")
        initializer = check_initializer(path, initializer)
        unless [true, false].include?(trainable)
          raise ConfigError, "parameter #{path.inspect}: trainable must be true or false"
        end

        super(id: Integer(id), path: -path, shape:, dtype:, initializer:, trainable:)
      end

      def to_h
        {
          "id" => id, "path" => path, "shape" => shape, "dtype" => dtype.to_s,
          "initializer" => Json.canonical(initializer), "trainable" => trainable
        }
      end

      def self.from_h(h)
        new(id: h.fetch("id"), path: h.fetch("path"), shape: h.fetch("shape"),
            dtype: h.fetch("dtype").to_sym, initializer: h.fetch("initializer"),
            trainable: h.fetch("trainable"))
      end

      private

      # Parameters are storage: every dimension must be concrete.
      def check_shape(path, shape)
        unless shape.is_a?(Array) && !shape.empty? &&
               shape.all? { |d| d.is_a?(Integer) && d.positive? }
          raise ConfigError,
                "parameter #{path.inspect}: shape must be a non-empty array of " \
                "positive integers, got #{shape.inspect}"
        end
        Freeze.deep(shape.dup)
      end

      def check_initializer(path, initializer)
        unless initializer.is_a?(Hash) && (initializer["type"] || initializer[:type])
          raise ConfigError,
                "parameter #{path.inspect}: initializer must be a Hash with a \"type\" key, " \
                "got #{initializer.inspect}"
        end
        Freeze.deep(Json.primitive!(initializer.transform_keys(&:to_s),
                                    where: "parameter #{path.inspect} initializer"))
      end
    end
  end
end
