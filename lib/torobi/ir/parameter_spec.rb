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

        where = "parameter #{path.inspect}"
        shape = check_shape(shape, where:)
        Dtype.check!(dtype, where:)
        initializer = check_initializer(initializer, where:)
        unless [true, false].include?(trainable)
          raise ConfigError, "#{where}: trainable must be true or false"
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

      # Parameters are storage: every dimension must be concrete, and there
      # is at least one.
      def check_shape(shape, where:)
        raise ConfigError, "#{where}: shape must not be empty (a scalar)" if shape == []

        Dimensions.check!(shape, where:, symbolic: false)
      end

      # The top level is keyed like keyword arguments, so `{type: "zeros"}`
      # reads as `{"type" => "zeros"}`; below it an initializer is JSON
      # data like any other, and a symbol key there is refused.
      def check_initializer(initializer, where:)
        unless initializer.is_a?(Hash)
          raise ConfigError, "#{where}: initializer must be a Hash, got #{initializer.inspect}"
        end

        initializer = initializer.transform_keys(&:to_s)
        unless initializer["type"]
          raise ConfigError,
                "#{where}: initializer must have a \"type\" key, got #{initializer.inspect}"
        end
        Freeze.deep(Json.primitive!(initializer, where: "#{where} initializer"))
      end
    end
  end
end
