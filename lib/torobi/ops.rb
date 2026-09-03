# frozen_string_literal: true

require "yaml"

module Torobi
  # The operation vocabulary, loaded from config/ops.yml, the single source
  # of truth. The DSL asks it which ops exist and what they accept; the
  # inventory tests hold it against the shape rules and the handle methods;
  # the engine's dispatch table will be generated from the same file.
  module Ops
    ATTRIBUTE_TYPES = {
      "int" => ->(v) { v.is_a?(Integer) },
      "number" => ->(v) { v.is_a?(Numeric) },
      "number_or_null" => ->(v) { v.nil? || v.is_a?(Numeric) },
      "bool" => ->(v) { [true, false].include?(v) },
      # A dtype by name, from the IR's own small vocabulary.
      "dtype" => ->(v) { IR::Dtype::ALL.include?(v.to_s.to_sym) },
      "int_list" => ->(v) { v.is_a?(Array) && v.all?(Integer) },
      "int_list_or_null" => ->(v) { v.nil? || (v.is_a?(Array) && v.all?(Integer)) }
    }.freeze

    # One manifest entry.
    class Spec < Data.define(:name, :input_range, :params, :attributes, :shape_rule, :handle)
      def self.from_manifest(entry)
        inputs = entry.fetch("inputs")
        input_range = inputs.is_a?(Array) ? (inputs[0]..inputs[1]) : (inputs..inputs)
        attributes = (entry["attributes"] || {}).each do |key, type|
          ATTRIBUTE_TYPES.fetch(type) do
            raise ConfigError, "ops.yml: #{entry["name"]}.#{key} has unknown type #{type.inspect}"
          end
        end
        new(name: entry.fetch("name"), input_range:, params: entry["params"] || 0,
            attributes:, shape_rule: entry.fetch("shape_rule").to_sym,
            handle: entry["handle"] || false)
      end

      # Checks arity and attributes for one emission; the caller names the site.
      def check!(inputs:, params:, attrs:, where:)
        unless input_range.cover?(inputs)
          raise ConfigError,
                "#{where}: #{name} takes #{input_range} input(s), got #{inputs}"
        end
        unless params == self.params
          raise ConfigError, "#{where}: #{name} takes #{self.params} parameter(s), got #{params}"
        end

        attrs.each_key do |key|
          unless attributes.key?(key)
            raise ConfigError, "#{where}: #{name} has no attribute #{key.inspect}"
          end
        end
        attributes.each do |key, type|
          value = attrs[key]
          next if ATTRIBUTE_TYPES.fetch(type).call(value)

          raise ConfigError,
                "#{where}: #{name} attribute #{key} must be #{type}, got #{value.inspect}"
        end
      end
    end

    MANIFEST_PATH = File.expand_path("../../config/ops.yml", __dir__)

    REGISTRY = YAML.safe_load_file(MANIFEST_PATH).to_h do |entry|
      spec = Spec.from_manifest(entry)
      [spec.name, spec]
    end.freeze

    module_function

    def fetch(name, where:)
      REGISTRY.fetch(name.to_s) do
        raise ConfigError, "#{where}: unknown op #{name.inspect}"
      end
    end

    def handle_ops = REGISTRY.each_value.select(&:handle)
  end
end
