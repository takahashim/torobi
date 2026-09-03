# frozen_string_literal: true

module Torobi
  module IR
    # Where an input's data comes from. Every input says so, uniformly:
    #
    #   {"batch" => "x"}                          from the batch, by field
    #   {"model" => "student", "output" => "logits"}   a model's named output
    #
    # A model graph reads only from the batch. An objective graph reads from
    # both, and that is how the two halves of a GraphConfig are wired
    # (docs/plan.md section 5A.3).
    module Source
      module_function

      def batch(field) = { "batch" => field.to_s }.freeze

      def model_output(model, output)
        { "model" => model.to_s, "output" => output.to_s }.freeze
      end

      def batch?(source) = source.key?("batch")
      def model?(source) = source.key?("model")

      def check!(source, where:)
        unless source.is_a?(Hash)
          raise ConfigError, "#{where}: source must be a Hash, got #{source.inspect}"
        end

        case source.keys.sort
        when ["batch"]
          field!(source["batch"], "batch field", where:)
        when %w[model output]
          field!(source["model"], "model name", where:)
          field!(source["output"], "output name", where:)
        else
          raise ConfigError,
                "#{where}: a source is {\"batch\" => field} or " \
                "{\"model\" => name, \"output\" => name}, got #{source.inspect}"
        end
        Freeze.deep(source.dup)
      end

      def field!(value, what, where:)
        return if value.is_a?(String) && !value.empty?

        raise ConfigError, "#{where}: #{what} must be a non-empty String, got #{value.inspect}"
      end

      # How a source reads in an error message.
      def describe(source)
        if batch?(source)
          "batch[#{source["batch"].inspect}]"
        else
          "#{source["model"]}.#{source["output"]}"
        end
      end
    end
  end
end
