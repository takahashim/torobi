# frozen_string_literal: true

module Torobi
  module IR
    # Where an input's data comes from. Every input says so, uniformly, as
    # one of two kinds:
    #
    #   Batch.new(field: "x")                      {"batch" => "x"}
    #   ModelOutput.new(model: "student",          {"model" => "student",
    #                   output: "logits")           "output" => "logits"}
    #
    # A model graph reads only from the batch. An objective graph reads from
    # both, and that is how the two halves of a GraphConfig are wired
    # (docs/plan.md section 5A.3); GraphConfig holds both to it.
    module Source
      # A field of the batch the run is fed.
      Batch = Data.define(:field) do
        def initialize(field:)
          super(field: Source.name!(field, "batch field"))
        end

        def to_h = { "batch" => field }
      end

      # A named output of another model in the same config.
      ModelOutput = Data.define(:model, :output) do
        def initialize(model:, output:)
          super(model: Source.name!(model, "model name"),
                output: Source.name!(output, "output name"))
        end

        def to_h = { "model" => model, "output" => output }
      end

      module_function

      # What the DSL calls, with the symbols it is written in.
      def batch(field) = Batch.new(field: field.to_s)
      def model_output(model, output) = ModelOutput.new(model: model.to_s, output: output.to_s)

      # The source a serialized input names.
      def from_h(h, where:)
        raise ConfigError, "#{where}: source must be a Hash, got #{h.inspect}" unless h.is_a?(Hash)

        case h.keys.sort
        when ["batch"] then Batch.new(field: h["batch"])
        when %w[model output] then ModelOutput.new(model: h["model"], output: h["output"])
        else
          raise ConfigError,
                "#{where}: a source is {\"batch\" => field} or " \
                "{\"model\" => name, \"output\" => name}, got #{h.inspect}"
        end
      end

      # The one check both kinds make of what they name. An empty or
      # missing name is refused as the value is made, so no input can hold
      # one, whichever way it arrived.
      def name!(value, what)
        return -value if value.is_a?(String) && !value.empty?

        raise ConfigError, "#{what} must be a non-empty String, got #{value.inspect}"
      end
    end
  end
end
