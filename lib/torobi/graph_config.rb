# frozen_string_literal: true

require "json"
require "digest"

module Torobi
  # The serializable artifact the Graph DSL produces and everything else
  # consumes: named model graphs, an optional objective graph wiring them
  # into a loss, which of them are trained, and metadata.
  #
  # Immutable; its canonical JSON is deterministic (fixed key order, models
  # sorted by name), so the same definition always yields the same digest.
  class GraphConfig < Data.define(:schema_version, :semantics_version, :models,
                                  :objective, :train, :metadata)
    SCHEMA_VERSION = 1
    SEMANTICS_VERSION = 1

    # A parameter as the engine sees it: namespaced by its model, and
    # carrying whether gradients are wanted for it.
    Parameter = Data.define(:model, :path, :spec, :trained) do
      # "student.layers.0.wqkv.weight"
      def qualified_path = "#{model}.#{path}"
    end

    def initialize(models:, objective: nil, train: nil, metadata: {},
                   schema_version: SCHEMA_VERSION, semantics_version: SEMANTICS_VERSION)
      unless models.is_a?(Hash) && !models.empty?
        raise ConfigError, "models must be a non-empty Hash of name => Graph"
      end

      models = models.to_h do |name, graph|
        name = name.to_s
        raise ConfigError, "model name must not be empty" if name.empty?
        unless graph.is_a?(IR::Graph)
          raise ConfigError, "model #{name.inspect} is a #{graph.class}, expected Torobi::IR::Graph"
        end

        [-name, graph]
      end
      if objective && !objective.is_a?(IR::Graph)
        raise ConfigError, "objective is a #{objective.class}, expected Torobi::IR::Graph"
      end
      train = check_train(train, models)
      check_wiring(objective, models) if objective
      metadata = IR::Json.primitive!(metadata.transform_keys(&:to_s), where: "metadata")

      super(schema_version: Integer(schema_version),
            semantics_version: Integer(semantics_version),
            models: Freeze.deep(models.sort.to_h), objective:,
            train: Freeze.deep(train), metadata: Freeze.deep(metadata))
    end

    # Every parameter, model by model in name order, each namespaced and
    # marked with whether it is differentiated. This order is the contract:
    # checkpoints and the engine's argnums both follow it.
    def parameters
      models.flat_map do |name, graph|
        trained = train.include?(name)
        graph.parameters.map do |spec|
          Parameter.new(model: name, path: spec.path, spec:,
                        trained: trained && spec.trainable)
        end
      end
    end

    # The positions in `parameters` that autodiff differentiates: trainable
    # parameters of trained models, and nothing else (docs/plan.md 5A.3).
    def argnums
      parameters.each_index.select { |i| parameters[i].trained }
    end

    def to_h
      {
        "schema_version" => schema_version,
        "semantics_version" => semantics_version,
        "models" => models.to_h { |name, graph| [name, graph.to_h] },
        "objective" => objective&.to_h,
        "train" => train,
        "metadata" => IR::Json.canonical(metadata)
      }
    end

    # The one serialization the digest is defined over. Key order is fixed by
    # construction (ordered hashes; models, train and metadata sorted).
    def canonical_json
      JSON.generate(to_h)
    end

    def digest
      Digest::SHA256.hexdigest(canonical_json)
    end

    def self.from_h(h)
      new(schema_version: h.fetch("schema_version"),
          semantics_version: h.fetch("semantics_version"),
          models: h.fetch("models").transform_values { |g| IR::Graph.from_h(g) },
          objective: h["objective"]&.then { |g| IR::Graph.from_h(g) },
          train: h["train"],
          metadata: h.fetch("metadata"))
    end

    private

    # Which models are trained. Default: all of them, which is right for a
    # single model and wrong for nothing, since a teacher is named
    # explicitly the moment there is one.
    def check_train(train, models)
      return models.keys.sort if train.nil?

      unless train.is_a?(Array)
        raise ConfigError, "train must be an Array of model names, got #{train.inspect}"
      end

      train.map do |name|
        name = name.to_s
        unless models.key?(name)
          raise ConfigError,
                "train names #{name.inspect}, which is not a model here " \
                "(#{models.keys.map(&:inspect).join(", ")})"
        end
        -name
      end.sort.uniq
    end

    # An objective may read a model output only if that model declares it,
    # with the shape and dtype it declared. The two halves are written
    # separately, so this is where they are held to each other.
    def check_wiring(objective, models)
      objective.inputs.each do |input|
        next unless input.from_model?

        model_name = input.source.fetch("model")
        output = input.source.fetch("output")
        where = "objective input #{input.name.inspect}"
        graph = models[model_name] or
          raise ConfigError,
                "#{where} reads #{model_name}.#{output}, but there is no model " \
                "named #{model_name.inspect} (#{models.keys.map(&:inspect).join(", ")})"

        shape, dtype = graph.output_signature(output)
        unless shape == input.shape && dtype == input.dtype
          raise ConfigError,
                "#{where} expects #{dtype}#{input.shape.inspect} from " \
                "#{model_name}.#{output}, which is #{dtype}#{shape.inspect}"
        end
      rescue ConfigError => e
        raise ConfigError, e.message
      end
    end
  end
end
