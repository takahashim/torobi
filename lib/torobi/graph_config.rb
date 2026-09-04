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

    # The one output an objective declares. The engine differentiates it,
    # so there is no room for a second, and no room for a guess.
    LOSS = "loss"

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
        # A model name qualifies what the model declares ("m.embedding",
        # "m.layers.0.attn.Wqkv.weight"), and the dot is what joins them.
        # An output's own name may hold one (`towers` declares
        # "queries.embedding"), so a name with a dot on both sides has two
        # readings: model "a" with output "b.c", or model "a.b" with
        # output "c". Refused here, where the ambiguity is created.
        if name.include?(".")
          raise ConfigError,
                "model name #{name.inspect} holds a dot, and a dot is what joins a " \
                "model to what it declares; #{name}.x could then be two things"
        end
        unless graph.is_a?(IR::Graph)
          raise ConfigError, "model #{name.inspect} is a #{graph.class}, expected Torobi::IR::Graph"
        end

        [-name, graph]
      end
      if objective && !objective.is_a?(IR::Graph)
        raise ConfigError, "objective is a #{objective.class}, expected Torobi::IR::Graph"
      end

      train = check_train(train, models)
      if objective
        check_objective(objective)
        check_wiring(objective, models)
      elsif !train.empty?
        check_single_loss(models)
      end
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

    # Whether this config has a loss: something to train on, to evaluate,
    # or to differentiate.
    #
    # An objective's is the one it declares. Without an objective it is the
    # single model's single scalar output, which is the shorthand a spike
    # is written in. A config that trains nothing and declares neither has
    # no loss at all, and is one opened to look at what a model produces
    # (`Session#forward`) rather than to move it.
    def loss?
      return true if objective
      return false unless models.size == 1

      graph = models.values.first
      return false unless graph.outputs.size == 1

      shape, dtype = graph.output_signature(graph.outputs.keys.first)
      shape&.empty? && dtype == :f32
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

    # Which models a run differentiates. Omitted means all of them, which
    # is what training is; `train: []` means none, which is a graph opened
    # to be read rather than trained (a loss over representations computed
    # elsewhere, say). Saying it is what separates that from the mistake
    # of declaring nothing trainable by accident, which is still refused.
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

    # What an objective must be, because the engine has no way to guess
    # otherwise: exactly one output, named "loss", a scalar, and no
    # parameters of its own. Every one of these was representable before,
    # and each ended in the engine picking something arbitrary or indexing
    # past the end of an empty slice.
    def check_objective(objective)
      unless objective.parameters.empty?
        raise ConfigError,
              "an objective has no parameters of its own; move " \
              "#{objective.parameters.map(&:path).map(&:inspect).join(", ")} into a model"
      end

      names = objective.outputs.keys
      unless names == [LOSS]
        raise ConfigError,
              "an objective declares exactly one output, named #{LOSS.inspect}; " \
              "this one declares #{names.map(&:inspect).join(", ")}"
      end

      shape, dtype = objective.output_signature(LOSS)
      unless shape && shape.empty?
        raise ConfigError,
              "the loss must be a scalar, and this one has shape #{shape.inspect}; " \
              "reduce it (mean or sum) before declaring it"
      end
      # Read as an f32 at the boundary, which is the whole of why this is
      # checked here rather than found out on the first step.
      return if dtype == :f32

      raise ConfigError,
            "the loss must be f32, and this one is #{dtype}; a model held in " \
            "another precision says where it comes back (g.cast(x, :f32))"
    end

    # Without an objective, the single model's own single output is the
    # loss, and the same demands apply to it.
    #
    # Only when something is trained. A config that trains nothing is not
    # obliged to produce a number nobody will differentiate, and `train: []`
    # is how a run is opened to be read rather than moved.
    def check_single_loss(models)
      unless models.size == 1
        raise ConfigError,
              "with more than one model an objective says which loss to train; " \
              "this config has #{models.keys.map(&:inspect).join(", ")} and none"
      end

      name, graph = models.first
      names = graph.outputs.keys
      unless names.size == 1
        raise ConfigError,
              "without an objective, model #{name.inspect} must declare exactly one " \
              "output to be the loss; it declares #{names.map(&:inspect).join(", ")}"
      end

      shape, dtype = graph.output_signature(names.first)
      return if shape&.empty? && dtype == :f32

      raise ConfigError,
            "without an objective, model #{name.inspect}'s output " \
            "#{names.first.inspect} is the loss, so it must be an f32 scalar, " \
            "and it is #{dtype}#{shape.inspect}. A config that trains nothing " \
            "(train: []) needs no loss, and can be asked what it produces."
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
        next if shape == input.shape && dtype == input.dtype

        raise ConfigError,
              "#{where} expects #{input.dtype}#{input.shape.inspect} from " \
              "#{model_name}.#{output}, which is #{dtype}#{shape.inspect}"
      end
    end
  end
end
