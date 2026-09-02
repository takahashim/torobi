# frozen_string_literal: true

require "json"
require "digest"

module Torobi
  # The serializable artifact the Graph DSL produces and everything else
  # consumes: named model graphs, an optional objective graph wiring them
  # into a loss, and metadata. Immutable; its canonical JSON is deterministic
  # (fixed key order, models sorted by name), so the same definition always
  # yields the same digest.
  class GraphConfig < Data.define(:schema_version, :semantics_version, :models,
                                  :objective, :metadata)
    SCHEMA_VERSION = 1
    SEMANTICS_VERSION = 1

    def initialize(models:, objective: nil, metadata: {},
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
      metadata = IR::Json.primitive!(metadata.transform_keys(&:to_s), where: "metadata")

      super(schema_version: Integer(schema_version),
            semantics_version: Integer(semantics_version),
            models: Freeze.deep(models), objective:, metadata: Freeze.deep(metadata))
    end

    def to_h
      {
        "schema_version" => schema_version,
        "semantics_version" => semantics_version,
        "models" => models.sort.to_h { |name, graph| [name, graph.to_h] },
        "objective" => objective&.to_h,
        "metadata" => IR::Json.canonical(metadata)
      }
    end

    # The one serialization the digest is defined over. Key order is fixed by
    # construction (ordered hashes; models and metadata sorted by key).
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
          metadata: h.fetch("metadata"))
    end
  end
end
