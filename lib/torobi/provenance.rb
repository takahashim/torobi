# frozen_string_literal: true

require "json"
require "digest"

module Torobi
  # Everything about a run that is not an event: what was trained, with
  # what, on what, and by which build. Gathered once, at the top of a
  # journal and in every checkpoint (docs/plan.md sections 8.5 and 11.2).
  #
  #   config     what the GraphConfig was: its digest, versions, models,
  #              what is trained, and every parameter by qualified path
  #   dataset    whatever the caller said about the data, verbatim
  #   runtime    the build: torobi, Ruby, the platform, and the engine's
  #              own report when the extension is there
  #   optimizer  the update rule the run was opened with
  #   seed       where its randomness starts
  Provenance = Data.define(:config, :dataset, :runtime, :optimizer, :seed) do
    def self.of(config, dataset: nil, optimizer: nil, seed: nil)
      new(config: {
            "digest" => config.digest,
            "schema_version" => config.schema_version,
            "semantics_version" => config.semantics_version,
            "models" => config.models.keys,
            "train" => config.train,
            "parameters" => config.parameters.map(&:qualified_path)
          },
          dataset:, runtime: runtime, optimizer:, seed:)
    end

    # The build this ran on. The engine's half is asked for only when the
    # extension is there, so a pure-Ruby run still has a provenance.
    def self.runtime
      info = { "torobi" => Torobi::VERSION, "ruby" => RUBY_VERSION, "platform" => RUBY_PLATFORM }
      if Torobi.const_defined?(:Native) && Torobi::Native.respond_to?(:build_info)
        info["engine"] = Torobi::Native.build_info
      end
      info
    end

    # A digest of data, so that a record names what it was fed without
    # holding it.
    #
    # Framed: each part contributes its length before its bytes, so that
    # ("ab", "c") and ("a", "bc") do not agree, which plain concatenation
    # let them do. Hashes are serialized with their keys sorted, so a
    # digest depends on the data and not on the order it was written in.
    def self.digest_of(*parts)
      digest = Digest::SHA256.new
      digest << "torobi/1\n"
      parts.each do |part|
        bytes = part.is_a?(String) ? part : JSON.generate(IR::Json.canonical(part))
        digest << "#{bytes.bytesize}:"
        digest << bytes
      end
      digest.hexdigest
    end

    def self.from_h(h)
      new(config: h.fetch("config"), dataset: h["dataset"], runtime: h.fetch("runtime"),
          optimizer: h["optimizer"], seed: h["seed"])
    end

    def initialize(config:, runtime:, dataset: nil, optimizer: nil, seed: nil)
      optimizer = optimizer&.to_h { |k, v| [k.to_s, v.is_a?(Symbol) ? v.to_s : v] }
      super(config: Freeze.deep(config), dataset: Freeze.deep(dataset),
            runtime: Freeze.deep(runtime), optimizer: Freeze.deep(optimizer), seed:)
    end

    def to_h
      { "config" => config, "dataset" => dataset, "runtime" => runtime,
        "optimizer" => optimizer, "seed" => seed }.compact
    end
  end
end
