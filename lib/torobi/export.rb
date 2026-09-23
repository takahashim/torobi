# frozen_string_literal: true

require "json"
require "fileutils"

module Torobi
  # What a trained model needs around its weights to be a model somebody
  # else can load (docs/plan.md section 14).
  #
  # The engine writes `model.safetensors`. A sentence-transformers
  # checkpoint is that file plus a tokenizer, the transformer's config, and
  # three small descriptions of how the pieces fit. Without the tokenizer
  # there is nothing to turn text into ids, so a directory holding only
  # weights and metadata does not load at all, however correct the
  # metadata is.
  #
  # **So the source model is copied rather than reinvented.** Fine-tuning
  # starts from a published checkpoint, and everything except the weights
  # is still true of the result: the tokenizer, the vocabulary, the
  # sequence length, and how the pooling was configured. Writing those from
  # scratch means writing what this code happens to know, and what it
  # happens to know is not everything: ruri-v3's pooling config carries
  # `include_prompt`, which decides whether the task prefix is inside the
  # average that becomes the vector.
  #
  #   Export.new("out/ruri-ft", from: "ruri-v3-130m", pooling: :mean).publish
  #
  # One export is one directory the engine has already written its weights
  # into, and one source to carry from. Held as that, so each step below
  # asks the export rather than being handed its pieces.
  class Export
    # The files a sentence-transformers model is made of, apart from its
    # weights. Named rather than "everything in the directory": a source
    # may also hold `pytorch_model.bin`, which is the same weights before
    # training and must not travel with the new ones, and a README that
    # describes a model this is no longer.
    CARRIED = %w[
      config.json
      sentence_bert_config.json
      config_sentence_transformers.json
      modules.json
      tokenizer.json
      tokenizer_config.json
      tokenizer.model
      special_tokens_map.json
      vocab.txt
      vocab.json
      merges.txt
      spiece.model
    ].freeze

    # Where the pooling module's own configuration lives.
    POOLING = File.join("1_Pooling", "config.json").freeze

    # What the engine writes the weights as (`state.rs`). Named here
    # because everything in this file is about what sits beside them.
    WEIGHTS = "model.safetensors"

    # How the pooled vector is made, when the caller says: `mode` (:cls or
    # :mean) and `dim`, the width it pools. Either may be left out, and
    # then what the source said stands.
    class Pooling < Data.define(:mode, :dim)
      MODES = { cls: "pooling_mode_cls_token", mean: "pooling_mode_mean_tokens" }.freeze

      # A config nobody carried starts from every mode off.
      BLANK = {
        "pooling_mode_cls_token" => false,
        "pooling_mode_mean_tokens" => false,
        "pooling_mode_max_tokens" => false,
        "pooling_mode_mean_sqrt_len_tokens" => false,
        "pooling_mode_weightedmean_tokens" => false,
        "pooling_mode_lasttoken" => false
      }.freeze

      def initialize(mode: nil, dim: nil)
        if mode && !MODES.key?(mode)
          raise ArgumentError, "pooling must be one of #{MODES.keys.inspect}, got #{mode.inspect}"
        end

        super
      end

      # Whether the caller said anything about pooling at all.
      def given? = !(mode.nil? && dim.nil?)

      # The pooling module's configuration, starting from whatever was
      # carried so that keys this does not know about survive.
      #
      # `include_prompt` is the one that matters and the one most easily
      # lost: it says whether the task prefix is part of the average, which
      # is a property of how the source model was trained rather than a
      # choice being made here.
      def config(carried)
        config = (carried || BLANK).dup
        config["word_embedding_dimension"] = dim if dim
        unless config["word_embedding_dimension"]
          raise ArgumentError, "pooling_dim: is required when nothing carries one"
        end

        if mode
          MODES.each_value { |flag| config[flag] = false }
          config[MODES.fetch(mode)] = true
        end
        config
      end
    end

    # The one model this run holds, or a refusal when it holds several.
    def self.sole_model(paths)
      names = paths.map { |path| path.split(".").first }.uniq
      return names.first if names.size == 1

      raise ArgumentError, "model: is required (this run has #{names.inspect})"
    end

    # LoRA leaves the weight alone and trains a pair of small matrices
    # next to it, so an adapted run holds the base weights it started
    # with and never moved. Exporting writes every parameter, which would
    # be those weights plus `lora_A` and `lora_B` tensors nothing else
    # knows: **a directory that looks trained and is not.** Merging them
    # (W + scale * BA, and the adapter dropped) is what makes one, and it
    # is not built yet. Until it is, this refuses rather than writes
    # something misleading; the checkpoint holds both halves, so nothing
    # is lost by waiting.
    #
    # Before the weights are written, so a refusal leaves the directory
    # as it was.
    def self.refuse_adapted!(model, paths)
      adapted = paths.select do |path|
        path.start_with?("#{model}.") && LoRA.adapted?(path)
      end
      return if adapted.empty?

      raise ConfigError,
            "#{model.inspect} was trained through an adapter (#{adapted.size} " \
            "parameters like #{adapted.first.inspect}), and its base weights have " \
            "not moved. Writing them would be a model that looks trained and is " \
            "not. Merging the adapter into the weights is not implemented; the " \
            "checkpoint holds both halves."
    end

    def initialize(dir, from: nil, pooling: nil, pooling_dim: nil)
      @dir = dir.to_s
      @from = from&.to_s
      @pooling = Pooling.new(mode: pooling, dim: pooling_dim)
    end

    attr_reader :dir, :from, :pooling

    # Everything that goes beside the weights, in the order it has to
    # happen. Returns what was carried from the source.
    #
    # **The order is why this is one call rather than three.** The widths
    # have to be read before anything is written, because what they
    # answer is whether the descriptions about to be written are about
    # these weights; and carrying has to happen before the metadata,
    # which starts from whatever pooling config the source had so that
    # keys this code knows nothing about survive.
    def publish
      seen = widths
      carried = carry
      write_metadata(seen)
      carried
    end

    private

    def path(name) = File.join(@dir, name)

    # Copies what the source holds beside the weights, and returns the
    # names copied. No source copies nothing, which is what a model
    # trained from nothing has to do.
    def carry
      return [] unless @from
      raise ArgumentError, "from: #{@from.inspect} is not a directory" unless File.directory?(@from)

      copied = CARRIED.select { |name| File.file?(File.join(@from, name)) }
      copied.each { |name| FileUtils.cp(File.join(@from, name), path(name)) }
      if File.file?(File.join(@from, POOLING))
        FileUtils.mkdir_p(File.dirname(path(POOLING)))
        FileUtils.cp(File.join(@from, POOLING), path(POOLING))
        copied << POOLING
      end
      copied
    end

    # The pooling configuration already in the directory, or nil.
    def carried_pooling
      JSON.parse(File.read(path(POOLING)))
    rescue SystemCallError, JSON::ParserError
      nil
    end

    # Writes the descriptions that are not carried, and overrides the
    # pooling when the caller asks for one.
    #
    # Everything is built and checked before anything is written. What is
    # already there was carried from the source and is the source's to be
    # right about; what this writes is this code's, and it should not write
    # something it can see is wrong.
    def write_metadata(widths)
      carried = carried_pooling
      # Before the pooling config, which refuses a missing dimension: a
      # source that belongs to another model is worth saying so about,
      # rather than reporting whatever else is wrong downstream of it.
      check_widths(carried, widths)
      modules = File.exist?(path("modules.json")) ? nil : modules_json
      pooling = @pooling.config(carried) if @pooling.given? || carried.nil?
      described = File.exist?(path("config_sentence_transformers.json"))

      FileUtils.mkdir_p(@dir)
      File.write(path("modules.json"), modules) if modules
      if pooling
        FileUtils.mkdir_p(File.dirname(path(POOLING)))
        File.write(path(POOLING), JSON.pretty_generate(pooling))
      end
      return if described

      File.write(path("config_sentence_transformers.json"),
                 sentence_transformers_json)
    end

    # That the descriptions in the directory are about its weights.
    #
    # Two ways to be wrong, and neither is an error anywhere else: a
    # `pooling_dim` the model does not have makes a pooling layer of the
    # wrong shape, and a source model that is not the one these weights
    # came from carries a tokenizer for another vocabulary. Both are
    # visible in one number, because the hidden state's width is a width
    # the tensors have.
    #
    # Before anything is written, so that a refusal leaves the weights and
    # whatever was carried, and nothing this code made up.
    def check_widths(carried, widths)
      return if widths.empty?

      # The carried config only when nothing was given, because a given
      # dimension is what overrides it: two complaints about a number the
      # caller has already replaced is one complaint too many.
      claims = { "the given pooling_dim" => @pooling.dim,
                 "the source's config.json" => hidden_size }
      unless @pooling.dim
        claims["the carried pooling config"] =
          carried&.dig("word_embedding_dimension")
      end

      wrong = claims.reject { |_, dim| dim.nil? || widths.include?(dim) }
      return if wrong.empty?

      said = wrong.map { |what, dim| "#{what} says #{dim}" }.join(", ")
      raise ArgumentError,
            "#{said}, which is not a width these weights have " \
            "(#{widths.to_a.sort.inspect}). The pooled vector is as wide as the " \
            "model's hidden state, so this is either the wrong pooling_dim or the " \
            "wrong `from:`. model.safetensors and anything carried are in #{@dir}; " \
            "no pooling config was written."
    end

    # What the transformer's own config calls its hidden state, under
    # whichever of the three names its family uses.
    def hidden_size
      config = JSON.parse(File.read(path("config.json")))
      config.values_at("hidden_size", "d_model", "n_embd").compact.first
    rescue SystemCallError, JSON::ParserError
      nil
    end

    # Every width the tensors in the weights have, as a Set of trailing
    # dimensions.
    #
    # Enough to answer one question: is the pooled vector's width a width
    # this model actually has? Reading the header is all it takes, and the
    # header is the first thing in the file: eight bytes of length, then
    # that many bytes of JSON.
    #
    # A file that cannot be read raises rather than answering "no widths".
    # The engine reads an export back before returning (docs/plan.md
    # section 15.27), so a file that gets here is one that loaded; an empty
    # answer means every tensor is a scalar, which is a fact about the
    # model rather than a failure to look.
    def widths
      header = File.open(path(WEIGHTS), "rb") do |io|
        JSON.parse(io.read(io.read(8).unpack1("Q<")))
      end
      header.except("__metadata__")
            .values.filter_map { |entry| Array(entry["shape"]).last }
            .to_set
    end

    # The two modules a plain encoder is: the transformer, and the pooling
    # that turns its tokens into one vector.
    def modules_json
      JSON.pretty_generate([
                             { "idx" => 0, "name" => "0", "path" => "",
                               "type" => "sentence_transformers.models.Transformer" },
                             { "idx" => 1, "name" => "1", "path" => "1_Pooling",
                               "type" => "sentence_transformers.models.Pooling" }
                           ])
    end

    def sentence_transformers_json
      JSON.pretty_generate({
        "__version__" => { "torobi" => Torobi::VERSION },
        "prompts" => {},
        "default_prompt_name" => nil,
        "similarity_fn_name" => "cosine"
      })
    end
  end
end
