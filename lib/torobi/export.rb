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
  module Export
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

    module_function

    # Copies what `from` holds beside the weights, and returns the names
    # copied. `nil` copies nothing, which is what a model trained from
    # nothing has to do.
    def carry(from, into)
      return [] unless from

      from = from.to_s
      raise ArgumentError, "from: #{from.inspect} is not a directory" unless File.directory?(from)

      copied = CARRIED.select { |name| File.file?(File.join(from, name)) }
      copied.each { |name| FileUtils.cp(File.join(from, name), File.join(into, name)) }
      if File.file?(File.join(from, POOLING))
        FileUtils.mkdir_p(File.join(into, File.dirname(POOLING)))
        FileUtils.cp(File.join(from, POOLING), File.join(into, POOLING))
        copied << POOLING
      end
      copied
    end

    # The pooling configuration already in `dir`, or nil.
    def pooling_config(dir)
      JSON.parse(File.read(File.join(dir, POOLING)))
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
    def write_metadata(dir, pooling: nil, pooling_dim: nil, widths: nil)
      dir = dir.to_s
      carried = pooling_config(dir)
      # Before pooling_json, which refuses a missing dimension: a source
      # that belongs to another model is worth saying so about, rather
      # than reporting whatever else is wrong downstream of it.
      check_widths(dir, pooling_dim, carried, widths) if widths
      modules = File.exist?(File.join(dir, "modules.json")) ? nil : modules_json
      asked = pooling || pooling_dim || carried.nil?
      pooling_file = pooling_json(pooling, pooling_dim, carried) if asked
      described = File.exist?(File.join(dir, "config_sentence_transformers.json"))

      FileUtils.mkdir_p(dir)
      File.write(File.join(dir, "modules.json"), modules) if modules
      if pooling_file
        FileUtils.mkdir_p(File.join(dir, File.dirname(POOLING)))
        File.write(File.join(dir, POOLING), pooling_file)
      end
      unless described
        File.write(File.join(dir, "config_sentence_transformers.json"),
                   sentence_transformers_json)
      end
      dir
    end

    # That the descriptions in `dir` are about the weights in `dir`.
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
    def check_widths(dir, pooling_dim, carried, widths)
      return if widths.empty?

      # The carried config only when nothing was given, because a given
      # dimension is what overrides it: two complaints about a number the
      # caller has already replaced is one complaint too many.
      claims = { "the given pooling_dim" => pooling_dim,
                 "the source's config.json" => hidden_size(dir) }
      unless pooling_dim
        claims["the carried pooling config"] = carried&.dig("word_embedding_dimension")
      end

      wrong = claims.reject { |_, dim| dim.nil? || widths.include?(dim) }
      return if wrong.empty?

      said = wrong.map { |what, dim| "#{what} says #{dim}" }.join(", ")
      raise ArgumentError,
            "#{said}, which is not a width these weights have " \
            "(#{widths.to_a.sort.inspect}). The pooled vector is as wide as the " \
            "model's hidden state, so this is either the wrong pooling_dim or the " \
            "wrong `from:`. model.safetensors and anything carried are in #{dir}; " \
            "no pooling config was written."
    end

    # What the transformer's own config calls its hidden state, under
    # whichever of the three names its family uses.
    def hidden_size(dir)
      config = JSON.parse(File.read(File.join(dir, "config.json")))
      config.values_at("hidden_size", "d_model", "n_embd").compact.first
    rescue SystemCallError, JSON::ParserError
      nil
    end

    # Every width the tensors in a safetensors file have, as a Set of
    # trailing dimensions.
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
    def widths(file)
      header = File.open(file, "rb") do |io|
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

    # The pooling module's configuration, starting from whatever was
    # carried so that keys this does not know about survive.
    #
    # `include_prompt` is the one that matters and the one most easily
    # lost: it says whether the task prefix is part of the average, which
    # is a property of how the source model was trained rather than a
    # choice being made here.
    def pooling_json(pooling, dim, carried = nil)
      config = carried || {
        "pooling_mode_cls_token" => false,
        "pooling_mode_mean_tokens" => false,
        "pooling_mode_max_tokens" => false,
        "pooling_mode_mean_sqrt_len_tokens" => false,
        "pooling_mode_weightedmean_tokens" => false,
        "pooling_mode_lasttoken" => false
      }
      config = config.dup
      config["word_embedding_dimension"] = dim if dim
      unless config["word_embedding_dimension"]
        raise ArgumentError, "pooling_dim: is required when nothing carries one"
      end

      if pooling
        mode = MODES.fetch(pooling) do
          raise ArgumentError, "pooling must be one of #{MODES.keys.inspect}, " \
                               "got #{pooling.inspect}"
        end
        MODES.each_value { |flag| config[flag] = false }
        config[mode] = true
      end
      JSON.pretty_generate(config)
    end

    MODES = { cls: "pooling_mode_cls_token", mean: "pooling_mode_mean_tokens" }.freeze

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
