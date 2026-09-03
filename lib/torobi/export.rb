# frozen_string_literal: true

require "json"
require "fileutils"

module Torobi
  # Writes the metadata files that make a safetensors export a sentence-
  # transformers checkpoint (docs/plan.md section 14).
  #
  # The engine writes `model.safetensors`; this writes the files around it:
  # modules.json, 1_Pooling/config.json, and config_sentence_transformers.json.
  # An optional `config.json` for the underlying transformer is copied as-is.
  module Export
    module_function

    # Writes sentence-transformers metadata to `dir`.
    #
    # `pooling` is :cls or :mean. `pooling_dim` is the model's hidden size.
    # `config_json` is the underlying transformer's config.json text, if the
    # caller wants it copied into the export.
    def write_metadata(dir, pooling:, pooling_dim:, config_json: nil)
      dir = dir.to_s
      FileUtils.mkdir_p(dir)
      FileUtils.mkdir_p(File.join(dir, "1_Pooling"))

      File.write(File.join(dir, "modules.json"), modules_json)
      File.write(File.join(dir, "1_Pooling", "config.json"),
                 pooling_json(pooling, pooling_dim))
      File.write(File.join(dir, "config_sentence_transformers.json"),
                 sentence_transformers_json)
      if config_json
        File.write(File.join(dir, "config.json"), config_json)
      end
      dir
    end

    def modules_json
      JSON.pretty_generate([
        { "idx" => 0, "name" => "0", "path" => "",
          "type" => "sentence_transformers.models.Transformer" },
        { "idx" => 1, "name" => "1", "path" => "1_Pooling",
          "type" => "sentence_transformers.models.Pooling" }
      ])
    end

    def pooling_json(pooling, dim)
      flags = {
        "pooling_mode_cls_token" => false,
        "pooling_mode_mean_tokens" => false,
        "pooling_mode_max_tokens" => false,
        "pooling_mode_mean_sqrt_len_tokens" => false,
        "pooling_mode_weightedmean_tokens" => false,
        "pooling_mode_lasttoken" => false
      }
      key = case pooling
            when :cls then "pooling_mode_cls_token"
            when :mean then "pooling_mode_mean_tokens"
            else
              raise ArgumentError, "pooling must be :cls or :mean, got #{pooling.inspect}"
            end
      flags[key] = true
      JSON.pretty_generate({ "word_embedding_dimension" => dim }.merge(flags))
    end

    def sentence_transformers_json
      JSON.pretty_generate({
        "prompts" => {},
        "default_prompt_name" => nil,
        "default_prompt_template" => nil,
        "trust_remote_code" => false
      })
    end
  end
end
