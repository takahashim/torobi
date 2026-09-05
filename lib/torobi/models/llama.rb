# frozen_string_literal: true

module Torobi
  module Models
    # The Llama-shaped decoder, as a GraphConfig.
    #
    # Llama itself, and everything built to its shape: Qwen2, sarashina2.2,
    # Mistral without its sliding window. One description rather than one
    # per name, because they are one architecture and the differences
    # between them are things their configs already say (docs/plan.md
    # section 15.53). The tensor names are identical, which is the
    # strongest evidence that this is so.
    #
    # ModernBERT reads a sequence; this continues one, which changes four
    # things and nothing else: attention only looks backwards, the key
    # heads are fewer than the query heads, the MLP is gated with SiLU
    # rather than GELU, and the output projection is sometimes the
    # embedding table read a second time.
    #
    # Written against the published configuration, and named the way the
    # checkpoint names things, so a model imports with no renaming:
    # `pretrained: { m: "Qwen2.5-0.5B/model.safetensors" }`.
    #
    # What it is not: a tokenizer, a training recipe, or a way to generate
    # text. **There is no KV cache and no sampling loop here**, and there
    # is not going to be: a decoder in Torobi is a decoder being
    # fine-tuned, and what serves the result is llama.cpp or vLLM
    # (docs/plan.md section 14). One forward is `Session#forward`.
    module Llama
      # What one of these configs says, with the defaults the reference
      # uses when a file leaves one out.
      Config = Data.define(
        :vocab_size, :hidden_size, :intermediate_size, :num_hidden_layers,
        :num_attention_heads, :num_key_value_heads, :head_dim, :rms_norm_eps,
        :rope_theta, :tie_word_embeddings, :attention_bias, :pad_token_id
      ) do
        # How many query heads share one key head. Grouped-query
        # attention: 14 heads over 2 in Qwen2.5-0.5B, so seven of them
        # read the same keys and the checkpoint carries a seventh of the
        # keys it otherwise would.
        def group = num_attention_heads / num_key_value_heads

        def check!
          unless head_dim * num_attention_heads == hidden_size
            raise ConfigError,
                  "#{num_attention_heads} heads of #{head_dim} is not " \
                  "hidden_size #{hidden_size}"
          end
          unless (num_attention_heads % num_key_value_heads).zero?
            raise ConfigError,
                  "#{num_attention_heads} query heads do not divide into " \
                  "#{num_key_value_heads} key heads"
          end
          self
        end
      end

      module_function

      def from_config_file(path)
        require "json"
        from_hash(JSON.parse(File.read(path.to_s)))
      end

      # Reads one of these configs, whichever of the family it is.
      #
      # Two things differ between them and neither is optional to get
      # right. **Qwen2 puts a bias on q, k and v and says so nowhere**:
      # its config has no `attention_bias` because the implementation
      # hardcodes it, while Llama's says false and means it. And whether
      # the output projection is the embedding table is `tie_word_embeddings`,
      # which decides whether the checkpoint holds an `lm_head.weight` at
      # all. So the family is read off `model_type` and the rest is read
      # off the file.
      def from_hash(raw)
        family = raw.fetch("model_type", "llama")
        refuse_what_is_not_built(raw, family)
        heads = raw.fetch("num_attention_heads")
        Config.new(
          vocab_size: raw.fetch("vocab_size"),
          hidden_size: raw.fetch("hidden_size"),
          intermediate_size: raw.fetch("intermediate_size"),
          num_hidden_layers: raw.fetch("num_hidden_layers"),
          num_attention_heads: heads,
          num_key_value_heads: raw.fetch("num_key_value_heads", heads),
          head_dim: raw["head_dim"] || (raw.fetch("hidden_size") / heads),
          rms_norm_eps: raw.fetch("rms_norm_eps", 1e-6),
          rope_theta: raw.fetch("rope_theta", 10_000.0),
          tie_word_embeddings: raw.fetch("tie_word_embeddings", false),
          attention_bias: raw.fetch("attention_bias", family == "qwen2"),
          # A decoder is trained on what it should say next, and the
          # positions it should say nothing at are dropped by the
          # objective rather than attended around. What pads them only has
          # to be a token id, and the end-of-text one is the one to hand.
          pad_token_id: raw["pad_token_id"] || raw.fetch("eos_token_id", 0)
        ).check!
      end

      # What this family does that this description does not.
      #
      # Refused rather than ignored. A rope that should have been scaled
      # and was not, or a window that should have been slid and was not,
      # is a model that runs, trains, and is not the model the file names;
      # nothing downstream would notice.
      def refuse_what_is_not_built(raw, family)
        if raw["rope_scaling"]
          raise ConfigError,
                "#{family}: rope_scaling #{raw["rope_scaling"].inspect} is not " \
                "implemented here, and a rotary embedding that should have been " \
                "scaled and was not is a different model"
        end
        return unless raw["use_sliding_window"] || (family == "mistral" && raw["sliding_window"])

        raise ConfigError,
              "#{family}: sliding window attention is not implemented here; " \
              "this builds a decoder that attends to everything before it"
      end

      # The language model: token ids in, one score per token of the
      # vocabulary out, at every position.
      #
      # The loss is not here. What to predict (the next token, or the
      # answer and not the question), which positions count and how they
      # are weighed is the recipe's, and it is a `cross_entropy` and a
      # weighted mean away:
      #
      #   objective = Torobi.objective(m: model) do |g|
      #     logits = g.from_model(:m, :logits)
      #     targets = g.from_batch(:targets, [nil, seq], dtype: :i32)
      #     kept = g.from_batch(:kept, [nil, seq])
      #     at = g.cross_entropy(logits, targets) * kept
      #     g.output :loss, g.sum(at) / g.sum(kept)
      #   end
      #
      # `rows:` fixes how many rows a batch has, and is normally left
      # alone; it is for an objective that reads across the batch rather
      # than down it.
      # `adapter:` is a `Torobi::LoRA`, and adapts the linears it names
      # rather than this description having to know about it.
      # `dtype:` is what the model is held in. bf16 halves it, which is
      # what a published checkpoint is stored in anyway; the loss is read
      # as f32, so an objective over a bf16 model says where it comes
      # back (`g.cast(logits, :f32)`).
      def causal_lm(config, seq:, rows: nil, adapter: nil, dtype: :f32)
        config.check!
        build = Build.new(seq:, rows:, dtype:)
        Torobi.graph { |g| Describe.new(g, config, build).causal_lm(adapter) }
      end

      # The description itself: the builder, the config and what it is
      # built for are held once (`Models::Description`), so what a method
      # takes is the value flowing through it.
      class Describe < Description
        def causal_lm(adapter)
          hidden = adapting adapter do
            encoded = scope "model" do
              encode
            end
            name "hidden", encoded
          end
          # Not named: an untied head is a `linear`, which names its own
          # node after its parameters, and `forward` reaches an output by
          # the name the output has.
          output :logits, head(hidden)
        end

        # The output projection, which is usually not a parameter of its
        # own.
        #
        # **Tied weights are one parameter read twice**, not two that are
        # kept equal: the table that turns an id into a vector turns a
        # vector back into scores over ids, transposed. Qwen2.5-0.5B holds
        # no `lm_head.weight` at all for that reason, and a graph that
        # declared one would be asking a checkpoint for something it does
        # not have.
        def head(hidden)
          unless @config.tie_word_embeddings
            return linear(hidden, @config.vocab_size, name: "lm_head", bias: false)
          end

          table = parameter("model.embed_tokens.weight")
          matmul(hidden, table.transpose(axes: [1, 0]))
        end

        # The decoder body: ids in, hidden states out, under `model.` as
        # the checkpoint has it.
        def encode
          ids = input(@build.field(:input_ids), [@build.rows, @build.seq], dtype: :i32)
          x = embedding(ids, vocab: @config.vocab_size, dim: @config.hidden_size,
                        name: "embed_tokens", dtype: @build.dtype)
          @config.num_hidden_layers.times do |i|
            x = scope "layers.#{i}" do
              layer(x)
            end
          end
          norm(x, name: "norm")
        end

        # One block: attention with a residual, then the MLP with another,
        # each normalized before rather than after (pre-norm).
        def layer(x)
          x += attention(norm(x, name: "input_layernorm"))
          x + mlp(norm(x, name: "post_attention_layernorm"))
        end

        # Attention that only looks backwards, over fewer keys than queries.
        #
        # `causal: true` rather than a mask in the batch: the triangle
        # depends on nothing but the sequence length, and handing over
        # [1, 1, seq, seq] of the same number every step is megabytes to say
        # what the attention already knows.
        #
        # There is no padding mask either, and that is a claim rather than
        # an omission: rows are padded on the right, so a real position is
        # never after a padded one and never attends to it. What the padded
        # positions themselves produce is discarded by the objective.
        #
        # Whether q, k and v carry biases is the family's own asymmetry:
        # Qwen2 has them and Llama does not. The output projection has none
        # either way.
        def attention(x)
          heads = @config.num_attention_heads
          kv = @config.num_key_value_heads
          dim = @config.head_dim
          theta = @config.rope_theta
          bias = @config.attention_bias
          scope "self_attn" do
            q = linear(x, heads * dim, name: "q_proj", bias:)
            k = linear(x, kv * dim, name: "k_proj", bias:)
            v = linear(x, kv * dim, name: "v_proj", bias:)
            # The key heads are fewer and stay that way: the backend takes
            # them untiled, which is the whole saving.
            attended = sdpa(q.split_heads(heads).rope(theta:),
                            k.split_heads(kv).rope(theta:),
                            v.split_heads(kv),
                            causal: true)
            linear(attended.merge_heads, @config.hidden_size, name: "o_proj", bias: false)
          end
        end

        # SwiGLU: one projection gated by another, through SiLU, then back
        # down. SiLU is `x * sigmoid(x)`, which is two ops here rather than
        # one; it is what the name means, and nothing is fused away.
        def mlp(x)
          scope "mlp" do
            gate = linear(x, @config.intermediate_size, name: "gate_proj", bias: false)
            up = linear(x, @config.intermediate_size, name: "up_proj", bias: false)
            linear((gate * gate.sigmoid) * up, @config.hidden_size,
                   name: "down_proj", bias: false)
          end
        end

        # Every norm here is an RMS norm with a gain and no bias.
        def norm(x, name:)
          rms_norm(x, name:, eps: @config.rms_norm_eps)
        end
      end

      # One batch, from token ids.
      #
      # Ids and nothing else: a causal attention needs no mask, and what
      # a position should have said next is the recipe's to decide (see
      # `causal_lm`). Rows are padded on the right, which is what makes
      # the missing padding mask sound.
      def batch(config, rows, seq:)
        too_long = rows.each_with_index.select { |row, _| row.size > seq }
        unless too_long.empty?
          row, at = too_long.first
          raise ConfigError,
                "row #{at} has #{row.size} tokens and this graph was built for #{seq}. " \
                "Tokenize to at most #{seq}, or build the graph for a longer sequence; " \
                "where to cut a long text is the caller's to decide."
        end

        ids = rows.flat_map { |row| row + Array.new(seq - row.size, config.pad_token_id) }
        { input_ids: TensorData.from_a([rows.size, seq], ids, dtype: :i32) }
      end
    end
  end
end
