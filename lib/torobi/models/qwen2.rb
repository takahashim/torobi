# frozen_string_literal: true

module Torobi
  module Models
    # Qwen2, as a GraphConfig.
    #
    # The second architecture Torobi describes, and the first decoder
    # (docs/plan.md section 15.49). ModernBERT reads a sequence; this
    # continues one, which changes four things and nothing else: attention
    # only looks backwards, the key heads are fewer than the query heads,
    # the MLP is gated with SiLU rather than GELU, and the output
    # projection is usually the embedding table read a second time.
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
    module Qwen2
      # What a Qwen2 config says, with the defaults the reference uses
      # when a file leaves one out.
      Config = Data.define(
        :vocab_size, :hidden_size, :intermediate_size, :num_hidden_layers,
        :num_attention_heads, :num_key_value_heads, :rms_norm_eps, :rope_theta,
        :tie_word_embeddings, :pad_token_id
      ) do
        def head_dim = hidden_size / num_attention_heads

        # How many query heads share one key head. Grouped-query
        # attention: 14 heads over 2 in Qwen2.5-0.5B, so seven of them
        # read the same keys and the checkpoint carries a seventh of the
        # keys it otherwise would.
        def group = num_attention_heads / num_key_value_heads

        def check!
          unless (hidden_size % num_attention_heads).zero?
            raise ConfigError,
                  "hidden_size #{hidden_size} does not divide into " \
                  "#{num_attention_heads} heads"
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

      def from_hash(raw)
        Config.new(
          vocab_size: raw.fetch("vocab_size"),
          hidden_size: raw.fetch("hidden_size"),
          intermediate_size: raw.fetch("intermediate_size"),
          num_hidden_layers: raw.fetch("num_hidden_layers"),
          num_attention_heads: raw.fetch("num_attention_heads"),
          num_key_value_heads: raw.fetch("num_key_value_heads"),
          rms_norm_eps: raw.fetch("rms_norm_eps", 1e-6),
          rope_theta: raw.fetch("rope_theta", 10_000.0),
          tie_word_embeddings: raw.fetch("tie_word_embeddings", false),
          # A decoder is trained on what it should say next, and the
          # positions it should say nothing at are dropped by the
          # objective rather than attended around. What pads them only has
          # to be a token id, and the end-of-text one is the one to hand.
          pad_token_id: raw["pad_token_id"] || raw.fetch("eos_token_id", 0)
        ).check!
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
      def causal_lm(config, seq:, rows: nil, adapter: nil)
        config.check!
        Torobi.graph do |g|
          hidden = g.adapting(adapter) do
            g.name("hidden", g.scope("model") { encode(g, config, seq:, rows:) })
          end
          # Not named: an untied head is a `linear`, which names its own
          # node after its parameters, and `forward` reaches an output by
          # the name the output has.
          g.output :logits, head(g, hidden, config)
        end
      end

      # The output projection, which is usually not a parameter of its own.
      #
      # **Tied weights are one parameter read twice**, not two that are
      # kept equal: the table that turns an id into a vector turns a
      # vector back into scores over ids, transposed. Qwen2.5-0.5B holds
      # no `lm_head.weight` at all for that reason, and a graph that
      # declared one would be asking a checkpoint for something it does
      # not have.
      def head(g, hidden, config)
        unless config.tie_word_embeddings
          return g.linear(hidden, config.vocab_size, name: "lm_head", bias: false)
        end

        table = g.parameter("model.embed_tokens.weight")
        g.matmul(hidden, table.transpose(axes: [1, 0]))
      end

      # The decoder body: ids in, hidden states out, under `model.` as the
      # checkpoint has it.
      def encode(g, config, seq:, rows: nil)
        ids = g.input(:input_ids, [rows, seq], dtype: :i32)
        x = g.embedding(ids, vocab: config.vocab_size, dim: config.hidden_size,
                        name: "embed_tokens")
        config.num_hidden_layers.times do |i|
          x = g.scope("layers.#{i}") { layer(g, x, config, seq:) }
        end
        norm(g, x, config, name: "norm")
      end

      # One block: attention with a residual, then the MLP with another,
      # each normalized before rather than after (pre-norm).
      def layer(g, x, config, seq:)
        x += attention(g, norm(g, x, config, name: "input_layernorm"), config, seq:)
        x + mlp(g, norm(g, x, config, name: "post_attention_layernorm"), config)
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
      # q, k and v carry biases, which is Qwen2's own asymmetry: the
      # output projection has none.
      def attention(g, x, config, seq:)
        heads = config.num_attention_heads
        kv = config.num_key_value_heads
        dim = config.head_dim
        theta = config.rope_theta
        q = g.linear(x, heads * dim, name: "self_attn.q_proj")
        k = g.linear(x, kv * dim, name: "self_attn.k_proj")
        v = g.linear(x, kv * dim, name: "self_attn.v_proj")
        # [batch, seq, heads * dim] -> [batch, heads, seq, dim]. The key
        # heads are fewer and stay that way: the backend takes them
        # untiled, which is the whole saving.
        to_heads = lambda do |h, count|
          h.reshape(shape: [-1, seq, count, dim]).transpose(axes: [0, 2, 1, 3])
        end
        attended = g.sdpa(to_heads.call(q, heads).rope(theta:),
                          to_heads.call(k, kv).rope(theta:),
                          to_heads.call(v, kv),
                          causal: true)
        folded = attended.transpose(axes: [0, 2, 1, 3])
                         .reshape(shape: [-1, seq, heads * dim])
        g.linear(folded, config.hidden_size, name: "self_attn.o_proj", bias: false)
      end

      # SwiGLU: one projection gated by another, through SiLU, then back
      # down. SiLU is `x * sigmoid(x)`, which is two ops here rather than
      # one; it is what the name means, and nothing is fused away.
      def mlp(g, x, config)
        gate = g.linear(x, config.intermediate_size, name: "mlp.gate_proj", bias: false)
        up = g.linear(x, config.intermediate_size, name: "mlp.up_proj", bias: false)
        g.linear((gate * gate.sigmoid) * up, config.hidden_size,
                 name: "mlp.down_proj", bias: false)
      end

      # Every norm here is an RMS norm with a gain and no bias.
      def norm(g, x, config, name:)
        g.rms_norm(x, name:, eps: config.rms_norm_eps)
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
