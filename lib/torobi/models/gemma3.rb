# frozen_string_literal: true

module Torobi
  module Models
    # Gemma 3, as a GraphConfig (docs/plan.md section 15.54).
    #
    # Not one of the Llama-shaped family (`Models::Llama`), and worth
    # saying why, because the tensor names look almost the same. Five
    # things differ, and every one of them is the kind that leaves a model
    # that runs, trains, and is not the published one:
    #
    #   1. **four norms a layer**, not two: each block is wrapped rather
    #      than preceded, so the attention output is normalized before it
    #      is added back
    #   2. **q and k are normalized per head** before they are rotated
    #   3. **the norms scale by `1 + w`**, so Gemma's norm weights sit
    #      around zero where everyone else's sit around one
    #   4. **the gate is the tanh approximation of GELU**, which is a
    #      different function from the exact one
    #   5. **the embeddings are scaled by the square root of the width**
    #
    # And one that is structure rather than arithmetic: most layers attend
    # only within a window of the recent past, with a different rotary
    # base from the few that see everything. ModernBERT alternates that
    # way too (`Models::ModernBERT`), which is where the machinery came
    # from.
    #
    # No KV cache and no sampling loop, as everywhere else here: what
    # Torobi does with a decoder is fine-tune it (docs/plan.md section 14).
    module Gemma3
      # What a Gemma 3 text config says.
      Config = Data.define(
        :vocab_size, :hidden_size, :intermediate_size, :num_hidden_layers,
        :num_attention_heads, :num_key_value_heads, :head_dim, :rms_norm_eps,
        :rope_theta, :rope_local_base_freq, :sliding_window, :layer_types,
        :query_pre_attn_scalar, :tie_word_embeddings, :pad_token_id
      ) do
        # Whether this layer sees only the recent past. The config lists
        # them one by one, which is more honest than a pattern: what the
        # model does is the list.
        def sliding?(layer) = layer_types.fetch(layer) == "sliding_attention"

        def any_sliding? = layer_types.include?("sliding_attention")

        # The local layers rotate at a lower base than the global ones, so
        # a window of 512 and a context of 32k are addressed differently.
        def theta(layer) = sliding?(layer) ? rope_local_base_freq : rope_theta

        # Gemma names the number the scores are divided by rather than
        # deriving it from the head width. They are the same here and were
        # not in Gemma 2, so it is read rather than computed.
        def scale = 1.0 / Math.sqrt(query_pre_attn_scalar)

        # What the heads take up, which is not the width of the model:
        # Gemma 3 270m is 4 heads of 256 over a hidden state of 640.
        def attention_size = num_attention_heads * head_dim

        def check!
          unless layer_types.size == num_hidden_layers
            raise ConfigError,
                  "#{layer_types.size} layer types for #{num_hidden_layers} layers"
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

      # Reads a `config.json`, or the `text_config` inside one: the
      # multimodal Gemmas wrap the language model in a config of their
      # own, and what this builds is the language model.
      def from_hash(raw)
        raw = raw.fetch("text_config") if raw.key?("text_config")
        layers = raw.fetch("num_hidden_layers")
        Config.new(
          vocab_size: raw.fetch("vocab_size"),
          hidden_size: raw.fetch("hidden_size"),
          intermediate_size: raw.fetch("intermediate_size"),
          num_hidden_layers: layers,
          num_attention_heads: raw.fetch("num_attention_heads"),
          num_key_value_heads: raw.fetch("num_key_value_heads"),
          head_dim: raw.fetch("head_dim"),
          rms_norm_eps: raw.fetch("rms_norm_eps", 1e-6),
          rope_theta: raw.fetch("rope_theta", 1_000_000.0),
          rope_local_base_freq: raw.fetch("rope_local_base_freq", 10_000.0),
          sliding_window: raw.fetch("sliding_window", 512),
          layer_types: raw["layer_types"] || alternating(raw, layers),
          query_pre_attn_scalar: raw.fetch("query_pre_attn_scalar", raw.fetch("head_dim")),
          # Gemma ties, where Llama does not: the default is the other way
          # round from `Models::Llama` and it matters, because a tied
          # checkpoint holds no `lm_head.weight` at all.
          tie_word_embeddings: raw.fetch("tie_word_embeddings", true),
          pad_token_id: raw["pad_token_id"] || raw.fetch("eos_token_id", 0)
        ).check!
      end

      # Which layers see everything, for a config that gives the pattern
      # rather than the list: every nth, counting so that the last of each
      # run of n is the one.
      def alternating(raw, layers)
        every = raw["sliding_window_pattern"] || raw["_sliding_window_pattern"] || 6
        Array.new(layers) { |i| ((i + 1) % every).zero? ? "full_attention" : "sliding_attention" }
      end

      # The language model: token ids in, one score per token of the
      # vocabulary out, at every position. The loss is the recipe's
      # (`Models::Llama#causal_lm` says what that looks like).
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
          hidden = adapting(adapter) { name("hidden", scope("model") { encode }) }
          output :logits, head(hidden)
        end

        def head(hidden)
          unless @config.tie_word_embeddings
            return linear(hidden, @config.vocab_size, name: "lm_head", bias: false)
          end

          matmul(hidden, parameter("model.embed_tokens.weight").transpose(axes: [1, 0]))
        end

        def encode
          ids = input(@build.field(:input_ids), [@build.rows, @build.seq], dtype: :i32)
          x = embedding(ids, vocab: @config.vocab_size, dim: @config.hidden_size,
                        name: "embed_tokens", dtype: @build.dtype)
          # Gemma scales its embeddings by the square root of the width.
          # Part of the model rather than an initialization detail: leave
          # it out and every layer sees something 25 times too small.
          x *= Math.sqrt(@config.hidden_size)
          # One mask for every local layer, and none for the global ones,
          # which say what they need with `causal:`. It depends on nothing
          # in the batch, so there is one of it however many rows there are.
          window = if @config.any_sliding?
                     cast(input(@build.field(:window),
                                [1, 1, @build.seq, @build.seq]), @build.dtype)
                   end

          @config.num_hidden_layers.times do |i|
            x = scope("layers.#{i}") { layer(x, i, window) }
          end
          norm(x, name: "norm")
        end

        # One block, wrapped rather than preceded: normalized going in, and
        # normalized again on the way out before the residual takes it.
        def layer(x, index, window)
          attended = attention(norm(x, name: "input_layernorm"), index, window)
          x += norm(attended, name: "post_attention_layernorm")
          fed = mlp(norm(x, name: "pre_feedforward_layernorm"))
          x + norm(fed, name: "post_feedforward_layernorm")
        end

        # Attention over fewer keys than queries, normalized per head
        # before it is rotated, and looking back no further than this
        # layer may.
        def attention(x, index, window)
          heads = @config.num_attention_heads
          kv = @config.num_key_value_heads
          dim = @config.head_dim
          theta = @config.theta(index)
          to_heads = lambda do |h, count|
            h.reshape(shape: [-1, @build.seq, count, dim]).transpose(axes: [0, 2, 1, 3])
          end
          q = to_heads.call(linear(x, heads * dim, name: "self_attn.q_proj", bias: false), heads)
          k = to_heads.call(linear(x, kv * dim, name: "self_attn.k_proj", bias: false), kv)
          v = to_heads.call(linear(x, kv * dim, name: "self_attn.v_proj", bias: false), kv)
          q = norm(q, name: "self_attn.q_norm").rope(theta:)
          k = norm(k, name: "self_attn.k_norm").rope(theta:)

          attended =
            if @config.sliding?(index)
              # The window is already a triangle: what a position may see
              # is the recent past, and the past is the causal part of it.
              sdpa(q, k, v, mask: window, scale: @config.scale)
            else
              sdpa(q, k, v, causal: true, scale: @config.scale)
            end
          folded = attended.transpose(axes: [0, 2, 1, 3])
                           .reshape(shape: [-1, @build.seq, @config.attention_size])
          linear(folded, @config.hidden_size, name: "self_attn.o_proj", bias: false)
        end

        # GeGLU with the tanh approximation, which is the function Gemma
        # was trained with.
        def mlp(x)
          gate = linear(x, @config.intermediate_size, name: "mlp.gate_proj", bias: false)
          up = linear(x, @config.intermediate_size, name: "mlp.up_proj", bias: false)
          linear(gate.gelu_tanh * up, @config.hidden_size, name: "mlp.down_proj", bias: false)
        end

        # Every norm here scales by `1 + w`, which is Gemma's own
        # convention and the reason its norm weights are stored around zero.
        def norm(x, name:)
          rms_norm(x, name:, eps: @config.rms_norm_eps, offset: 1.0)
        end
      end

      # A large negative rather than -Infinity, for the reason
      # `Models::ModernBERT` gives: a row that could attend to nothing
      # would softmax to NaN, and a finite floor keeps the arithmetic
      # finite even where the answer is discarded.
      NEGATIVE = -1.0e9

      # One batch, from token ids.
      #
      # Rows are padded on the right, so a real position never attends to
      # a padded one and no padding mask is needed. What the local layers
      # do need is the window, which depends on the sequence length and
      # nothing else.
      def batch(config, rows, seq:)
        too_long = rows.each_with_index.select { |row, _| row.size > seq }
        unless too_long.empty?
          row, at = too_long.first
          raise ConfigError,
                "row #{at} has #{row.size} tokens and this graph was built for #{seq}. " \
                "Tokenize to at most #{seq}, or build the graph for a longer sequence."
        end

        ids = rows.flat_map { |row| row + Array.new(seq - row.size, config.pad_token_id) }
        fields = { input_ids: TensorData.from_a([rows.size, seq], ids, dtype: :i32) }
        return fields unless config.any_sliding?

        fields.merge(window: window(config, seq:))
      end

      # What a local layer may see: itself, and at most `sliding_window`
      # positions back. Causal and windowed in one, because a mask is
      # additive and there is no reason to hand over two.
      #
      # Built once per shape and handed out again: it is the same bytes at
      # every step, and at sequence 512 it is a megabyte.
      # What a shape's window is kept in, bounded (`Models::Windows`).
      WINDOWS = Windows.new

      def window(config, seq:)
        width = config.sliding_window
        WINDOWS.fetch([seq, width]) do
          TensorData.runs(
            [1, 1, seq, seq],
            (0...seq).flat_map do |i|
              first = [i - width + 1, 0].max
              [[first, NEGATIVE], [i - first + 1, 0.0], [seq - 1 - i, NEGATIVE]]
            end
          )
        end
      end
    end
  end
end
