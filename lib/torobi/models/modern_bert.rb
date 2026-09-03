# frozen_string_literal: true

module Torobi
  module Models
    # ModernBERT, as a GraphConfig.
    #
    # The first architecture Torobi describes, and the one its first target
    # is made of: ruri-v3 is a ModernBERT encoder, and the reranker being
    # distilled is another (docs/plan.md section 3).
    #
    # Written against the published configuration rather than a fixed set
    # of numbers, so `from_config_file` reads a checkpoint's `config.json`
    # and this builds what that says. What the file calls a thing, this
    # calls it too: the parameter paths are the checkpoint's own names, so
    # a model imports with `weights_file:` and no renaming
    # (`Session.open(config, weights_file: "model.safetensors")`).
    #
    # What it is not: a training recipe, a tokenizer, or a pooler. Those
    # are the caller's, and keeping them out is what makes this a
    # description rather than a framework inside a framework.
    module ModernBERT
      # What a ModernBERT config says, with the defaults the reference
      # implementation uses when a file leaves one out.
      Config = Data.define(
        :vocab_size, :hidden_size, :intermediate_size, :num_hidden_layers,
        :num_attention_heads, :norm_eps, :global_attn_every_n_layers,
        :global_rope_theta, :local_rope_theta, :local_attention
      ) do
        def head_dim = hidden_size / num_attention_heads

        # Layer 0 attends globally, and every nth after it; the rest are
        # local. The reference counts from zero, so layer 0 is global.
        def global?(layer) = (layer % global_attn_every_n_layers).zero?

        def theta(layer) = global?(layer) ? global_rope_theta : local_rope_theta

        def check!
          unless (hidden_size % num_attention_heads).zero?
            raise ConfigError,
                  "hidden_size #{hidden_size} does not divide into " \
                  "#{num_attention_heads} heads"
          end
          self
        end
      end

      module_function

      # Reads a checkpoint's `config.json`.
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
          norm_eps: raw.fetch("norm_eps", 1e-5),
          global_attn_every_n_layers: raw.fetch("global_attn_every_n_layers", 3),
          global_rope_theta: raw.fetch("global_rope_theta", 160_000.0),
          local_rope_theta: raw.fetch("local_rope_theta", 10_000.0),
          local_attention: raw.fetch("local_attention", 128)
        ).check!
      end

      # The encoder as a graph: token ids in, hidden states out.
      #
      # `seq` is the sequence length this graph is built for. It is fixed
      # rather than symbolic because the head split needs a concrete
      # dimension to reshape into; the batch stays symbolic, which is the
      # one that varies between steps.
      #
      # `mask:` names a batch field holding the additive attention mask,
      # [batch, 1, seq, seq], zero where a position may be attended to and
      # a large negative where it may not. Padding and the sliding window
      # are both expressed there, by whoever owns the data: the graph says
      # attention, not which positions this batch happens to have.
      def graph(config, seq:, mask: "mask")
        config.check!
        Torobi.graph do |g|
          ids = g.input(:input_ids, [nil, seq], dtype: :i32)
          attention = g.input(mask.to_sym, [nil, 1, seq, seq])

          x = g.embedding(ids, vocab: config.vocab_size, dim: config.hidden_size,
                          name: "embeddings.tok_embeddings")
          x = norm(g, x, config, name: "embeddings.norm")

          config.num_hidden_layers.times do |i|
            x = g.scope("layers.#{i}") { layer(g, x, config, i, attention, seq:) }
          end

          g.output :hidden, norm(g, x, config, name: "final_norm")
        end
      end

      # One encoder block: attention with a residual, then the MLP with
      # another. Layer 0 has no attention norm, which is the reference's
      # own asymmetry and not a mistake here.
      def layer(g, x, config, index, mask, seq:)
        normed = index.zero? ? x : norm(g, x, config, name: "attn_norm")
        x += attention(g, normed, config, index, mask, seq:)
        x + mlp(g, norm(g, x, config, name: "mlp_norm"), config)
      end

      def attention(g, x, config, index, mask, seq:)
        heads = config.num_attention_heads
        dim = config.head_dim
        theta = config.theta(index)
        q, k, v = g.scope("attn") do
          g.linear(x, config.hidden_size * 3, name: "Wqkv", bias: false).split(3, axis: -1)
        end
        # [batch, seq, hidden] -> [batch, heads, seq, head_dim], so
        # attention runs per head and the batch stays symbolic.
        to_heads = lambda do |h|
          h.reshape(shape: [-1, seq, heads, dim]).transpose(axes: [0, 2, 1, 3])
        end
        attended = g.sdpa(to_heads.call(q).rope(theta:),
                          to_heads.call(k).rope(theta:),
                          to_heads.call(v),
                          mask:)
        folded = attended.transpose(axes: [0, 2, 1, 3])
                         .reshape(shape: [-1, seq, config.hidden_size])
        g.scope("attn") { g.linear(folded, config.hidden_size, name: "Wo", bias: false) }
      end

      # GeGLU: one projection to twice the width, gelu on the first half,
      # gated by the second, then back down.
      def mlp(g, x, config)
        g.scope("mlp") do
          wide = g.linear(x, config.intermediate_size * 2, name: "Wi", bias: false)
          gate, up = wide.split(2, axis: -1)
          g.linear(gate.gelu * up, config.hidden_size, name: "Wo", bias: false)
        end
      end

      # Every norm here is a LayerNorm with a gain and no bias, which is
      # what `norm_bias: false` in the published configs means.
      def norm(g, x, config, name:)
        g.layer_norm(x, name:, bias: false, eps: config.norm_eps)
      end
    end
  end
end
