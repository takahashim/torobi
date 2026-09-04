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
        :global_rope_theta, :local_rope_theta, :local_attention, :pad_token_id,
        :num_labels, :classifier_pooling, :classifier_bias
      ) do
        def head_dim = hidden_size / num_attention_heads

        # Layer 0 attends globally, and every nth after it; the rest are
        # local. The reference counts from zero, so layer 0 is global.
        def global?(layer) = (layer % global_attn_every_n_layers).zero?

        def theta(layer) = global?(layer) ? global_rope_theta : local_rope_theta

        # Whether any layer is local. A configuration small enough to have
        # none needs no window, and asking for one it never reads would be
        # a batch field with nothing to do.
        def any_local? = num_hidden_layers.times.any? { |i| !global?(i) }

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
          local_attention: raw.fetch("local_attention", 128),
          pad_token_id: raw.fetch("pad_token_id", 0),
          num_labels: raw.fetch("num_labels", 1),
          classifier_pooling: raw.fetch("classifier_pooling", "cls").to_sym,
          classifier_bias: raw.fetch("classifier_bias", false)
        ).check!
      end

      # A sequence classifier: token ids in, one logit per label out.
      #
      # What the published rerankers are (`ModernBertForSequenceClassification`),
      # and what this project is distilling: a cross-encoder scores a
      # (query, text) pair with one number. The encoder sits under
      # `model.`, as the checkpoint has it, with a pooled head above.
      #
      # The score a caller compares against is the sigmoid of this logit,
      # which is what a CrossEncoder reports for a one-label model. Whether
      # to distil against the logit or its sigmoid is the recipe's, so this
      # emits the logit and stops.
      # `encoder_prefix:` is where the encoder's parameters sit. Published
      # classifiers keep it under `model.`; a classifier built on a bare
      # encoder checkpoint (ruri-v3-130m holds `embeddings.*` at the root)
      # wants none, and its head is `fresh:` because no file has one.
      # `adapter:` is a `Torobi::LoRA`, and adapts the linears it names
      # rather than this description having to know about it. The head is
      # inside it, so a classifier adapted this way trains its adapter
      # and nothing else; a head that has to be trained too is a
      # `fresh:` head on an unadapted model.
      def classifier(config, seq:, encoder_prefix: "model", rows: nil, adapter: nil,
                     dtype: :f32)
        config.check!
        build = Build.new(seq:, rows:, dtype:)
        Torobi.graph do |g|
          g.adapting(adapter) { classify(g, config, build, encoder_prefix) }
        end
      end

      # The classifier's body, so that `adapting` has something to wrap.
      def classify(g, config, build, encoder_prefix)
        x = body(g, config, build, encoder_prefix)
        pooled = pool(g, x, config, build)
        # ModernBERT's head: a dense, gelu, a norm, then the classifier.
        pooled = norm(g, g.linear(pooled, config.hidden_size, name: "head.dense",
                                  bias: false).gelu,
                      config, name: "head.norm")
        # `linear` names the node after the parameter scope, so this is
        # already called "classifier"; the output name is separate.
        g.output :logits, g.linear(pooled, config.num_labels, name: "classifier",
                                   bias: config.classifier_bias)
      end

      # A sentence embedder: token ids in, one vector per row out.
      #
      # What a sentence-transformers checkpoint is, as a graph: the
      # encoder, a pooling, and the normalization that turns a dot product
      # into a cosine. ruri-v3 is this, and `Torobi::Export` writes the
      # `modules.json` and `1_Pooling/config.json` that say so.
      #
      # The vector is named, so a tap reads it without a second graph:
      # "embedding" is what a gradient cache back-propagates through
      # (`Torobi::GradCache`), and "hidden" is what it was pooled from.
      #
      # No loss. A contrastive objective reads across the batch, which
      # makes it the recipe's rather than the model's; this stops at the
      # vector. `rows:` is for the objective that does read across, which
      # cannot be written against a dimension nothing knows.
      def embedder(config, seq:, pooling: :mean, encoder_prefix: "", rows: nil,
                   normalize: true, dtype: :f32)
        config.check!
        build = Build.new(seq:, rows:, dtype:)
        Torobi.graph do |g|
          x = g.name("hidden", body(g, config, build, encoder_prefix))
          pooled = pool(g, x, config, build, mode: pooling)
          pooled = normalized(g, pooled) if normalize
          g.output :embedding, g.name("embedding", pooled)
        end
      end

      # The same embedder, applied to several named sets of rows.
      #
      # One model and one set of weights, in a graph that embeds each side
      # at the length that side actually is. A contrastive run wants that:
      # its queries are eighteen tokens and its documents are 192, and
      # padding the queries out to the documents' length is nine tenths of
      # that work spent on padding (docs/plan.md 15.62).
      #
      # Two things make it possible, and both are new: `g.sharing` makes
      # the second application read the first's parameters rather than
      # declare its own, and the graph is built for no particular
      # sequence at all (`seq: nil`), so each side's batch says how long
      # it is (docs/plan.md 15.63).
      #
      #   graph = ModernBERT.towers(config, queries: 8, documents: 24)
      #   batch = ModernBERT.batch(config, qs, seq: 48, pooling: :mean,
      #                            fields: "queries.")
      #           .merge(ModernBERT.batch(config, ds, seq: 192, pooling: :mean,
      #                                   fields: "documents."))
      #
      # It answers one embedding per side, "queries.embedding" and
      # "documents.embedding", which is what the objective reads.
      #
      # `sides` is name => how many rows that side has, nil where the
      # count does not matter. It is named because an objective that reads
      # across the batch cannot be written against a dimension nothing
      # knows.
      def towers(config, sides, pooling: :mean, encoder_prefix: "", normalize: true,
                 dtype: :f32)
        config.check!
        raise ConfigError, "towers needs at least one side" if sides.empty?

        Torobi.graph do |g|
          sides.each do |side, rows|
            build = Build.new(seq: nil, rows:, dtype:, fields: "#{side}.")
            g.sharing(side) do
              x = g.name("hidden", body(g, config, build, encoder_prefix))
              pooled = pool(g, x, config, build, mode: pooling)
              pooled = normalized(g, pooled) if normalize
              g.output :"#{side}.embedding", g.name("embedding", pooled)
            end
          end
        end
      end

      # The encoder under wherever its parameters sit. Published
      # classifiers keep them under `model.` and bare encoders at the root,
      # and that is the only difference between the two graphs above.
      def body(g, config, build, encoder_prefix)
        return encode(g, config, build) if encoder_prefix.to_s.empty?

        g.scope(encoder_prefix) { encode(g, config, build) }
      end

      # A vector of length one, so a dot product is a cosine.
      def normalized(g, x)
        x / g.sum(x.square, axes: [-1], keepdims: true).sqrt
      end

      # One vector per row, the way `mode` asks for.
      #
      # `cls` is the first position, which for these tokenizers is the
      # sentence-start token: [batch, seq, hidden] keeping only seq 0.
      #
      # `mean` is over the tokens a row actually has. Padding is weighed
      # out of both the sum and the count, so a row's vector does not
      # depend on how much padding followed it, and two batches that
      # differ only in their longest row give the same answer for every
      # other row. The weights arrive as `:tokens` rather than being
      # derived from the attention mask: what a mask holds is whatever the
      # caller built it from, and a weight that must be exactly one or
      # exactly zero should not be read out of a large negative number.
      def pool(g, x, config, build, mode: config.classifier_pooling)
        case mode
        when :cls
          x.slice(axis: 1, start: 0, length: 1).reshape(shape: [-1, config.hidden_size])
        when :mean
          # Cast to what it is weighing: a batch carries f32, and the
          # model may be held in something narrower.
          weights = g.cast(g.input(build.field(:tokens), [build.rows, build.seq, 1]), x.dtype)
          g.sum(x * weights, axes: [1]) / g.sum(weights, axes: [1])
        else
          raise ConfigError, "unknown pooling #{mode.inspect}"
        end
      end

      # The encoder as a graph: token ids in, hidden states out.
      #
      # `build.seq` is the sequence length this graph is declared for,
      # and may be nil: since the head split keeps the dimensions it does
      # not know (docs/plan.md 15.63), nothing here needs a concrete one,
      # and each batch says how long it is.
      #
      # Two additive attention masks arrive from the batch, each
      # [batch, 1, seq, seq]: zero where a position may be attended to and
      # a large negative where it may not. Global layers read `mask`, local
      # ones read `local_mask`, which is the same padding with the sliding
      # window added. Two rather than one because they genuinely differ
      # (`Config#local_attention`), and building them is the caller's:
      # the graph says attention, not which positions this batch has.
      # `ModernBERT.masks` builds both.
      def graph(config, seq:, rows: nil, dtype: :f32)
        config.check!
        build = Build.new(seq:, rows:, dtype:)
        Torobi.graph do |g|
          # Named as well as declared an output, so a tap can read it: an
          # output is what the objective consumes, and a tap is how a
          # caller sees a value without a second graph to see it with.
          g.output :hidden, g.name("hidden", encode(g, config, build))
        end
      end

      # The encoder body, so the bare model and the classifier are one
      # description rather than two that must be kept in step.
      #
      # What it is being built for arrives as one value (`Models::Build`):
      # the sequence, the rows, the precision, and which batch fields to
      # read. `build.rows` is normally nil, because the batch is the
      # dimension that varies between steps and everything here works
      # without knowing it; naming it is for an objective that reads
      # across the batch rather than down it, which a contrastive loss
      # does (its negatives are the other rows), and which cannot be
      # written against a dimension nothing knows.
      def encode(g, config, build)
        ids = g.input(build.field(:input_ids), [build.rows, build.seq], dtype: :i32)
        # The masks arrive as f32, which is what a batch carries, and are
        # cast to what the model is held in. Nothing is cast when there is
        # nothing to cast to (`g.cast`).
        padding = g.cast(g.input(build.field(:mask), [build.rows, 1, 1, build.seq]),
                         build.dtype)
        # Summed once, not in each of the local layers: both are the same
        # for every one of them. Only when there is a local layer to read
        # it, so a configuration with none asks for no window.
        local = if config.any_local?
                  padding + g.cast(g.input(build.field(:window),
                                           [1, 1, build.seq, build.seq]), build.dtype)
                else
                  padding
                end

        x = g.embedding(ids, vocab: config.vocab_size, dim: config.hidden_size,
                        name: "embeddings.tok_embeddings", dtype: build.dtype)
        x = norm(g, x, config, name: "embeddings.norm")

        config.num_hidden_layers.times do |i|
          mask = config.global?(i) ? padding : local
          x = g.scope("layers.#{i}") { layer(g, x, config, i, mask) }
        end
        norm(g, x, config, name: "final_norm")
      end

      # One encoder block: attention with a residual, then the MLP with
      # another. Layer 0 has no attention norm, which is the reference's
      # own asymmetry and not a mistake here.
      def layer(g, x, config, index, mask)
        normed = index.zero? ? x : norm(g, x, config, name: "attn_norm")
        x += attention(g, normed, config, index, mask)
        x + mlp(g, norm(g, x, config, name: "mlp_norm"), config)
      end

      def attention(g, x, config, index, mask)
        heads = config.num_attention_heads
        dim = config.head_dim
        theta = config.theta(index)
        q, k, v = g.scope("attn") do
          g.linear(x, config.hidden_size * 3, name: "Wqkv", bias: false).split(3, axis: -1)
        end
        # [batch, seq, hidden] -> [batch, heads, seq, head_dim], so
        # attention runs per head. The two leading dimensions are kept as
        # they are (`0`), which is what lets both of them be symbolic:
        # only the hidden size is being divided up here, and it is the
        # one this knows (docs/plan.md 15.63).
        to_heads = lambda do |h|
          h.reshape(shape: [0, 0, heads, dim]).transpose(axes: [0, 2, 1, 3])
        end
        attended = g.sdpa(to_heads.call(q).rope(theta:),
                          to_heads.call(k).rope(theta:),
                          to_heads.call(v),
                          mask:)
        folded = attended.transpose(axes: [0, 2, 1, 3])
                         .reshape(shape: [0, 0, config.hidden_size])
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

      # One batch, from token ids.
      #
      # Where the line falls: **what the model's config determines is
      # Torobi's, and what is upstream of that is not.** The pad token, the
      # sliding window and the vocabulary all come out of the same
      # `config.json` this already reads, so leaving the caller to look
      # them up would be withholding what is in hand. Tokenizing is
      # upstream: it is decided by `tokenizer.json`, a different artifact
      # with a different lifecycle, and owning it would mean a second
      # native dependency for something the caller can do (docs/plan.md
      # section 15.19).
      #
      #   ids = tokenizer.encode_batch(texts).map(&:ids)   # yours
      #   batch = ModernBERT.batch(config, ids, seq: 128)  # this
      #   session.step!(batch)
      #
      # Rows are padded to `seq`, which is the length the graph was built
      # for; there is no other length it could be. A row longer than that
      # is refused rather than truncated, because which end to drop is a
      # decision about the data and not about the model.
      # `pooling:` is what the graph this feeds pools like, and adds the
      # per-token weights a mean over real tokens needs (`tokens`). It is
      # the same word `embedder` and `pool` take.
      def batch(config, rows, seq:, pooling: nil, fields: "")
        lengths = rows.map(&:size)
        too_long = lengths.each_with_index.select { |length, _| length > seq }
        unless too_long.empty?
          raise ConfigError,
                "row #{too_long.first.last} has #{too_long.first.first} tokens and this " \
                "batch pads to #{seq}. Tokenize to at most #{seq}, or pad to more; " \
                "where to cut a long text is the caller's to decide."
        end

        ids = rows.flat_map { |row| row + Array.new(seq - row.size, config.pad_token_id) }
        carried = { input_ids: TensorData.from_a([rows.size, seq], ids, dtype: :i32) }
                  .merge(masks(config, seq:, lengths:))
        carried = carried.merge(tokens: tokens(seq:, lengths:)) if pooling.to_s == "mean"
        return carried if fields.to_s.empty?

        carried.to_h { |name, value| [:"#{fields}#{name}", value] }
      end

      # Which positions are tokens and which are padding, [rows, seq, 1]:
      # one where a position is real and zero where it is not.
      #
      # What a mean over the real tokens weighs with, shaped to multiply a
      # [rows, seq, hidden] hidden state as it stands. Only a graph that
      # pools that way declares it, and a field nothing reads is refused
      # at the boundary rather than ignored, so `batch` is told which
      # pooling it is feeding rather than guessing.
      #
      # Public for a caller who has already padded its own ids.
      def tokens(seq:, lengths:)
        empty = lengths.index(0)
        if empty
          raise ConfigError,
                "row #{empty} has no tokens, and a mean over no tokens is not a vector"
        end

        TensorData.runs([lengths.size, seq, 1],
                        lengths.flat_map { |length| [[length, 1.0], [seq - length, 0.0]] })
      end

      # The two masks a batch must carry, as {mask:, local_mask:} ready to
      # go in. `batch` calls this; it is public for a caller who has
      # already padded its own ids.
      #
      # `lengths` is how many real tokens each row has; the rest is
      # padding, which nothing may attend to.
      #
      # Neither is built as an Array of Floats. A mask is runs of the same
      # number, so both are written as bytes directly (`TensorData.runs`);
      # at sequence 512 and batch 32 that is the difference between a few
      # hundred kilobytes and building 16.8 million Ruby Floats to throw
      # away.
      #
      # A large negative rather than -Infinity: a row that can attend to
      # nothing would otherwise softmax to NaN, and a finite floor keeps a
      # padded row's arithmetic finite even though its output is discarded.
      NEGATIVE = -1.0e9

      def masks(config, seq:, lengths:)
        padding = TensorData.runs(
          [lengths.size, 1, 1, seq],
          lengths.flat_map { |length| [[length, 0.0], [seq - length, NEGATIVE]] }
        )
        return { mask: padding } unless config.any_local?

        { mask: padding, window: window(config, seq:) }
      end

      # What a shape's window is kept in, bounded (`Models::Windows`).
      WINDOWS = Windows.new

      # The sliding window the local layers see, [1, 1, seq, seq].
      #
      # It depends on the sequence length and the configured width and on
      # nothing in the batch, so there is one of it however many rows
      # there are, and it is built once per shape and handed out again.
      def window(config, seq:)
        half = config.local_attention / 2
        WINDOWS.fetch([seq, half]) do
          TensorData.runs(
            [1, 1, seq, seq],
            (0...seq).flat_map do |i|
              first = [i - half, 0].max
              last = [i + half, seq - 1].min
              [[first, NEGATIVE], [last - first + 1, 0.0], [seq - 1 - last, NEGATIVE]]
            end
          )
        end
      end
    end
  end
end
