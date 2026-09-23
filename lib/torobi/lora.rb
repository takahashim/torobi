# frozen_string_literal: true

module Torobi
  # Low-rank adaptation (Hu et al., 2021), as something a graph is built
  # with (docs/plan.md section 15.50).
  #
  # A fine-tune that moves every weight of a 0.5B model needs the
  # gradients and the optimizer's two moments for every one of them,
  # which is four copies of the model. LoRA trains a pair of small
  # matrices beside each weight instead and leaves the weight alone:
  #
  #   y = Wx + b  +  (alpha / rank) * B(Ax)
  #
  # `A` is [rank, d_in] and starts random; `B` is [d_out, rank] and
  # **starts at zero**, so before any step the sum is exactly the base
  # model. That is not a detail: it is what makes an adapted model safe
  # to start from, and there is a test that it holds.
  #
  # What this holds is the decision, not the arithmetic:
  #
  #   adapter = Torobi::LoRA.new(rank: 8, alpha: 16, on: %w[q_proj v_proj])
  #   model = Torobi::Models::Llama.causal_lm(config, seq: 512, adapter:)
  #
  # `on` names the linears to adapt by their last segment, which is how
  # everyone else spells it (`target_modules` in PEFT) and is the only
  # part of the path a model's author and its adapter agree about: the
  # layer number is the model's business.
  #
  # There is no dropout on the adapter here. PEFT has one; it is a
  # regularizer rather than part of what LoRA means, and a graph that
  # draws randomness cannot be back-propagated twice, which is what a
  # gradient cache does (`Torobi::GradCache`).
  LoRA = Data.define(:rank, :alpha, :on) do
    # The two halves of the product, as a checkpoint spells them.
    SUFFIXES = %w[lora_A.weight lora_B.weight].freeze

    def initialize(rank:, on:, alpha: nil)
      rank = Integer(rank)
      raise ConfigError, "a LoRA rank must be positive, got #{rank}" unless rank.positive?

      on = Array(on).map(&:to_s)
      raise ConfigError, "a LoRA adapts something: on: names which linears" if on.empty?

      # alpha defaults to the rank, which is a scale of one: what the
      # rank changes is then capacity rather than step size.
      super(rank:, alpha: Float(alpha || rank), on: on.freeze)
    end

    # What the low-rank product is multiplied by before it is added.
    def scale = alpha / rank

    # Whether the linear at `path` is one of the ones being adapted.
    # Matched on the last segment, so `q_proj` names the one in every
    # layer and `layers.0.self_attn.q_proj` names one.
    def wraps?(path)
      path = path.to_s
      on.any? { |target| path == target || path.end_with?(".#{target}") }
    end

    # What an adapter adds to a linear, by the names PEFT gives them.
    def paths(name) = SUFFIXES.map { |suffix| "#{name}.#{suffix}" }

    # Whether the parameter at `path` is trained while this adapter is in
    # scope: only the adapter's own are. A base model left trainable by an
    # oversight is not a LoRA fine-tune, and the window's `unfreeze!` moves
    # only within what the graph declared trainable, so this is the
    # declaration that matters.
    def trains?(path) = adapted?(path)

    # The adapter's own arithmetic, added to the linear `name` over `x`:
    # `x` through a narrow matrix and back out to the width the linear
    # has, scaled. Built through the graph's public vocabulary, with the
    # names `paths` gives, which are the names every other part of Torobi
    # reads them by.
    #
    # `B` starts at zero, so this contributes nothing until something has
    # trained it. That is what makes an adapted model start as the model
    # it adapts.
    def contribution(g, x, d_in:, d_out:, name:)
      a_path, b_path = paths(name)
      a = g.param(a_path, [rank, d_in], dtype: x.dtype, init: { "type" => "kaiming_uniform" })
      b = g.param(b_path, [d_out, rank], dtype: x.dtype, init: { "type" => "zeros" })
      down = g.matmul(x, a.transpose(axes: [1, 0]))
      g.matmul(down, b.transpose(axes: [1, 0])) * scale
    end

    # What `fresh:` should say when starting from a published checkpoint:
    # the adapter's matrices are in no file, and everything else must be.
    #
    #   Session.open(config, pretrained: { m: file },
    #                fresh: adapter.fresh(config))
    #
    # The paths themselves rather than a pattern. The engine's patterns
    # are prefixes (`student.layers.3.*`), and what these have in common
    # is how they end; asking for a wildcard in the middle would be
    # asking for a query language to say something a list already says.
    def fresh(config)
      config.parameters.map(&:qualified_path).select { |path| adapted?(path) }
    end

    # Whether a parameter path is an adapter's own.
    #
    # A fact about the name rather than about any particular adapter, so
    # it is asked of the class: `Session#export_model!` wants to know
    # whether what it is about to write holds an adapter, and it has the
    # paths rather than the LoRA that made them.
    def self.adapted?(path)
      SUFFIXES.any? { |suffix| path.to_s.end_with?(".#{suffix}") }
    end

    def adapted?(path) = self.class.adapted?(path)
  end
end
