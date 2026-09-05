# frozen_string_literal: true

module Torobi
  module Models
    # What an architecture is described against: the builder it writes
    # into, the configuration it is built from, and what it is being built
    # for. One builder serves the whole graph and the config never
    # changes, so they are held here and a signature carries what actually
    # varies -- the value flowing through.
    #
    #   def layer(x)
    #     x += attention(norm(x, name: "input_layernorm"))
    #     x + mlp(norm(x, name: "post_attention_layernorm"))
    #   end
    #
    # A subclass is one architecture. It reaches the graph through
    # VOCABULARY below and the rest of what it needs through `@config` and
    # `@build`; `Torobi.graph` still owns the builder and the frozen IR
    # that comes out of it.
    class Description
      # What a description may say to the graph, and nothing else.
      #
      # Written out rather than inherited from `DSL::Builder` or forwarded
      # wholesale, for the same reason the op manifest is a file: a
      # description works in a vocabulary somebody chose, and one that can
      # reach `emit` can put a node in the graph that no layer here means.
      # Adding to this list is how the vocabulary grows, which is a
      # deliberate act rather than a side effect of inheriting.
      #
      # These read without a receiver, so a call that builds the graph and
      # a call that reads a number look different on the page:
      # `linear(x, ...)` against `@config.hidden_size`.
      VOCABULARY = %i[
        input from_batch from_model output
        scope name sharing adapting parameter
        linear embedding layer_norm rms_norm geglu
        matmul sdpa cross_entropy cast mean sum max
      ].freeze

      VOCABULARY.each do |op|
        define_method(op) do |*args, **kwargs, &block|
          @g.public_send(op, *args, **kwargs, &block)
        end
      end

      # `build` is what this graph is being built for (`Models::Build`).
      # ModernBERT's towers make one description per side, because each
      # side reads its own fields at its own length; they share the
      # builder, which is what makes them one graph.
      def initialize(g, config, build)
        @g = g
        @config = config
        @build = build
      end

      attr_reader :config, :build
    end
  end
end
