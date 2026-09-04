# frozen_string_literal: true

module Torobi
  module Models
    # The context a model graph is built in, as one value.
    #
    # Four things travel from the top of a description down to the leaves
    # and are read there: how long a sequence the graph is declared for
    # (nil where it is declared for none), how many rows it is declared
    # for, what precision the weights are held in, and which batch fields
    # to read. None of them is a decision any layer makes. They are what
    # the caller decided, carried.
    #
    # **As one value rather than four keywords, because that is what they
    # are.** Threaded separately they cost every signature between the top
    # and the leaf: adding `fields:`, so that one encoder could read two
    # sets of rows (docs/plan.md 15.63), meant editing `body`, `encode`,
    # `pool` and their callers to pass along something none of them looks
    # at. The next one should cost the constructor that sets it and the
    # leaf that reads it.
    #
    # A build says what the *graph* is; it does not say what a batch is.
    # `ModernBERT.batch(config, rows, seq:)` keeps its own `seq`, and
    # means something else by it: a graph's is the length it is declared
    # for, and a batch's is the length its rows are padded to. Since
    # 15.63 those are different numbers.
    Build = Data.define(:seq, :rows, :dtype, :fields) do
      def initialize(seq: nil, rows: nil, dtype: :f32, fields: "")
        super(seq:, rows:, dtype: dtype.to_sym, fields: fields.to_s)
      end

      # A batch field's name under this build's prefix: "queries." and
      # `:input_ids` is what the query side of `ModernBERT.towers` reads.
      def field(name) = :"#{fields}#{name}"
    end
  end
end
