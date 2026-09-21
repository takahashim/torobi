# frozen_string_literal: true

require "torobi"
require_relative "jev_format"

module Jev
  # Turns the JSONL `data.py` produces into the batches a ModernBERT
  # cross-encoder trains on: each question is one group of K candidates,
  # rendered and tokenized as pairs, and the loss is the listwise
  # cross-entropy over a group's logits.
  #
  # **K and the sequence are free**, as the reference has them. The graph
  # is built with `seq: nil`, so a batch says how long it is, and the
  # grouped loss is written with membership masks rather than a reshape to
  # a fixed K (see `experiments/modernbert_ja_choice.rb`), so a batch may
  # mix questions of different K. Questions are bundled by a pair budget,
  # which is the reference's own batching.
  #
  # Tokenizing is separated from batching so the batching can be tested
  # without the `tokenizers` gem: `encode` uses the tokenizer, `build_batch`
  # takes rows already tokenized.
  class Candidates
    def initialize(config, tokenizer: nil, pair_budget: 128, max_length: 512)
      @config = config
      @tokenizer = tokenizer
      @pair_budget = pair_budget
      @max_length = max_length
      # The reference truncates the context side only, so the question is
      # never lost; set here so a caller cannot forget it.
      tokenizer&.enable_truncation(max_length, strategy: "only_first")
    end

    attr_reader :config, :pair_budget, :max_length

    # Parsed JSONL rows in, groups of rows out, bundled so a step's pairs
    # do not exceed the budget. The bundling on its own, so evaluation can
    # read the rows a batch was made from.
    def bundles(rows)
      Enumerator.new do |yielder|
        bundle = []
        pairs = 0
        rows.each do |row|
          k = row.fetch("choices").size
          if !bundle.empty? && pairs + k > @pair_budget
            yielder << bundle
            bundle = []
            pairs = 0
          end
          bundle << row
          pairs += k
        end
        yielder << bundle unless bundle.empty?
      end
    end

    # The same, rendered and tokenized.
    def batches(rows)
      Enumerator.new do |yielder|
        bundles(rows).each { |bundle| yielder << build_batch(bundle.map { |held| encode(held) }) }
      end
    end

    # One row: rendered, tokenized, and measured.
    def encode(row)
      context = Jev.render_context(row.fetch("state"), row.fetch("question"))
      candidates = Jev.render_candidates(row.fetch("choices"), row.fetch("descriptions"))
      ids = candidates.map { |candidate| @tokenizer.encode(context, candidate).ids }
      { ids: ids, k: ids.size, gold: Integer(row.fetch("answer")),
        weight: Float(row.fetch("weight")) }
    end

    # One step's questions as a batch.
    #
    # Candidates are flattened in question order, so the rows of a group are
    # a contiguous run; the sequence is the longest pair in the batch; and
    # the two membership masks say which rows belong to which group and
    # where each group's answer is. Those are what let the loss be a
    # grouped cross-entropy without a fixed K.
    def build_batch(encoded)
      rows = encoded.flat_map { |one| one[:ids] }
      groups = encoded.size
      seq = rows.map(&:size).max
      member, gold_member = memberships(encoded, groups)
      Torobi::Models::ModernBERT.batch(@config, rows, seq:).merge(
        member: Torobi::TensorData.from_a([rows.size, groups], member),
        gold_member: Torobi::TensorData.from_a([rows.size, groups], gold_member),
        weight: Torobi::TensorData.from_a([groups], encoded.map { |one| one[:weight] })
      )
    end

    private

    # [rows, groups]: one where a row is one of a group's candidates, and a
    # second with one at the row a group's answer points to.
    def memberships(encoded, groups)
      rows = encoded.sum { |one| one[:k] }
      member = Array.new(rows * groups, 0.0)
      gold_member = Array.new(rows * groups, 0.0)
      offset = 0
      encoded.each_with_index do |one, group|
        one[:k].times { |i| member[((offset + i) * groups) + group] = 1.0 }
        gold_member[((offset + one[:gold]) * groups) + group] = 1.0
        offset += one[:k]
      end
      [member, gold_member]
    end
  end
end
