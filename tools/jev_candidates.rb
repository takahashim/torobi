# frozen_string_literal: true

require "torobi"
require_relative "jev_format"

module Jev
  # Turns the JSONL `data.py` produces into the batches a ModernBERT
  # cross-encoder trains on: each question is one group of K candidates,
  # rendered and tokenized as pairs, and the loss is the listwise
  # cross-entropy over a group's logits.
  #
  # Two of Torobi's constraints shape this. The graph's sequence length is
  # fixed, so rows are padded to a sequence bucket, and **K must be a graph
  # constant**, so questions are grouped by a K bucket and short candidate
  # sets are padded to it and masked. The reference (`train.py`) bundles by
  # a free pair count instead; the loss is the same listwise cross-entropy
  # either way, but which questions share a step differs, which is the one
  # place this port knowingly departs from it.
  #
  # Tokenizing is separated from batching so the batching can be tested
  # without the `tokenizers` gem: `encode` uses the tokenizer, `build_batch`
  # takes rows already tokenized.
  class Candidates
    # The K the data actually has (Noul/JCoLA/Moral 2, JNLI 3, JCommonSenseQA
    # 5, JSTS 6, MASSIVE 18, and MASSIVE subsets 2..6).
    DEFAULT_K_BUCKETS = [2, 3, 4, 5, 6, 18].freeze
    DEFAULT_SEQ_BUCKETS = [64, 96, 128, 192, 256, 384, 512].freeze

    # A padded candidate is attended to by nothing, so its logit must not
    # enter the softmax; a large finite floor rather than -Infinity, as the
    # masks elsewhere use.
    NEGATIVE = -1.0e9

    def initialize(config, tokenizer: nil, pair_budget: 128, max_length: 512,
                   k_buckets: DEFAULT_K_BUCKETS, seq_buckets: DEFAULT_SEQ_BUCKETS)
      @config = config
      @tokenizer = tokenizer
      @pair_budget = pair_budget
      @max_length = max_length
      @k_buckets = k_buckets.sort.freeze
      @seq_buckets = seq_buckets.sort.freeze
      # The reference truncates the context side only, so the question is
      # never lost; set here so a caller cannot forget it.
      tokenizer&.enable_truncation(max_length, strategy: "only_first")
    end

    attr_reader :config, :pair_budget, :max_length

    # Parsed JSONL rows in, batch hashes out, grouped by (K, sequence)
    # bucket and sliced to the pair budget.
    def batches(rows)
      Enumerator.new do |yielder|
        grouped = rows.map { |row| encode(row) }
                      .group_by { |encoded| [bucket_k(encoded[:k]), bucket_seq(encoded[:seq])] }
        grouped.each do |(k, seq), group|
          group.each_slice(per_batch(k)) { |slice| yielder << build_batch(slice, k, seq) }
        end
      end
    end

    # One row: rendered, tokenized, and measured. What `build_batch` needs
    # and nothing more.
    def encode(row)
      context = Jev.render_context(row.fetch("state"), row.fetch("question"))
      candidates = Jev.render_candidates(row.fetch("choices"), row.fetch("descriptions"))
      ids = candidates.map { |candidate| @tokenizer.encode(context, candidate).ids }
      { ids: ids, k: ids.size, seq: ids.map(&:size).max,
        gold: Integer(row.fetch("answer")), weight: Float(row.fetch("weight")) }
    end

    # One batch from rows already tokenized.
    #
    # Real candidates are the rows; a padded candidate is an empty row,
    # which `ModernBERT.batch` fills with the pad token and masks out, so
    # the model sees the padding it would have seen and the objective's own
    # candidate mask is what keeps the padded logit out of the loss.
    def build_batch(encoded, k, seq)
      rows = encoded.flat_map { |one| one[:ids] + Array.new(k - one[:k]) { [] } }
      carried = Torobi::Models::ModernBERT.batch(@config, rows, seq:)
      carried.merge(
        cand_mask: Torobi::TensorData.from_a(
          [encoded.size, k],
          encoded.flat_map { |one| Array.new(one[:k], 0.0) + Array.new(k - one[:k], NEGATIVE) }
        ),
        gold: Torobi::TensorData.from_a([encoded.size], encoded.map do |one|
          one[:gold]
        end, dtype: :i32),
        weight: Torobi::TensorData.from_a([encoded.size], encoded.map { |one| one[:weight] })
      )
    end

    def bucket_k(k) = bucket(@k_buckets, k, "K")

    def bucket_seq(seq) = bucket(@seq_buckets, seq, "sequence")

    private

    # How many questions fit in one step at this K, so the pairs do not
    # exceed the budget.
    def per_batch(k) = [@pair_budget / k, 1].max

    def bucket(buckets, value, what)
      found = buckets.find { |edge| edge >= value }
      unless found
        raise Torobi::ConfigError,
              "#{what} #{value} is past every bucket (#{buckets.inspect})"
      end

      found
    end
  end
end
