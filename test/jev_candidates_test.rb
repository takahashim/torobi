# frozen_string_literal: true

require_relative "test_helper"
require_relative "../tools/jev_candidates"

# The batching half of the Jev adapter, without the `tokenizers` gem: a
# stub tokenizer stands in for it, so what is tested here is the K/sequence
# bucketing, the padding and the masks -- the parts Torobi's fixed graph
# shapes force.
class JevCandidatesTest < Minitest::Test
  # A small ModernBERT with a local layer, so a window is built.
  CONFIG = Torobi::Models::ModernBERT.from_hash(
    "vocab_size" => 100, "hidden_size" => 8, "intermediate_size" => 16,
    "num_hidden_layers" => 3, "num_attention_heads" => 2,
    "global_attn_every_n_layers" => 2, "local_attention" => 4, "pad_token_id" => 3
  )

  # Whatever the adapter asks of a tokenizer, and nothing else.
  class StubTokenizer
    Encoding = Struct.new(:ids)

    def enable_truncation(*) = nil

    def encode(_context, candidate) = Encoding.new(Array.new((candidate.length % 3) + 1, 1))
  end

  def adapter(**)
    Jev::Candidates.new(CONFIG, tokenizer: StubTokenizer.new,
                                k_buckets: [2, 4], seq_buckets: [8], **)
  end

  def jnli(id, answer: 0)
    { "id" => id, "state" => { "前提" => "a", "仮説" => "b" }, "question" => "q",
      "choices" => %w[entailment contradiction neutral], "descriptions" => [nil, nil, nil],
      "answer" => answer, "task" => "jnli", "weight" => 1.0 }
  end

  def jcola(id)
    { "id" => id, "state" => "文", "question" => "自然か？", "choices" => [true, false],
      "descriptions" => %w[yes no], "answer" => 1, "task" => "jcola", "weight" => 1.0 }
  end

  def test_a_batch_carries_what_both_graphs_read
    encoded = [{ ids: [[1, 2], [1], [1, 2, 3]], k: 3, seq: 3, gold: 1, weight: 1.0 }]
    batch = adapter.build_batch(encoded, 4, 8)

    # The encoder sees bg * K rows; the objective sees bg groups.
    assert_equal [4, 8], batch[:input_ids].shape
    assert_equal :i32, batch[:input_ids].dtype
    assert_equal [4, 1, 1, 8], batch[:mask].shape
    assert_equal [1, 1, 8, 8], batch[:window].shape, "a local layer reads a window"
    assert_equal [1, 4], batch[:cand_mask].shape
    assert_equal [1], batch[:gold].shape
    assert_equal :i32, batch[:gold].dtype
    assert_equal [1], batch[:weight].shape
  end

  def test_the_padded_candidate_is_masked_and_the_gold_is_kept
    encoded = [{ ids: [[1, 2], [1], [1, 2, 3]], k: 3, seq: 3, gold: 1, weight: 1.0 }]
    batch = adapter.build_batch(encoded, 4, 8)

    assert_equal [0.0, 0.0, 0.0, Jev::Candidates::NEGATIVE], batch[:cand_mask].to_a
    assert_equal [1], batch[:gold].to_a
    assert_equal [1.0], batch[:weight].to_a
  end

  def test_a_question_is_padded_to_its_k_bucket
    encoded = [{ ids: [[1], [1]], k: 2, seq: 1, gold: 0, weight: 1.0 }]
    batch = adapter.build_batch(encoded, 4, 8)

    assert_equal [1, 4], batch[:cand_mask].shape
    assert_equal [0.0, 0.0, Jev::Candidates::NEGATIVE, Jev::Candidates::NEGATIVE],
                 batch[:cand_mask].to_a
  end

  def test_a_bucket_is_the_smallest_that_fits
    subject = adapter

    assert_equal 4, subject.bucket_k(3)
    assert_equal 2, subject.bucket_k(2)
    assert_equal 8, subject.bucket_seq(5)
    assert_raises(Torobi::ConfigError) { subject.bucket_k(5) }
    assert_raises(Torobi::ConfigError) { subject.bucket_seq(9) }
  end

  def test_questions_are_grouped_by_k_bucket_and_sliced_to_the_budget
    batches = adapter(pair_budget: 8).batches([jnli(0), jnli(1), jcola(2)]).to_a

    assert_equal 2, batches.size, "K=3 and K=2 are different buckets"
    k4, k2 = batches

    assert_equal [2, 4], k4[:cand_mask].shape, "two JNLI questions at the K=4 bucket"
    assert_equal [1, 2], k2[:cand_mask].shape
  end

  def test_the_tokenizer_is_told_to_truncate_the_context_only
    tokenizer = Class.new(StubTokenizer) do
      attr_reader :asked

      def enable_truncation(max_length, **options) = (@asked = [max_length, options])
    end.new
    Jev::Candidates.new(CONFIG, tokenizer: tokenizer)

    assert_equal [512, { strategy: "only_first" }], tokenizer.asked
  end
end
