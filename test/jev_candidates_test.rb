# frozen_string_literal: true

require_relative "test_helper"
require_relative "../tools/jev_candidates"

# The batching half of the Jev adapter, without the `tokenizers` gem: a
# stub tokenizer stands in for it, so what is tested here is the flattening
# and the membership masks the grouped loss reads.
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
    Jev::Candidates.new(CONFIG, tokenizer: StubTokenizer.new, **)
  end

  def jnli(answer: 0)
    { "id" => 0, "state" => { "前提" => "a", "仮説" => "b" }, "question" => "q",
      "choices" => %w[entailment contradiction neutral], "descriptions" => [nil, nil, nil],
      "answer" => answer, "task" => "jnli", "weight" => 1.0 }
  end

  def jcola
    { "id" => 1, "state" => "文", "question" => "自然か？", "choices" => [true, false],
      "descriptions" => %w[yes no], "answer" => 1, "task" => "jcola", "weight" => 1.0 }
  end

  def test_a_batch_carries_what_both_graphs_read
    encoded = [{ ids: [[1, 2], [1], [1, 2, 3]], k: 3, gold: 1, weight: 1.0 }]
    batch = adapter.build_batch(encoded)

    # The encoder sees one row per candidate; the sequence is the longest.
    assert_equal [3, 3], batch[:input_ids].shape
    assert_equal :i32, batch[:input_ids].dtype
    assert_equal [3, 1, 1, 3], batch[:mask].shape
    assert_equal [1, 1, 3, 3], batch[:window].shape, "a local layer reads a window"
    assert_equal [3, 1], batch[:member].shape
    assert_equal [3, 1], batch[:gold_member].shape
    assert_equal [1], batch[:weight].shape
  end

  def test_the_membership_masks_mark_the_candidates_and_the_answer
    encoded = [{ ids: [[1, 2], [1], [1, 2, 3]], k: 3, gold: 1, weight: 1.0 }]
    batch = adapter.build_batch(encoded)

    assert_equal [1.0, 1.0, 1.0], batch[:member].to_a
    assert_equal [0.0, 1.0, 0.0], batch[:gold_member].to_a, "the answer is the second candidate"
  end

  def test_two_questions_of_different_k_share_a_batch
    encoded = [
      { ids: [[1, 2], [1], [1, 2, 3]], k: 3, gold: 2, weight: 1.0 },
      { ids: [[1], [1, 2]], k: 2, gold: 0, weight: 1.0 }
    ]
    batch = adapter.build_batch(encoded)

    # Five rows, two groups; a group's rows are a contiguous run.
    assert_equal [5, 2], batch[:member].shape
    assert_equal [1, 0, 1, 0, 1, 0, 0, 1, 0, 1], batch[:member].to_a
    # Group 0's answer is its third row, group 1's is its first.
    assert_equal [0, 0, 0, 0, 1, 0, 0, 1, 0, 0], batch[:gold_member].to_a
  end

  def test_questions_are_bundled_to_the_pair_budget
    batches = adapter(pair_budget: 4).batches([jnli, jcola]).to_a

    assert_equal 2, batches.size, "3 pairs then 2 pairs do not fit in one budget of 4"
    assert_equal [3, 1], batches.first[:member].shape
    assert_equal [2, 1], batches.last[:member].shape
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
