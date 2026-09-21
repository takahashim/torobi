# frozen_string_literal: true

require_relative "test_helper"
require_relative "../tools/jev_candidates"

# The grouped loss of the Jev cross-encoder, held to the reference's own
# arithmetic: for each question, the cross-entropy over that question's
# candidate logits. It is written with membership masks rather than a
# reshape to a fixed K, which is what lets a batch mix questions of
# different K; this is the test that the masks compute what the reference's
# per-group Python loop computes.
class JevGroupedLossTest < Minitest::Test
  CONFIG = Torobi::Models::ModernBERT.from_hash(
    "vocab_size" => 20, "hidden_size" => 4, "intermediate_size" => 8,
    "num_hidden_layers" => 2, "num_attention_heads" => 1,
    "global_attn_every_n_layers" => 2, "local_attention" => 2, "pad_token_id" => 0
  )

  class StubTokenizer
    Encoding = Struct.new(:ids)

    def enable_truncation(*) = nil

    def encode(_context, candidate)
      Encoding.new(Array.new((candidate.length % 3) + 1) { |i| (i % 7) + 1 })
    end
  end

  def model = Torobi::Models::ModernBERT.classifier(CONFIG, seq: nil)

  def objective(model)
    Torobi.objective(student: model) do |g|
      logits = g.from_model(:student, :logits)
      member = g.input(:member, [nil, nil])
      gold = g.input(:gold_member, [nil, nil])
      weight = g.input(:weight, [nil])
      group_max = g.max(logits + ((member - 1.0) * 1.0e9), axes: [0])
      sum_exp = g.sum((logits - group_max.reshape(shape: [1, -1])).exp * member, axes: [0])
      gold_logit = g.sum(logits * gold, axes: [0])
      g.output :loss, g.mean((sum_exp.log + group_max - gold_logit) * weight)
    end
  end

  def weights(graph)
    rng = Random.new(7)
    { params: graph.parameters.to_h do |parameter|
      shape = parameter.spec.shape
      ["#{parameter.model}.#{parameter.path}",
       { shape:, data: Array.new(shape.inject(1, :*)) { rng.rand(-0.3..0.3) } }]
    end }
  end

  def rows
    [
      { "state" => { "a" => "1" }, "question" => "q1", "choices" => %w[x y z],
        "descriptions" => [nil, nil, nil], "answer" => 2, "task" => "t", "weight" => 1.0 },
      { "state" => "s2", "question" => "q2", "choices" => %w[p q],
        "descriptions" => [nil, nil], "answer" => 0, "task" => "t", "weight" => 1.0 }
    ]
  end

  def test_the_loss_is_the_cross_entropy_over_each_questions_candidates
    graph = Torobi::GraphConfig.new(models: { student: model }, objective: objective(model))
    candidates = Jev::Candidates.new(CONFIG, tokenizer: StubTokenizer.new, pair_budget: 8)
    batch = candidates.build_batch(rows.map { |row| candidates.encode(row) })
    model_fields = batch.slice(:input_ids, :mask, :window)

    Torobi::Session.open(graph, weights: weights(graph), seed: 1) do |s|
      logits = s.forward(model_fields).fetch("student.logits").to_a
      expected = expected_loss(logits, [3, 2], [2, 0])

      assert_in_delta expected, s.evaluate(batch), 1e-5
    end
  end

  # The reference's loss, in Ruby: logsumexp over a question's logits minus
  # the answer's logit, averaged over questions.
  def expected_loss(logits, ks, golds)
    offset = 0
    nlls = ks.each_with_index.map do |k, group|
      values = logits[offset, k]
      offset += k
      top = values.max
      (Math.log(values.sum { |value| Math.exp(value - top) }) + top) - values[golds[group]]
    end
    nlls.sum / nlls.size
  end
end
