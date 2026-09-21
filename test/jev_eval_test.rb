# frozen_string_literal: true

require_relative "test_helper"
require_relative "../tools/jev_candidates"
require_relative "../tools/jev_graph"
require_relative "../tools/jev_eval"

# The reference's validation report: per task, accuracy and NLL, and the
# macro accuracy that picks a checkpoint. What is checked here is the shape
# of the report and that every question is counted once in its task; the
# arithmetic is `softmax` over a question's logits, which is the same one
# the grouped loss uses.
class JevEvalTest < Minitest::Test
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

  def rows
    [
      { "state" => "s", "question" => "q", "choices" => %w[x y z],
        "descriptions" => [nil, nil, nil], "answer" => 1, "task" => "choice", "weight" => 1.0 },
      { "state" => "s", "question" => "q", "choices" => [true, false],
        "descriptions" => [nil, nil], "answer" => 0, "task" => "noul", "weight" => 1.0 },
      { "state" => "s", "question" => "q", "choices" => %w[p q r],
        "descriptions" => [nil, nil, nil], "answer" => 2, "task" => "choice", "weight" => 1.0 }
    ]
  end

  def weights(graph)
    rng = Random.new(3)
    { params: graph.parameters.to_h do |parameter|
      shape = parameter.spec.shape
      ["#{parameter.model}.#{parameter.path}",
       { shape:, data: Array.new(shape.inject(1, :*)) { rng.rand(-0.3..0.3) } }]
    end }
  end

  def test_the_report_counts_every_question_in_its_task
    graph = Jev.graph(CONFIG)
    candidates = Jev::Candidates.new(CONFIG, tokenizer: StubTokenizer.new, pair_budget: 8)

    Torobi::Session.open(graph, weights: weights(graph), seed: 1) do |session|
      report = Jev::Eval.run(session, candidates, rows)

      assert_equal %w[_all choice noul], report.keys.sort
      assert_equal 3, report["_all"][:n]
      assert_equal 2, report["choice"][:n]
      assert_equal 1, report["noul"][:n]
      assert_in_delta (report["choice"][:accuracy] + report["noul"][:accuracy]) / 2,
                      report["_all"][:macro_accuracy], 1e-9
      report.each_value do |task|
        assert_operator task[:accuracy], :>=, 0.0
        assert_operator task[:accuracy], :<=, 1.0
        assert_operator task[:nll], :>=, 0.0
      end
    end
  end
end
