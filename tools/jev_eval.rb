# frozen_string_literal: true

require_relative "jev_graph"

module Jev
  # The reference's validation report (jev_local `train.py` `evaluate`):
  # per task, accuracy and NLL over each question's candidates, and the
  # macro accuracy that decides which checkpoint is best.
  #
  # One forward per batch, before the objective: what is scored is the
  # model's logits, not a loss, so `forward` is the pass (docs/plan.md
  # section 15.47). Candidate probabilities are the softmax over a
  # question's logits, which is the same normalization the reference
  # serves with.
  module Eval
    module_function

    def run(session, candidates, rows, logits: Jev::LOGITS)
      per_task = Hash.new { |hash, task| hash[task] = { n: 0, correct: 0, nll: 0.0 } }
      candidates.bundles(rows).each do |bundle|
        batch = candidates.build_batch(bundle.map { |row| candidates.encode(row) })
        logits_values = session.forward(batch.slice(:input_ids, :mask, :window)).fetch(logits).to_a
        offset = 0
        bundle.each do |row|
          k = row.fetch("choices").size
          values = logits_values[offset, k]
          offset += k
          gold = Integer(row.fetch("answer"))
          stats = per_task[row.fetch("task")]
          stats[:n] += 1
          stats[:correct] += 1 if values.each_index.max_by { |i| values[i] } == gold
          stats[:nll] += -Math.log(softmax(values)[gold])
        end
      end
      report(per_task)
    end

    # Stable softmax of one question's candidate logits.
    def softmax(values)
      top = values.max
      weights = values.map { |value| Math.exp(value - top) }
      total = weights.sum
      weights.map { |weight| weight / total }
    end

    def report(per_task)
      report = per_task.to_h do |task, stats|
        [task, { n: stats[:n], accuracy: stats[:correct] / stats[:n].to_f,
                 nll: stats[:nll] / stats[:n] }]
      end
      total = per_task.values.sum { |stats| stats[:n] }
      report["_all"] = {
        n: total,
        accuracy: per_task.values.sum { |stats| stats[:correct] } / total.to_f,
        nll: per_task.values.sum { |stats| stats[:nll] } / total.to_f,
        macro_accuracy: report.values.sum { |task| task[:accuracy] } / report.size
      }
      report
    end
  end
end
