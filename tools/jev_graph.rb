# frozen_string_literal: true

require "torobi"

module Jev
  # The model and the loss the Jev cross-encoder is trained and evaluated
  # with, in one place so both use the same graph (jev_local
  # `modernbert/train.py`, docs/plan.md section 15.74 for the clipping the
  # reference's `clip_grad_norm_` maps to).
  #
  # The sequence is free (`seq: nil`): a batch says how long it is, which
  # is what the reference's per-batch padding does and what keeps the
  # padding to the longest pair actually in the step. K is free too -- the
  # loss is written over membership masks, not a reshape to a fixed K --
  # so a batch may mix questions of different K.

  module_function

  # The model's name in the graph and the qualified output the objective
  # reads; named here rather than spelled out where they are used.
  MODEL = :student
  LOGITS = "student.logits"

  # A one-logit cross-encoder over the base's encoder, with a head the
  # checkpoint does not have. The base is `ModernBertForMaskedLM`, whose
  # encoder sits under `model.` and whose `head.dense`/`head.norm` are the
  # same tensors a classification head has, so only `classifier` is fresh.
  def model(config)
    Torobi::Models::ModernBERT.classifier(config, seq: nil, encoder_prefix: "model")
  end

  # The reference's loss, per question: the cross-entropy over that
  # question's candidate logits.
  #
  # `member` is [rows, groups] and one where a row is one of a question's
  # candidates; `gold_member` is one at the row the answer points to. The
  # logsumexp is over a question's rows, which is what makes this exact
  # without a fixed K. A large negative, not -Infinity, so a row outside a
  # group does not make the max NaN.
  def objective(model)
    Torobi.objective(MODEL => model) do |g|
      logits = g.from_model(MODEL, :logits)
      member = g.input(:member, [nil, nil])
      gold = g.input(:gold_member, [nil, nil])
      weight = g.input(:weight, [nil])
      group_max = g.max(logits + ((member - 1.0) * 1.0e9), axes: [0])
      sum_exp = g.sum((logits - group_max.reshape(shape: [1, -1])).exp * member, axes: [0])
      gold_logit = g.sum(logits * gold, axes: [0])
      g.output :loss, g.mean((sum_exp.log + group_max - gold_logit) * weight)
    end
  end

  def graph(config)
    built = model(config)
    Torobi::GraphConfig.new(models: { MODEL => built }, objective: objective(built))
  end
end
