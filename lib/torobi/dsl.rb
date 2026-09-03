# frozen_string_literal: true

require_relative "dsl/handle"
require_relative "dsl/builder"

module Torobi
  # Builds one model graph. The block runs exactly once, at definition time;
  # `g` is the only way in, and the result is a frozen IR::Graph.
  #
  #   model = Torobi.graph do |g|
  #     x = g.input :x, [nil, 4]
  #     g.output :logits, g.linear(x, 2, name: "linear")
  #   end
  def self.graph
    builder = DSL::Builder.new
    yield builder
    builder.to_graph
  end

  # Builds an objective graph: the same DSL, plus the ability to read a
  # model's named outputs (docs/plan.md section 5A.3). It reaches a named
  # scalar, conventionally "loss".
  #
  #   objective = Torobi.objective(student: model) do |g|
  #     s = g.from_model :student, :logits
  #     t = g.from_batch :teacher, [nil, 2]
  #     g.output :loss, g.mse(s, t)
  #   end
  def self.objective(models)
    builder = DSL::Builder.new(models:)
    yield builder
    builder.to_graph
  end
end
