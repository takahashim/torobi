# frozen_string_literal: true

require_relative "dsl/handle"
require_relative "dsl/builder"

module Torobi
  # Builds one graph. The block runs exactly once, at definition time; `g`
  # is the only way in, and the result is a frozen IR::Graph.
  #
  #   model = Torobi.graph do |g|
  #     x = g.input :x, [nil, 4]
  #     g.output g.linear(x, 2, name: "linear")
  #   end
  def self.graph
    builder = DSL::Builder.new
    yield builder
    builder.to_graph
  end
end
