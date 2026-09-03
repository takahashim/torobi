# frozen_string_literal: true

require_relative "ir/ref"
require_relative "ir/dtype"
require_relative "ir/json"
require_relative "ir/source"
require_relative "ir/input_spec"
require_relative "ir/parameter_spec"
require_relative "ir/node_spec"
require_relative "ir/graph"

module Torobi
  # The intermediate representation behind the Graph DSL.
  #
  # A graph is data: inputs, parameters, nodes and outputs, each a frozen
  # value object with a stable serialization. The DSL produces these; the
  # engine (and the Python oracle runner) consume them. Nothing in here
  # computes anything.
  module IR
  end
end
