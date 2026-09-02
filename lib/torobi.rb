# frozen_string_literal: true

require_relative "torobi/version"
require_relative "torobi/errors"
require_relative "torobi/freeze"
require_relative "torobi/ir"
require_relative "torobi/graph_config"

# Torobi describes models and training objectives in Ruby, once, as an
# immutable GraphConfig; an MLX-backed engine executes them. Ruby owns the
# language, the engine owns the execution. See docs/plan.md.
module Torobi
end
