# frozen_string_literal: true

require_relative "torobi/version"
require_relative "torobi/errors"
require_relative "torobi/freeze"
require_relative "torobi/ir"
require_relative "torobi/graph_config"
require_relative "torobi/journal"
require_relative "torobi/ops"
require_relative "torobi/shape"
require_relative "torobi/dsl"
require_relative "torobi/lora"
require_relative "torobi/models/build"
require_relative "torobi/models/windows"
require_relative "torobi/models/modern_bert"
require_relative "torobi/models/gemma3"
require_relative "torobi/models/llama"

# The engine, and the session that drives it. Optional: the pure-Ruby half
# (the DSL and the IR) stands on its own, and its tests need no extension.
begin
  require_relative "torobi/torobi"
  require_relative "torobi/preflight"
  require_relative "torobi/tensor_data"
  require_relative "torobi/weights"
  require_relative "torobi/batch"
  require_relative "torobi/checkpoint"
  require_relative "torobi/session"
  require_relative "torobi/memory"
  require_relative "torobi/hooks"
  require_relative "torobi/replay"
  require_relative "torobi/runner"
  require_relative "torobi/export"
  require_relative "torobi/grad_cache"
  require_relative "torobi/freshness"
  Torobi::Freshness.warn!
rescue LoadError
  # Not compiled yet (rake compile); Torobi::Session is simply absent.
end

# Torobi describes models and training objectives in Ruby, once, as an
# immutable GraphConfig; an MLX-backed engine executes them. Ruby owns the
# language, the engine owns the execution. See docs/plan.md.
module Torobi
end
