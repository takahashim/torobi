# frozen_string_literal: true

module Torobi
  # Base class for everything Torobi raises.
  class Error < StandardError; end

  # A GraphConfig that is structurally wrong: bad references, duplicate
  # names, unreachable nodes. Raised at construction time, so an invalid
  # graph is never representable as a value.
  class ConfigError < Error; end

  # The engine cannot run here: the extension is missing, or something it
  # needs at runtime is not in place. Raised before MLX is touched, since
  # some of those failures would otherwise abort the process.
  class EngineUnavailable < Error; end
end
