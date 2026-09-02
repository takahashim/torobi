# frozen_string_literal: true

module Torobi
  # Base class for everything Torobi raises.
  class Error < StandardError; end

  # A GraphConfig that is structurally wrong: bad references, duplicate
  # names, unreachable nodes. Raised at construction time, so an invalid
  # graph is never representable as a value.
  class ConfigError < Error; end
end
