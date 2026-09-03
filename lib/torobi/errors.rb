# frozen_string_literal: true

module Torobi
  # Base class for everything Torobi raises.
  #
  # The hierarchy is the error contract of docs/plan.md section 5A.4, and it
  # is a contract about *when* a failure is found and *what survives it*:
  #
  #   ConfigError        found while building, before any engine exists.
  #                      Nothing to recover: the description is wrong.
  #   StepError          found while running. The engine reported it, the
  #                      session survives, and another step may be tried.
  #   EngineUnavailable  the engine cannot run here at all. Raised before
  #                      MLX is touched, because some of these failures
  #                      would otherwise end the process (section 4.1).
  #
  # A fourth kind exists and has no class, by necessity: a failure MLX
  # raises as a C++ exception during initialization aborts the process.
  # Preflight refuses the cases we can foresee; the rest is why the plan
  # says to run under a supervisor.
  class Error < StandardError; end

  # A description that is wrong: bad references, duplicate names,
  # unreachable nodes, shapes that cannot meet, an objective reading an
  # output no model declares. Raised at construction time, so an invalid
  # graph is never representable as a value.
  class ConfigError < Error; end

  # A step that could not be taken: a batch that does not match what the
  # graph declared, a parameter that is missing, a shape only MLX could
  # settle. The session is still usable.
  class StepError < Error; end

  # The engine cannot run here: the extension is missing, or something it
  # needs at runtime is not in place.
  class EngineUnavailable < Error; end
end
