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
  #     Busy             the session is serving another thread. Nothing is
  #                      wrong with it; the call may be retried.
  #   SessionPoisoned    the engine panicked. The session is not usable.
  #   RuntimePoisoned    the engine panicked while it held MLX. No session
  #                      in this process is usable.
  #   SessionClosed      the session was closed.
  #   EngineUnavailable  the engine cannot run here at all. Raised before
  #                      MLX is touched, because some of these failures
  #                      would otherwise end the process (section 4.1).
  #
  # Why Busy sits under StepError and the last two do not: a caller that
  # rescues StepError means "the engine refused, the session is still mine".
  # That is true of Busy and false of the others, where the session is gone.
  #
  # There is no class for "interrupted". A pending interrupt can stop the
  # engine's blocking region from starting, but the binding retries rather
  # than reporting it (notes/SESSION_CONCURRENCY_SPEC.md section 5), because
  # the flag it reads is also the scheduler's timer. Ruby raises what it was
  # actually holding, at its own next checkpoint.
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

  # The session is serving another thread. It serves one at a time, so this
  # is not a fault: wait and try again, or use one session per thread.
  class Busy < StepError; end

  # The engine panicked under this session. Whatever state it left behind
  # cannot be trusted, so the session refuses everything afterwards. Open a
  # new one, from a checkpoint if there is one.
  class SessionPoisoned < Error; end

  # An operation on a session that has been closed.
  class SessionClosed < Error; end

  # The engine panicked while it held MLX. MLX is process-global, so what
  # that left behind is not knowable and nothing more is attempted in this
  # process: every session refuses, not only the one that panicked. Start a
  # new process, from a checkpoint if there is one
  # (notes/ENGINE_RUNTIME_BOUNDARY_PLAN.md section 5.2).
  class RuntimePoisoned < Error; end

  # The engine cannot run here: the extension is missing, or something it
  # needs at runtime is not in place.
  class EngineUnavailable < Error; end
end
