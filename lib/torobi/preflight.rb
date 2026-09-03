# frozen_string_literal: true

module Torobi
  # What must hold before the engine is asked to do anything.
  #
  # Not defensive programming for its own sake: some MLX failures do not
  # come back as errors. A missing metallib aborts the process (exit 255,
  # no Ruby exception, nothing to rescue), because the failure is raised as
  # a C++ exception during device initialization and Rust cannot catch a
  # foreign exception. The remedy for the failures we can foresee is to
  # refuse before touching MLX, with a message that says what to do.
  #
  # See docs/plan.md, "the boundary is not closed": what is checked here is
  # the known list, not a guarantee.
  module Preflight
    # MLX finds its Metal kernels through dladdr, so the metallib must sit
    # beside the library holding the MLX symbols: the extension bundle.
    METALLIB = File.expand_path("mlx.metallib", __dir__)

    module_function

    def check!
      return if File.exist?(METALLIB)

      raise Torobi::EngineUnavailable,
            "MLX's metallib is not beside the extension (expected #{METALLIB}). " \
            "Without it MLX aborts the process rather than raising, so this " \
            "refuses first. Run `rake metallib` in a checkout, or reinstall the gem."
    end
  end
end
