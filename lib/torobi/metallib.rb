# frozen_string_literal: true

require_relative "mlx_prebuilt"

module Torobi
  # MLX finds its Metal kernels through dladdr: `mlx.metallib` must sit
  # beside whichever library holds the MLX symbols. A source install puts
  # it there (`ext/torobi/extconf.rb`); a platform gem cannot, because the
  # file is 129 MB and most of the package, so it is fetched on first use
  # and left beside the installed bundle.
  #
  # macOS only, and only when it is not already there: a source install, a
  # checkout after `rake metallib`, and every session after the first do
  # nothing. Linux compiles its CUDA kernels at run time and has no such
  # file.
  module Metallib
    module_function

    # Puts the kernels where MLX will look, fetching them if they are not
    # there, and returns the path. Raises EngineUnavailable when they are
    # missing and cannot be fetched, which is the same failure the probe
    # below would find, said sooner and with the reason.
    def ensure!(io: $stderr)
      target = Preflight::METALLIB
      return target if File.file?(target)

      MlxPrebuilt.fetch_metallib(into: File.dirname(target), io:)
    rescue MlxPrebuilt::Refused => e
      raise EngineUnavailable,
            "MLX's Metal kernels are not beside the extension and could not be " \
            "fetched: #{e.message}"
    end
  end
end
