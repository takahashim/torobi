# frozen_string_literal: true

require_relative "mlx_prebuilt"

module Torobi
  # MLX finds its Metal kernels through dladdr: `mlx.metallib` must sit
  # beside whichever library holds the MLX symbols, unless it is told where
  # to look first. A source install puts it there (`ext/torobi/extconf.rb`,
  # `rake metallib`); a platform gem cannot ship the 129 MB file, so on
  # first use it is fetched into a writable cache and MLX is pointed at it
  # (`mlx_metal_set_metallib_path`).
  #
  # macOS only. Linux compiles its CUDA kernels at run time and has no such
  # file.
  module Metallib
    module_function

    # Points MLX at the kernels and returns where they are.
    #
    # The copy a checkout put beside the bundle wins; otherwise they are
    # fetched into the cache the pin names, which is writable, and MLX is
    # told to look there (`mlx_metal_set_metallib_path`). That is what lets
    # a platform gem work from a directory it cannot write: its own.
    # Raises EngineUnavailable when they are missing and cannot be fetched,
    # which is the same failure the probe below would find, said sooner.
    def ensure!(io: $stderr)
      path = resolved(io:)
      Native.set_metallib_path(path)
      path
    rescue MlxPrebuilt::Refused => e
      raise EngineUnavailable, "MLX's Metal kernels could not be obtained: #{e.message}"
    end

    # Where the kernels are, fetching them if a checkout did not leave a
    # copy beside the bundle and none has been fetched before. The paths
    # are parameters so the order can be held to without a filesystem.
    def resolved(io: $stderr, beside: Preflight::METALLIB, cached: cached_path)
      return beside if File.file?(beside)
      return cached if File.file?(cached)

      MlxPrebuilt.fetch_metallib(into: File.dirname(cached), io:)
    end

    # The cache the pin names, per release: one fetch, reused by every
    # session after it, and never written beside a read-only gem.
    def cached_path
      File.join(MlxPrebuilt.cache_dir, "mlx.metallib")
    end
  end
end
