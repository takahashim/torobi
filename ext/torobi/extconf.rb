# frozen_string_literal: true

require "mkmf"
require "rb_sys/mkmf"
require_relative "mlx_prebuilt"

# Before cargo runs, so that mlx-sys takes its first branch (a prebuilt
# directory it was given) rather than its third (fetching one itself,
# with nothing checking what arrives). See ext/torobi/mlx_prebuilt.rb.
prebuilt =
  begin
    MlxPrebuilt.ensure!
  rescue MlxPrebuilt::Refused => e
    abort "torobi: #{e.message}"
  end

create_rust_makefile("torobi/torobi")

# MLX finds its Metal kernels through dladdr: mlx.metallib must sit beside
# the library that holds the MLX symbols, which is this extension. mlx-sys
# drops it next to the built artifact, so install it from there, alongside
# the bundle. Without it MLX ends the process rather than raising, which is
# why Torobi::Preflight refuses first (docs/plan.md section 4.1).
File.open("Makefile", "a") do |makefile|
  # Exported rather than set in this process: make spawns cargo, and this
  # process is gone by then.
  makefile.puts("\nexport MLX_PREBUILT_PATH := #{prebuilt}")
  makefile.puts(<<~MAKE)

    install-so: install-metallib
    install-metallib: $(RUSTLIB)
    \t$(ECHO) installing mlx.metallib to $(RUBYARCHDIR)
    \t$(Q) $(MAKEDIRS) $(RUBYARCHDIR)
    \t$(Q) $(COPY) "$(dir $(RUSTLIB))mlx.metallib" $(RUBYARCHDIR)
  MAKE
end
