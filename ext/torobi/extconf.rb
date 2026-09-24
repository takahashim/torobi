# frozen_string_literal: true

require "mkmf"
require "rb_sys/mkmf"
require_relative "../../lib/torobi/mlx_prebuilt"

# Before cargo runs: fetch the pre-built MLX and check it, so the engine's
# build.rs has a prefix to bind and link (lib/torobi/mlx_prebuilt.rb).
prefix =
  begin
    MlxPrebuilt.ensure!
  rescue MlxPrebuilt::Refused => e
    abort "torobi: #{e.message}"
  end

create_rust_makefile("torobi/torobi")

# MLX finds its Metal kernels through dladdr: mlx.metallib must sit beside
# the library that holds the MLX symbols, which is this extension. It is
# installed from the prefix, where the archive put it. Without it MLX ends
# the process rather than raising, which is why Torobi::Preflight refuses
# first (docs/plan.md section 4.1).
File.open("Makefile", "a") do |makefile|
  # Exported rather than set in this process: make spawns cargo, and this
  # process is gone by then.
  # What engine/build.rs reads: the headers it generates bindings from and
  # the archives it links.
  makefile.puts("\nexport TOROBI_MLX_PREFIX := #{prefix}")

  # The rest is Metal's: CUDA compiles its kernels at run time and ships
  # no metallib to install beside the bundle.
  next unless MlxPrebuilt.metal?

  makefile.puts(<<~MAKE)

    install-so: install-metallib
    install-metallib: $(RUSTLIB)
    \t$(ECHO) installing mlx.metallib to $(RUBYARCHDIR)
    \t$(Q) $(MAKEDIRS) $(RUBYARCHDIR)
    \t$(Q) $(COPY) "#{MlxPrebuilt.metallib(prefix)}" $(RUBYARCHDIR)
  MAKE
end
