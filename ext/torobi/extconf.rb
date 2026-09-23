# frozen_string_literal: true

require "mkmf"
require "rb_sys/mkmf"
require_relative "mlx_prebuilt"

# Before cargo runs. Upstream mlx-sys has no variable of its own for
# "use this MLX", so it is told through a generated CMake toolchain file,
# and this is what fetched the MLX that file names and wrote the file.
# See ext/torobi/mlx_prebuilt.rb.
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
  makefile.puts("export CMAKE_TOOLCHAIN_FILE := #{MlxPrebuilt.toolchain_file(prefix)}")
  # mlx-sys links MLX's archive by name, and with a system MLX that
  # archive is in the prefix rather than in mlx-sys's build tree. One
  # search path, appended so a caller's RUSTFLAGS survive.
  makefile.puts("export RUSTFLAGS := $(RUSTFLAGS) -L native=#{MlxPrebuilt.link_dir(prefix)}")

  # The rest is Metal's: CUDA compiles its kernels at run time and ships
  # no metallib to install beside the bundle.
  next unless MlxPrebuilt.metal?

  # Where mlx-sys looks for the kernels and decides whether they are
  # there. Pointing it at the prefix is what keeps a pre-built MLX from
  # being mistaken for a missing one and warned about.
  makefile.puts("export MLX_RS_METAL_PATH := #{MlxPrebuilt.link_dir(prefix)}")
  makefile.puts(<<~MAKE)

    install-so: install-metallib
    install-metallib: $(RUSTLIB)
    \t$(ECHO) installing mlx.metallib to $(RUBYARCHDIR)
    \t$(Q) $(MAKEDIRS) $(RUBYARCHDIR)
    \t$(Q) $(COPY) "#{MlxPrebuilt.metallib(prefix)}" $(RUBYARCHDIR)
  MAKE
end
