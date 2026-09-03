# frozen_string_literal: true

require "mkmf"
require "rb_sys/mkmf"

create_rust_makefile("torobi/torobi")

# MLX finds its Metal kernels through dladdr: mlx.metallib must sit beside
# the library that holds the MLX symbols, which is this extension. mlx-sys
# drops it next to the built artifact, so install it from there, alongside
# the bundle. Without it MLX ends the process rather than raising, which is
# why Torobi::Preflight refuses first (docs/plan.md section 4.1).
File.open("Makefile", "a") do |makefile|
  makefile.puts(<<~MAKE)

    install-so: install-metallib
    install-metallib: $(RUSTLIB)
    \t$(ECHO) installing mlx.metallib to $(RUBYARCHDIR)
    \t$(Q) $(MAKEDIRS) $(RUBYARCHDIR)
    \t$(Q) $(COPY) "$(dir $(RUSTLIB))mlx.metallib" $(RUBYARCHDIR)
  MAKE
end
