# frozen_string_literal: true

require_relative "lib/torobi/version"

Gem::Specification.new do |spec|
  spec.name    = "torobi"
  spec.version = Torobi::VERSION
  spec.authors = ["takahashim"]
  spec.email   = ["takahashimm@gmail.com"]

  spec.summary     = "A slow-cooking training framework for Apple Silicon: " \
                     "declare models in Ruby, train them on MLX."
  spec.description = "Torobi describes models and training objectives in a Ruby Graph DSL, " \
                     "compiles them to an immutable GraphConfig IR, and executes training " \
                     "through an MLX-backed engine. Aimed at fine-tuning and distilling " \
                     "known architectures locally, without owning the training loop."
  spec.homepage = "https://github.com/takahashim/torobi"
  spec.license  = "MIT"
  spec.required_ruby_version = ">= 3.2"

  # The extension is built from source at install time, so the crates and
  # the workspace manifest ship with it. docs/vendoring.md is in here
  # because it is the record of what the engine is built against.
  spec.files = Dir["lib/**/*.rb", "config/ops.yml", "ext/**/*.{rs,rb,toml}",
                   "engine/**/*.{rs,toml}", "Cargo.toml", "Cargo.lock",
                   "README.md", "LICENSE", "docs/plan.md", "docs/vendoring.md"]
  spec.require_paths = ["lib"]
  spec.extensions = ["ext/torobi/extconf.rb"]

  spec.add_development_dependency "minitest", "~> 5.0"
  spec.add_development_dependency "rake-compiler", "~> 1.2"
  spec.add_development_dependency "rb_sys", "~> 0.9"
  spec.add_development_dependency "rake", "~> 13.0"
end
