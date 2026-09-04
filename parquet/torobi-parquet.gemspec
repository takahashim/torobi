# frozen_string_literal: true

require_relative "lib/torobi/parquet/version"

Gem::Specification.new do |spec|
  spec.name    = "torobi-parquet"
  spec.version = Torobi::Parquet::VERSION
  spec.authors = ["takahashim"]
  spec.email   = ["takahashimm@gmail.com"]

  spec.summary     = "The parquet a dataset arrives as, read in Ruby and nothing else."
  spec.description = "A reader for the part of parquet that datasets are written in: " \
                     "flat columns of strings and numbers, snappy or nothing, " \
                     "dictionary-encoded data pages. No arrow, no native extension, " \
                     "no dependencies. What it does not implement, it refuses by name."
  spec.homepage = "https://github.com/takahashim/torobi"
  spec.license  = "MIT"
  spec.required_ruby_version = ">= 3.2"
  spec.metadata = { "rubygems_mfa_required" => "true" }

  spec.files = Dir["lib/**/*.rb", "README.md"]
  spec.require_paths = ["lib"]
end
