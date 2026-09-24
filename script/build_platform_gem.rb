#!/usr/bin/env ruby
# frozen_string_literal: true

# Assemble a precompiled platform gem from the per-Ruby-ABI extensions
# staged under lib/torobi/<ruby_minor>/torobi.{bundle,so}.
#
# Usage: ruby script/build_platform_gem.rb <gem-platform>
#   e.g. ruby script/build_platform_gem.rb arm64-darwin
#
# Run by .github/workflows/release.yml after downloading the compile
# artifacts, and by `rake platform_gem` for the one ABI this machine built.
# The gem ships the compiled extensions and declares no C extension, so
# `gem install` does not recompile or need a Rust toolchain; lib/torobi.rb
# selects lib/torobi/<minor>/torobi at require time.

require "rubygems"
require "rubygems/package"

platform = ARGV[0]
abort "usage: build_platform_gem.rb <gem-platform>" if platform.nil? || platform.empty?

root = File.expand_path("..", __dir__)
Dir.chdir(root)

spec = Gem::Specification.load("torobi.gemspec")

libs = Dir["lib/torobi/*/torobi.{so,bundle}"]
abort "no precompiled extensions under lib/torobi/*/ - stage them first" if libs.empty?

# The platform gem carries what runs, not the sources it was built from:
# the crates and the extension's own source are only needed to compile.
spec.platform = platform
spec.extensions = []
spec.files = spec.files.reject do |file|
  file.start_with?("ext/", "engine/", "Cargo.toml", "Cargo.lock")
end
spec.files += libs
spec.files += Dir["lib/include/**/*"].select { |file| File.file?(file) }
spec.files.uniq!

# Bound the Ruby versions this gem serves, one subdir per ABI minor.
abis = libs.map { |path| File.basename(File.dirname(path)) }
           .uniq.sort_by { |version| version.split(".").map(&:to_i) }
lo = abis.first.split(".").map(&:to_i)
hi = abis.last.split(".").map(&:to_i)
spec.required_ruby_version = [">= #{lo.join(".")}.0", "< #{hi[0]}.#{hi[1] + 1}.dev"]

puts "Building platform gem for #{platform}"
puts "  ABIs:     #{abis.join(", ")}"
puts "  ruby:     #{spec.required_ruby_version}"
libs.each { |path| puts "  binary:   #{path}" }

gem = Gem::Package.build(spec)
puts "Built #{gem} (#{(File.size(gem) / 1024.0 / 1024).round(1)} MB)"
