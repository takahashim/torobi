#!/usr/bin/env ruby
# frozen_string_literal: true

# Records what a published checkpoint actually holds, so the graph Torobi
# builds can be held to it on a machine that does not have the model.
#
# The artifact is versioned and committed (docs/plan.md section 12): the
# comparison runs everywhere, and regenerating it is a deliberate act that
# fails closed when the checkpoint is not there.
#
#   ruby tools/inventory.rb <checkpoint-dir> <out.json>
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "torobi"
require "json"
require "time"

dir, out = ARGV
abort "usage: inventory.rb <checkpoint-dir> <out.json>" unless dir && out
%w[config.json model.safetensors].each do |name|
  path = File.join(dir, name)
  abort "no #{name} in #{dir}: this cannot be generated here" unless File.exist?(path)
end

config = JSON.parse(File.read(File.join(dir, "config.json")))
tensors = File.open(File.join(dir, "model.safetensors"), "rb") do |f|
  header = JSON.parse(f.read(f.read(8).unpack1("Q<")))
  header.reject { |k, _| k == "__metadata__" }
        .transform_values { |v| { "shape" => v["shape"], "dtype" => v["dtype"] } }
end

File.write(out, "#{JSON.pretty_generate({
  "schema_version" => 1,
  # The Hub cache spells a repo "models--owner--name"; say it the way a
  # person would.
  "source" => File.basename(File.dirname(File.dirname(dir)))
               .delete_prefix("models--").sub("--", "/"),
  "revision" => File.basename(dir),
  "generated_at" => Time.now.utc.iso8601,
  "config" => config,
  "parameters" => tensors.sort.to_h
})}\n")
puts "wrote #{out}: #{tensors.size} tensors from #{File.basename(dir)}"
