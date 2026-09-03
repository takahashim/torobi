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
#   ruby tools/inventory.rb owner/name <out.json>
#
# The second form reads the same two things from the Hub without the
# weights: a safetensors file says what it holds in a header at the front
# of it, so an inventory of a model costs a config.json and a few tens of
# kilobytes rather than the gigabytes underneath. What it cannot do is
# check numbers, which is the other kind of oracle (`rake oracle:forward`).
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "torobi"
require "json"
require "time"
require "net/http"

where, out = ARGV
abort "usage: inventory.rb <checkpoint-dir|owner/name> <out.json>" unless where && out

# One GET, following the Hub's redirect to storage, of a byte range.
def fetch(url, range, hops: 5)
  abort "#{url} redirects too far" if hops.zero?

  uri = URI.parse(url)
  request = Net::HTTP::Get.new(uri)
  request["Range"] = "bytes=#{range}"
  response = Net::HTTP.start(uri.host, uri.port, use_ssl: true) { |http| http.request(request) }
  case response
  when Net::HTTPRedirection
    # Relative, and to storage on another host: the commit is on the
    # answer that knows which repository this is, which is this one.
    body, = fetch(URI.join(url, response["location"]).to_s, range, hops: hops - 1)
    [body, response["x-repo-commit"]]
  when Net::HTTPSuccess then [response.body, response["x-repo-commit"]]
  else abort "#{url} answered #{response.code} #{response.message}"
  end
end

def from_hub(repo)
  base = "https://huggingface.co/#{repo}/resolve/main"
  config, revision = fetch("#{base}/config.json", "0-1048575")
  length, = fetch("#{base}/model.safetensors", "0-7")
  header, = fetch("#{base}/model.safetensors", "8-#{7 + length.unpack1("Q<")}")
  [JSON.parse(config), JSON.parse(header), repo, revision]
end

def from_directory(dir)
  %w[config.json model.safetensors].each do |name|
    path = File.join(dir, name)
    abort "no #{name} in #{dir}: this cannot be generated here" unless File.exist?(path)
  end
  header = File.open(File.join(dir, "model.safetensors"), "rb") do |f|
    JSON.parse(f.read(f.read(8).unpack1("Q<")))
  end
  # The Hub cache spells a repo "models--owner--name"; say it the way a
  # person would.
  [JSON.parse(File.read(File.join(dir, "config.json"))), header,
   File.basename(File.dirname(dir, 2)).delete_prefix("models--").sub("--", "/"),
   File.basename(dir)]
end

config, header, source, revision =
  File.directory?(where) ? from_directory(where) : from_hub(where)
tensors = header.except("__metadata__")
                .transform_values { |v| { "shape" => v["shape"], "dtype" => v["dtype"] } }

inventory = {
  "schema_version" => 1,
  "source" => source,
  "revision" => revision,
  "generated_at" => Time.now.utc.iso8601,
  "config" => config,
  "parameters" => tensors.sort.to_h
}
File.write(out, "#{JSON.pretty_generate(inventory)}\n")
puts "wrote #{out}: #{tensors.size} tensors from #{source} #{revision}"
