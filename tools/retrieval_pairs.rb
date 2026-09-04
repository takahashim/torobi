#!/usr/bin/env ruby
# frozen_string_literal: true

# The same preprocessing as `tools/retrieval_pairs.py`, in Ruby, for data
# that is already yours.
#
# Why both: the Python one reads parquet, which is how a dataset arrives
# from the Hub and is not something Ruby reaches without Homebrew's
# arrow. This one reads JSON Lines, which is what your own data looks
# like, and then nothing about the pipeline needs another language.
#
# The tokenizer is the same tokenizer. `tokenizers` is HuggingFace's,
# with the same `tokenizer.json` and the same ids: measured on ruri-v3,
# prefixes and truncation included, the two agree exactly.
#
#   gem install tokenizers
#
# **Not a dependency of Torobi.** The library is handed ids and does not
# tokenize (docs/plan.md 15.19); this is a script beside it, and asks for
# what it needs the way the Python one asks `uv`.
#
#   ruby tools/retrieval_pairs.rb --tokenizer <ruri>/tokenizer.json \
#     --pairs mine.jsonl --query-key question --text-key body \
#     --seq 192 --out train.jsonl
#
#   ruby tools/retrieval_pairs.rb --tokenizer <ruri>/tokenizer.json \
#     --queries q.jsonl --corpus c.jsonl --qrels qrels.jsonl --out eval.json
#
# **The prefixes matter.** ruri-v3 is trained with the task written into
# the text, and its pooling config says the prefix is inside the average.
# Leaving them out is a different model; `--query-prefix ""` says so on
# purpose for a model that wants none.

require "json"
require "optparse"
require "fileutils"

begin
  require "tokenizers"
rescue LoadError
  abort "this needs the tokenizers gem (gem install tokenizers); " \
        "it is HuggingFace's, and gives the same ids as the Python one"
end

options = { seq: 256, query_prefix: "検索クエリ: ", text_prefix: "検索文書: ",
            query_key: "query", text_key: "text", id_key: "id", rows: 100_000 }
OptionParser.new do |o|
  o.banner = "usage: retrieval_pairs.rb --tokenizer <tokenizer.json> --out <path> [...]"
  o.on("--tokenizer PATH") { |v| options[:tokenizer] = v }
  o.on("--out PATH") { |v| options[:out] = v }
  o.on("--seq N", Integer) { |v| options[:seq] = v }
  o.on("--rows N", Integer) { |v| options[:rows] = v }
  o.on("--pairs PATH", "JSON Lines of (query, text) rows") { |v| options[:pairs] = v }
  o.on("--query-key KEY") { |v| options[:query_key] = v }
  o.on("--text-key KEY") { |v| options[:text_key] = v }
  o.on("--title-key KEY", "prepended to the text where a row has one") do |v|
    options[:title_key] = v
  end
  o.on("--queries PATH", "JSON Lines of {id, text}") { |v| options[:queries] = v }
  o.on("--corpus PATH", "JSON Lines of {id, text, title?}") { |v| options[:corpus] = v }
  o.on("--qrels PATH", "JSON Lines of {query-id, corpus-id, score}") { |v| options[:qrels] = v }
  o.on("--query-prefix TEXT") { |v| options[:query_prefix] = v }
  o.on("--text-prefix TEXT") { |v| options[:text_prefix] = v }
end.parse!

abort "--tokenizer is required" unless options[:tokenizer]
abort "--out is required" unless options[:out]

# Before anything is read: tokenizing a hundred thousand rows and then
# finding there is nowhere to put them is minutes for nothing.
FileUtils.mkdir_p(File.dirname(File.expand_path(options[:out])))

tokenizer = Tokenizers.from_file(options[:tokenizer])
tokenizer.enable_truncation(options[:seq])

encode = lambda do |text, prefix|
  tokenizer.encode("#{prefix}#{text}").ids
end

def lines_of(path)
  File.foreach(path).filter_map do |line|
    JSON.parse(line) unless line.strip.empty?
  end
end

# A document is what somebody would index, and a corpus that has titles
# indexes them.
def document(row, text_key, title_key)
  title = title_key && row[title_key]
  title.to_s.empty? ? row.fetch(text_key) : "#{title}\n#{row.fetch(text_key)}"
end

if options[:pairs]
  rows = lines_of(options[:pairs]).first(options[:rows])
  File.open(options[:out], "w") do |out|
    rows.each do |row|
      unless row.key?(options[:query_key]) && row.key?(options[:text_key])
        abort "no #{options[:query_key].inspect} and #{options[:text_key].inspect} " \
              "in #{row.keys.inspect}"
      end

      out.puts JSON.generate(
        "query_ids" => encode.call(row.fetch(options[:query_key]), options[:query_prefix]),
        "text_ids" => encode.call(document(row, options[:text_key], options[:title_key]),
                                  options[:text_prefix])
      )
    end
  end
  puts "wrote #{options[:out]}: #{rows.size} pairs from #{options[:pairs]}"
elsif options[:queries] && options[:corpus] && options[:qrels]
  relevant = Hash.new { |h, k| h[k] = {} }
  lines_of(options[:qrels]).each do |row|
    relevant[row.fetch("query-id").to_s][row.fetch("corpus-id").to_s] = row.fetch("score").to_i
  end
  bundle = {
    "queries" => lines_of(options[:queries])
                 .select { |q| relevant.key?(q.fetch(options[:id_key]).to_s) }
                 .map do |q|
                   { "id" => q.fetch(options[:id_key]).to_s,
                     "ids" => encode.call(q.fetch("text"), options[:query_prefix]) }
                 end,
    "corpus" => lines_of(options[:corpus]).map do |c|
      { "id" => c.fetch(options[:id_key]).to_s,
        "ids" => encode.call(document(c, "text", "title"), options[:text_prefix]) }
    end,
    "qrels" => relevant
  }
  File.write(options[:out], "#{JSON.generate(bundle)}\n")
  puts "wrote #{options[:out]}: #{bundle["queries"].size} queries over " \
       "#{bundle["corpus"].size} documents"
else
  abort "one of --pairs, or --queries with --corpus and --qrels"
end
