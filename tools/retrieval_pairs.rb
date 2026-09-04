#!/usr/bin/env ruby
# frozen_string_literal: true

# The same preprocessing as `tools/retrieval_pairs.py`, in Ruby.
#
# It reads JSON Lines, which is what your own data looks like, and
# parquet, which is what a dataset from the Hub arrives as
# (`torobi-parquet`, in this repository, with no dependencies of its
# own). So the pipeline needs no other language, and the Python one
# stays for the parquet this does not read: gzip, pages of the second
# version, nested columns.
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
#     --pairs 'data/*.parquet' --query-key query --text-key text \
#     --title-key title --rows 100000 --seq 192 --out train.jsonl
#
#   ruby tools/retrieval_pairs.rb --tokenizer <ruri>/tokenizer.json \
#     --mteb <snapshot> --seq 192 --out eval.json
#
# A path that ends in `.parquet` (or a glob that matches some) is read
# as parquet; anything else is read as JSON Lines.
#
# **The prefixes matter.** ruri-v3 is trained with the task written into
# the text, and its pooling config says the prefix is inside the average.
# Leaving them out is a different model; `--query-prefix ""` says so on
# purpose for a model that wants none.

require "json"
require "optparse"
require "fileutils"

$LOAD_PATH.unshift File.expand_path("../parquet/lib", __dir__)
require "torobi/parquet"

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
  o.on("--queries PATH", "rows of {id, text}") { |v| options[:queries] = v }
  o.on("--corpus PATH", "rows of {id, text, title?}") { |v| options[:corpus] = v }
  o.on("--qrels PATH", "rows of {query-id, corpus-id, score}") { |v| options[:qrels] = v }
  o.on("--mteb DIR", "a directory of queries/corpus/qrels parquet") { |v| options[:mteb] = v }
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

# How many came back at the cap, said out loud.
#
# **Truncation is silent otherwise, and it is not a small thing.**
# Measured on NLPJournal: the same untouched ruri-v3-130m scores 0.7313
# at a 192-token cap, 0.8851 at 512 and 0.9466 at 1024, against a
# published 0.9645. Three quarters of every introduction had been thrown
# away and nothing said so, which cost a day of reading the difference as
# the model's (docs/plan.md 15.70).
def cut(name, rows, limit)
  at_cap = rows.count { |ids| ids.size >= limit }
  return "" if at_cap.zero?

  said = ", #{at_cap} of #{rows.size} #{name} at the #{limit}-token cap"
  said += " (so this is mostly their first #{limit} tokens)" if at_cap * 2 > rows.size
  said
end

# Rows from wherever they are, by what the path looks like: parquet is
# columnar and a glob of it is several files, JSON Lines is one.
def rows_of(path, columns: nil, limit: nil)
  files = Dir[path]
  return parquet_rows(files, columns:, limit:) if files.any? { |f| f.end_with?(".parquet") }

  File.foreach(path).filter_map { |line| JSON.parse(line) unless line.strip.empty? }
end

def parquet_rows(files, columns: nil, limit: nil)
  rows = []
  files.sort.each do |file|
    Torobi::Parquet.each_row(file, columns:, rows: limit && (limit - rows.size)) do |row|
      rows << row
    end
    break if limit && rows.size >= limit
  end
  limit ? rows.first(limit) : rows
end

# A document is what somebody would index, and a corpus that has titles
# indexes them.
def document(row, text_key, title_key)
  title = title_key && row[title_key]
  title.to_s.empty? ? row.fetch(text_key) : "#{title}\n#{row.fetch(text_key)}"
end

if options[:mteb]
  options[:queries] ||= File.join(options[:mteb], "queries", "*.parquet")
  options[:corpus] ||= File.join(options[:mteb], "corpus", "*.parquet")
  options[:qrels] ||= File.join(options[:mteb], "qrels", "*.parquet")
end

if options[:pairs]
  columns = [options[:query_key], options[:text_key], options[:title_key]].compact
  rows = rows_of(options[:pairs], columns:, limit: options[:rows])
  texts = []
  File.open(options[:out], "w") do |out|
    rows.each do |row|
      unless row.key?(options[:query_key]) && row.key?(options[:text_key])
        abort "no #{options[:query_key].inspect} and #{options[:text_key].inspect} " \
              "in #{row.keys.inspect}"
      end

      text_ids = encode.call(document(row, options[:text_key], options[:title_key]),
                             options[:text_prefix])
      texts << text_ids
      out.puts JSON.generate(
        "query_ids" => encode.call(row.fetch(options[:query_key]), options[:query_prefix]),
        "text_ids" => text_ids
      )
    end
  end
  puts "wrote #{options[:out]}: #{rows.size} pairs from #{options[:pairs]}" \
       "#{cut("texts", texts, options[:seq])}"
elsif options[:queries] && options[:corpus] && options[:qrels]
  relevant = Hash.new { |h, k| h[k] = {} }
  rows_of(options[:qrels]).each do |row|
    relevant[row.fetch("query-id").to_s][row.fetch("corpus-id").to_s] = row.fetch("score").to_i
  end
  bundle = {
    "queries" => rows_of(options[:queries])
                 .select { |q| relevant.key?(q.fetch(options[:id_key]).to_s) }
                 .map do |q|
                   { "id" => q.fetch(options[:id_key]).to_s,
                     "ids" => encode.call(q.fetch("text"), options[:query_prefix]) }
                 end,
    "corpus" => rows_of(options[:corpus]).map do |c|
      { "id" => c.fetch(options[:id_key]).to_s,
        "ids" => encode.call(document(c, "text", "title"), options[:text_prefix]) }
    end,
    "qrels" => relevant
  }
  File.write(options[:out], "#{JSON.generate(bundle)}\n")
  puts "wrote #{options[:out]}: #{bundle["queries"].size} queries over " \
       "#{bundle["corpus"].size} documents" \
       "#{cut("documents", bundle["corpus"].map { |d| d.fetch("ids") }, options[:seq])}"
else
  abort "one of --pairs, or --queries with --corpus and --qrels"
end
