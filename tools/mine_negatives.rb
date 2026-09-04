#!/usr/bin/env ruby
# frozen_string_literal: true

# Finding the documents a model already confuses with the answer.
#
# In-batch negatives are the other queries' documents, and against a
# retriever that is already good they are trivial: the first run of
# `experiments/embed_retrieval.rb` drove its loss to 0.005 and made the
# model worse, because nothing it was being asked was hard
# (docs/plan.md section 15.57). A mined negative is a document that
# looks like an answer to this query and is not one, which is the only
# thing left to learn from.
#
#   ruby tools/mine_negatives.rb <encoder-dir> <pairs.jsonl> <out.jsonl> [count]
#
# Both files are token ids: this reads what `tools/retrieval_pairs.rb`
# wrote and writes the same with `negative_ids` added.
#
# **The model does the mining.** `Session#forward` embeds every query
# and every passage in the file with the encoder the run will start
# from, which is what makes the negatives hard for *that* model rather
# than for some other one.
#
# It is quadratic: every query is scored against every passage. That is
# the honest shape of the job and the reason to mine a slice rather than
# a corpus (a hundred thousand against a hundred thousand is a hundred
# thousand million dot products, which is not a Ruby program). At ten
# thousand each it is minutes.

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "torobi"
require "json"

SEQ = Integer(ENV.fetch("SEQ", 192))
BATCH = Integer(ENV.fetch("BATCH", 16))
# How many of the nearest are skipped before taking negatives. The very
# nearest documents to a query are often other correct answers that the
# dataset did not label, and teaching a model that a right answer is
# wrong is worse than teaching it nothing.
SKIP = Integer(ENV.fetch("SKIP", 1))

def rows_of(path)
  File.readlines(path).reject { |line| line.strip.empty? }.map { |line| JSON.parse(line) }
end

# The encoder, read-only: nothing is trained here, so the run needs no
# loss and asks for no seed (docs/plan.md section 15.47).
def reading(config)
  model = Torobi::Models::ModernBERT.embedder(config, seq: SEQ, pooling: :mean,
                                              encoder_prefix: "")
  Torobi::GraphConfig.new(models: { m: model }, train: [])
end

def embed(session, config, ids, width)
  ids.each_slice(BATCH).flat_map do |slice|
    batch = Torobi::Models::ModernBERT.batch(config, slice, seq: SEQ, pooling: :mean)
    session.forward(batch).fetch("m.embedding").to_a.each_slice(width).to_a
  end
end

def dot(a, b)
  total = 0.0
  a.each_index { |i| total += a[i] * b[i] }
  total
end

def main
  encoder, source, out, count = ARGV
  unless encoder && source && out
    abort "usage: mine_negatives.rb <encoder-dir> <pairs.jsonl> <out.jsonl> [count]"
  end

  count = (count || 1).to_i
  config = Torobi::Models::ModernBERT.from_config_file(File.join(encoder, "config.json"))
  width = config.hidden_size
  pairs = rows_of(source)
  puts "#{pairs.size} pairs, mining #{count} negative(s) each"

  mined = Torobi::Session.open(
    reading(config), pretrained: { m: File.join(encoder, "model.safetensors") }
  ) do |session|
    documents = embed(session, config, pairs.map { |row| row.fetch("text_ids") }, width)
    queries = embed(session, config, pairs.map { |row| row.fetch("query_ids") }, width)
    puts "embedded #{queries.size} queries and #{documents.size} passages"

    pairs.each_with_index.map do |row, i|
      # The nearest passages that are not this query's own answer. Taken
      # after `SKIP` of them, because the nearest is often another right
      # answer nobody labelled.
      ranked = documents.each_index.reject { |j| j == i }
                        .max_by(count + SKIP) { |j| dot(queries[i], documents[j]) }
      row.merge("negative_ids" => ranked.drop(SKIP).first(count)
                                        .map { |j| pairs[j].fetch("text_ids") })
    end
  end

  File.open(out, "w") { |file| mined.each { |row| file.puts JSON.generate(row) } }
  short = mined.count { |row| row.fetch("negative_ids").size < count }
  puts "wrote #{out}: #{mined.size} pairs" \
       "#{", #{short} of them short of #{count}" if short.positive?}"
end

main if $PROGRAM_NAME == __FILE__
