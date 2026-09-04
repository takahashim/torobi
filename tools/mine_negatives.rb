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
# It is quadratic: every query is scored against every passage. **The
# scoring is a matmul, so it happens where matmuls happen.** Measured on
# 510 pairs of ruri-v3-30m, the same answers to the last index: 2.943 s
# of Ruby dot products against 0.039 s of MLX, which is 75x
# (docs/plan.md 15.61).
#
# And **padding is work**: the queries were eighteen tokens long and
# were being padded to 192, which is nine tenths of that half of the
# embedding spent on nothing (docs/plan.md 15.62). The graph is now
# built for no particular length at all, so each batch is padded to its
# own longest row and nothing is padded to a number somebody chose
# (docs/plan.md 15.63).
#
# Where that leaves it, on the same 510 pairs and the same machine
# (3.5 s of wall clock, against about nine before any of this):
#
#   embedding  3.0 s    (0.30 queries + 2.70 documents)
#   scoring    0.039 s
#   -> ten thousand pairs: about 75 seconds, most of it the documents
#   -> a hundred thousand: about half an hour, most of it the scoring
#
# The queries cost 2.81 s when they were padded to 192 and cost 0.30 s
# now, which is 9.4x for taking the padding out. Thirteen batch shapes
# rather than one costs about 0.04 s, measured cold: a new shape is
# cheap enough that sorting by length and padding to the batch is worth
# it even for a tool that runs once.
#
# So ten thousand is a coffee and a hundred thousand is not. Past that
# the quadratic half is the expensive half again and wants the picking
# to happen on the device, which is an `argpartition` this engine does
# not have; until then, mine a slice.

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "torobi"
require "json"

# Measured at 16, 32, 64 and 128 on ruri-v3-30m: 2.72 s to 2.81 s for the
# same 510 documents, which is no difference at all. The sequence is what
# costs, so this is only about memory. There is no SEQ: the rows arrive
# tokenized and the batch is as long as its longest row.
BATCH = Integer(ENV.fetch("BATCH", 32))
# How many of the nearest are skipped before taking negatives. The very
# nearest documents to a query are often other correct answers that the
# dataset did not label, and teaching a model that a right answer is
# wrong is worse than teaching it nothing.
SKIP = Integer(ENV.fetch("SKIP", 1))
# How many queries are scored at once. The scores of one chunk against
# every document come back to Ruby as numbers, so this is what decides
# how many are in memory: 256 against ten thousand documents is two and
# a half million of them. Against a hundred thousand it is ten times
# that, and the answer is a smaller chunk rather than a bigger machine.
CHUNK = Integer(ENV.fetch("CHUNK", 256))

def rows_of(path)
  File.readlines(path).reject { |line| line.strip.empty? }.map { |line| JSON.parse(line) }
end

# The encoder, read-only: nothing is trained here, so the run needs no
# loss and asks for no seed (docs/plan.md section 15.47).
def reading(config)
  model = Torobi::Models::ModernBERT.embedder(config, seq: nil, pooling: :mean,
                                              encoder_prefix: "")
  Torobi::GraphConfig.new(models: { m: model }, train: [])
end

# Every text, embedded, in the order it arrived.
#
# **Padding is work.** A query of eighteen tokens costs what 192 tokens
# cost if it is padded to 192, and mean pooling then throws the padding
# away. So the rows are sorted by length before they are cut into
# batches, and each batch is padded to its own longest row: rows of a
# like length end up together, which is what makes that number small.
#
# One graph and one session, because the graph is built for no
# particular length (`seq: nil`).
def embed(session, config, ids, width)
  found = Array.new(ids.size)
  ids.each_with_index.sort_by { |row, _| row.size }.each_slice(BATCH) do |slice|
    seq = slice.map { |row, _| row.size }.max
    batch = Torobi::Models::ModernBERT.batch(config, slice.map(&:first), seq:,
                                             pooling: :mean)
    vectors = session.forward(batch).fetch("m.embedding").to_a.each_slice(width).to_a
    slice.each_with_index { |(_, at), i| found[at] = vectors[i] }
  end
  found
end

# Every query against every document, as the one matmul it is.
#
# The vectors are already of length one, so this is cosine similarity,
# and the graph holds nothing: it is opened to be a calculator.
def scorer(width, documents)
  graph = Torobi.graph do |g|
    q = g.input :queries, [nil, width]
    d = g.input :documents, [documents, width]
    g.output :scores, g.matmul(q, d.transpose(axes: [1, 0]))
  end
  Torobi::GraphConfig.new(models: { m: graph }, train: [])
end

# The `count` nearest documents to each query, by index, skipping its own
# and however many the caller wants left alone.
#
# In chunks because the answer is one number per query per document and
# all of them at once is a matrix nobody needs whole.
def nearest(queries, documents, width, count:, skip:)
  flat = documents.flatten
  held = Torobi::TensorData.from_a([documents.size, width], flat)
  Torobi::Session.open(scorer(width, documents.size), weights: { params: {} }) do |session|
    queries.each_slice(CHUNK).with_index.flat_map do |slice, chunk|
      scores = session.forward(
        { queries: Torobi::TensorData.from_a([slice.size, width], slice.flatten),
          documents: held }
      ).fetch("m.scores").to_a
      scores.each_slice(documents.size).with_index.map do |row, i|
        own = (chunk * CHUNK) + i
        row.each_index.reject { |j| j == own }
           .max_by(count + skip) { |j| row[j] }
           .drop(skip).first(count)
      end
    end
  end
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

  documents, queries = Torobi::Session.open(
    reading(config), pretrained: { m: File.join(encoder, "model.safetensors") }
  ) do |session|
    %w[text_ids query_ids].map do |key|
      embed(session, config, pairs.map { |row| row.fetch(key) }, width)
    end
  end
  puts "embedded #{queries.size} queries and #{documents.size} passages"

  # The nearest passages that are not this query's own answer, taken
  # after `SKIP` of them: the nearest is often another right answer
  # nobody labelled, and teaching a model that a right answer is wrong is
  # worse than teaching it nothing.
  nearby = nearest(queries, documents, width, count:, skip: SKIP)
  mined = pairs.each_with_index.map do |row, i|
    row.merge("negative_ids" => nearby[i].map { |j| pairs[j].fetch("text_ids") })
  end

  File.open(out, "w") { |file| mined.each { |row| file.puts JSON.generate(row) } }
  short = mined.count { |row| row.fetch("negative_ids").size < count }
  puts "wrote #{out}: #{mined.size} pairs" \
       "#{", #{short} of them short of #{count}" if short.positive?}"
end

main if $PROGRAM_NAME == __FILE__
