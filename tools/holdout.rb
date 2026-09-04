#!/usr/bin/env ruby
# frozen_string_literal: true

# Splitting a training file, so that a run can be measured on the
# distribution it was trained on.
#
# The first two runs of `experiments/embed_retrieval.rb` both made the
# model worse on NLPJournal (0.7313 to 0.5610, then to 0.6684 with mined
# negatives). Two very different things look like that:
#
#   the training damaged the model
#   the training worked, and NLPJournal is not what it worked on
#
# auto-wiki-qa is synthetic Wikipedia QA and NLPJournal is academic
# papers, so the second is entirely possible, and nothing measured so
# far can tell them apart. A held-out slice of the training file can:
# **if the number goes up here and down there, the model moved rather
# than broke** (docs/plan.md 15.65).
#
#   ruby tools/holdout.rb <pairs.jsonl> <train.jsonl> <eval.json> [pairs]
#
# **The split is by document, not by pair.** auto-wiki-qa asks several
# questions of one passage, so splitting rows would put a passage on
# both sides and the held-out query would be asking about something the
# run had trained on. Every query of a passage goes where the passage
# goes.
#
# **Mined negatives are dropped from the training half.** They were
# mined against the whole file and some of them are now held-out
# documents, which the run would then have read. Mine again on what this
# writes:
#
#   ruby tools/holdout.rb runs/train10k.jsonl runs/train9k.jsonl runs/held.json
#   ruby tools/mine_negatives.rb "$RURI" runs/train9k.jsonl runs/mined9k.jsonl 2
#
# The corpus is the held-out documents and nothing else, so it is a
# smaller haystack than a benchmark's and the number will be higher for
# that reason alone. It is worth comparing against itself over a run,
# not against NLPJournal.

require "json"
require "fileutils"

# The same default as the experiment's, so the two shuffles are talking
# about the same file.
SEED = Integer(ENV.fetch("SEED", 20_260_904))

def rows_of(path)
  File.readlines(path).reject { |line| line.strip.empty? }.map { |line| JSON.parse(line) }
end

# The documents to hold out: whole groups, until there are enough pairs.
#
# Shuffled, because a file is usually in some order that means something
# (auto-wiki-qa is in article order), and the last thousand rows of it
# are not a sample of it.
def held_out(by_document, wanted)
  taken = 0
  by_document.keys.shuffle(random: Random.new(SEED)).take_while do |document|
    keep = taken < wanted
    taken += by_document.fetch(document).size
    keep
  end
end

# The bundle `experiments/embed_retrieval.rb` measures against: queries,
# a corpus, and which document answers which query. One entry per
# distinct document, so nothing in the corpus is another entry's twin.
def bundle_of(pairs, documents)
  at = documents.each_with_index.to_h
  { "queries" => pairs.each_with_index.map do |row, i|
                   { "id" => "q#{i}", "ids" => row.fetch("query_ids") }
                 end,
    "corpus" => documents.each_with_index.map do |ids, i|
      { "id" => "d#{i}", "ids" => ids }
    end,
    "qrels" => pairs.each_with_index.to_h do |row, i|
      ["q#{i}", { "d#{at.fetch(row.fetch("text_ids"))}" => 1 }]
    end }
end

def main
  source, train_out, eval_out, wanted = ARGV
  unless source && train_out && eval_out
    abort "usage: holdout.rb <pairs.jsonl> <train.jsonl> <eval.json> [pairs]"
  end

  wanted = (wanted || 1000).to_i
  # Before anything is read, as `tools/retrieval_pairs.rb` learned: doing
  # the work and then finding there is nowhere to put it is the same
  # mistake whatever the work was.
  [train_out, eval_out].each { |path| FileUtils.mkdir_p(File.dirname(File.expand_path(path))) }
  pairs = rows_of(source)
  by_document = pairs.group_by { |row| row.fetch("text_ids") }
  documents = held_out(by_document, wanted)
  if documents.size == by_document.size
    abort "#{wanted} pairs is the whole file (#{pairs.size} pairs over " \
          "#{by_document.size} documents); hold out fewer"
  end

  held = documents.to_h { |ids| [ids, true] }
  evaluating, training = pairs.partition { |row| held.key?(row.fetch("text_ids")) }
  File.open(train_out, "w") do |file|
    # Without the negatives: they were mined against the whole file, and
    # some of them are held-out documents now.
    training.each { |row| file.puts JSON.generate(row.except("negative_ids")) }
  end
  File.write(eval_out, "#{JSON.generate(bundle_of(evaluating, documents))}\n")

  puts "#{pairs.size} pairs over #{by_document.size} documents"
  puts "  #{train_out}: #{training.size} pairs, negatives dropped (mine again)"
  puts "  #{eval_out}: #{evaluating.size} queries over #{documents.size} documents"
end

main if $PROGRAM_NAME == __FILE__
