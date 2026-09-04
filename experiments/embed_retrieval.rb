#!/usr/bin/env ruby
# frozen_string_literal: true

# Fine-tuning a retriever: the four pieces, on a real dataset, measured.
#
# The encoder is a published ruri-v3 (a ModernBERT), pooled over the
# tokens each row actually has and normalized. The loss is multiple
# negatives ranking: every other document in the batch is a negative for
# this query, and the only supervision is which pairs were put in
# together. The batch is larger than the machine holds, so it goes
# through a gradient cache. And the metric is nDCG@10 on a benchmark the
# training set does not overlap.
#
#   ruby experiments/embed_retrieval.rb <encoder-dir> <train.jsonl> \
#     <eval.json[,eval.json...]> <run-dir> [steps]
#
# **Measure against more than one.** A run that falls on a benchmark and
# rises on a held-out slice of its own training data has moved rather
# than broken, and either number alone would have been read as a verdict
# (docs/plan.md 15.67). `tools/holdout.rb` makes the second one.
#
# `LORA=8` trains an adapter instead of the whole model, which is the
# knob for how much of what the model already knew survives.
#
# **It resumes.** Run it again against a directory that holds a
# checkpoint and it carries on from there, with the same parameters, the
# same optimizer state, the same randomness and the same place in the
# file. A run that took hours is worth being able to lose the machine
# for a minute.
#
# Both inputs are token ids and nothing else: `tools/retrieval_pairs.py`
# made them, which is where the tokenizer lives (docs/plan.md 15.19).
#
# **Do not train on the benchmark.** The eval bundles are what the
# numbers are reported against; if the pairs came from one, that number
# means nothing. auto-wiki-qa trains, NLPJournal measures, and the
# held-out slice is split off before the negatives are mined.
#
# What it is not is a framework. Section 7.3 says experiments stay plain
# Ruby until dogfooding says what the shape should be, so this is a
# script: the graphs, the loop, and the metric, written out.

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "torobi"
require "json"

# The knobs, from the environment so that a run can be turned without
# editing the description of it, and recorded so that it can be said
# afterwards what was turned.
SEED = Integer(ENV.fetch("SEED", 20_260_904))
# What the pairs were tokenized to. Nothing is padded to it: the graph
# is built for no particular length and every part is padded to its own
# longest row (docs/plan.md 15.63). It is recorded, and it is what the
# exported model says it may be used at.
SEQ = Integer(ENV.fetch("SEQ", 192))
# How many pairs a step learns from. The contrastive signal is the other
# rows, so this is the number that matters most for quality, and the
# gradient cache is what makes it affordable.
PAIRS = Integer(ENV.fetch("PAIRS", 32))
# How many rows are held at once. Memory is set by this; the loss is set
# by PAIRS.
PART = Integer(ENV.fetch("PART", 8))
# How many mined negatives each pair brings. Zero is the in-batch loss,
# where what a query is told apart from is the other queries' documents;
# with a strong encoder those are too easy to say anything, which is
# what the first run of this showed (docs/plan.md section 15.57).
# `tools/mine_negatives.rb` is what puts them in the file.
NEGATIVES = Integer(ENV.fetch("NEGATIVES", 0))
# The peak of the schedule rather than the rate: a fine-tune of a model
# that is already good warms up and comes back down (`Policies::Schedule`).
# 5e-6 rather than the 2e-5 a from-scratch run would take, because what
# this is moving already knows how to retrieve.
LEARNING_RATE = Float(ENV.fetch("LR", 5e-6))
WARMUP = Float(ENV.fetch("WARMUP", 0.05))
# sentence-transformers' default temperature of 0.05, as its reciprocal.
SCALE = Float(ENV.fetch("SCALE", 20.0))
# The rank of a LoRA adapter, or 0 for a full fine-tune.
#
# A full fine-tune of a published retriever moves every weight, and what
# it loses is what it knew that the training set does not say: run 003
# fell from 0.7313 to 0.6684 on NLPJournal while its own distribution
# went up (docs/plan.md 15.65 and 15.67). An adapter is the standard
# answer to that: the update is confined to a low-rank subspace beside
# each weight, and the model it started from is still in there.
LORA = Integer(ENV.fetch("LORA", 0))
# Which linears the adapter wraps, by their last segment. These are
# ModernBERT's names; `Wqkv` is the whole of attention's input side.
LORA_ON = ENV.fetch("LORA_ON", "Wqkv,Wo").split(",").map(&:strip)
LORA_ALPHA = ENV.fetch("LORA_ALPHA", nil)&.then { |v| Float(v) }
# The adapter this run trains through, or nil for a full fine-tune.
ADAPTER = (Torobi::LoRA.new(rank: LORA, alpha: LORA_ALPHA, on: LORA_ON) unless LORA.zero?)
EVALUATION_BATCH = Integer(ENV.fetch("EVALUATION_BATCH", 16))
EVERY = Integer(ENV.fetch("EVERY", 100))
CAP = Integer(ENV.fetch("CAP_GIB", 12)) * 1024 * 1024 * 1024

def rows_of(path)
  File.readlines(path).reject { |line| line.strip.empty? }.map { |line| JSON.parse(line) }
end

# The encoder, as a graph that answers with one vector per row.
#
# Built for no particular sequence length, so the rows decide: a part of
# queries costs what queries cost rather than what documents do.
def embedder(config, rows: nil)
  Torobi::Models::ModernBERT.embedder(config, seq: nil, pooling: :mean,
                                      encoder_prefix: "", rows:, adapter: ADAPTER)
end

# One part of a step's rows, padded to its own longest row.
#
# **Padding is work**, and the three blocks are of very different
# lengths: the queries of auto-wiki-qa are eighteen tokens where its
# documents are 192, so padding them together makes the queries cost ten
# times what they are (docs/plan.md 15.62). The blocks are laid out in
# order, so a part is usually all queries or all documents, and each one
# is padded to what it holds.
def part_of(config, rows)
  Torobi::Models::ModernBERT.batch(config, rows, seq: rows.map(&:size).max,
                                   pooling: :mean)
end

# What a gradient cache asks of the model it back-propagates: a loss that
# is the product of the representations with a seed the caller supplies
# (docs/plan.md 15.37).
def seeded(model)
  Torobi::GraphConfig.new(
    models: { m: model },
    objective: Torobi.objective(m: model) do |g|
      g.output :loss, g.sum(g.from_model(:m, :embedding) * g.from_batch(:seed, [nil, nil]))
    end
  )
end

# Multiple negatives ranking, over representations computed elsewhere.
#
# The rows arrive as queries, then the document each one matches, then
# whatever negatives were mined for them: a query is scored against every
# document in the batch, and the only supervision is which pairs were put
# in together. At NEGATIVES=0 that is the in-batch loss and the two are
# one expression, which is why there is one.
def contrastive(width)
  held = PAIRS * (2 + NEGATIVES)
  graph = Torobi.graph do |g|
    v = g.input :vectors, [held, width]
    q = v.slice(axis: 0, start: 0, length: PAIRS)
    documents = v.slice(axis: 0, start: PAIRS, length: PAIRS * (1 + NEGATIVES))
    positives = documents.slice(axis: 0, start: 0, length: PAIRS)
    matched = g.sum(q * positives, axes: [-1]) * SCALE
    all = g.matmul(q, documents.transpose(axes: [1, 0])) * SCALE
    g.output :loss, g.mean(g.sum(all.exp, axes: [-1]).log - matched)
  end
  Torobi::GraphConfig.new(models: { m: graph }, train: [])
end

# One step's rows, in the three blocks the loss slices out, cut into
# parts small enough to hold.
#
# A row that was mined for fewer negatives than the run asks for would
# quietly shift every block after it, so it is refused rather than
# padded: what to do about it is the dataset's to answer.
def parts_of(config, pairs)
  rows = pairs.map { |row| row.fetch("query_ids") } +
         pairs.map { |row| row.fetch("text_ids") }
  NEGATIVES.times do |i|
    rows += pairs.map do |row|
      mined = row["negative_ids"] || []
      unless mined.size >= NEGATIVES
        raise "a pair has #{mined.size} negatives and this run wants #{NEGATIVES} " \
              "(tools/mine_negatives.rb writes them)"
      end

      mined[i]
    end
  end
  rows.each_slice(PART).map { |slice| part_of(config, slice) }
end

# Every row of `ids`, as vectors, through the model as it stands.
#
# `forward` rather than a tap on an evaluation: what is wanted is an
# output, and the batch it needs is what the model reads and no more, so
# the seed the training objective wants is not in it.
def embed(session, config, ids, width)
  ids.each_slice(EVALUATION_BATCH).flat_map do |slice|
    session.forward(part_of(config, slice)).fetch("m.embedding")
           .to_a.each_slice(width).to_a
  end
end

# nDCG@10 over the bundle: what the model would rank first, against what
# the dataset says is relevant.
#
# The vectors are already of length one, so a dot product is a cosine and
# ranking by it is ranking by similarity.
def ndcg(session, config, bundle, width, at: 10)
  documents = embed(session, config, bundle.fetch("corpus").map { |d| d.fetch("ids") }, width)
  ids = bundle.fetch("corpus").map { |d| d.fetch("id") }
  queries = embed(session, config, bundle.fetch("queries").map { |q| q.fetch("ids") }, width)

  scores = bundle.fetch("queries").each_with_index.map do |query, i|
    relevant = bundle.fetch("qrels").fetch(query.fetch("id"), {})
    ranked = documents.each_with_index
                      .max_by(at) { |vector, _| dot(queries[i], vector) }
                      .map { |_, j| ids[j] }
    gained = ranked.each_with_index.sum do |id, rank|
      relevant.fetch(id, 0).to_f / Math.log2(rank + 2)
    end
    ideal = relevant.values.sort.reverse.first(at).each_with_index
                    .sum { |score, rank| score.to_f / Math.log2(rank + 2) }
    ideal.zero? ? 0.0 : gained / ideal
  end
  scores.sum / scores.size
end

# The evaluation bundles, by the name each will be reported under.
#
# **More than one, because one cannot tell being better from being
# somewhere else.** Run 003 fell on NLPJournal and run 004 rose on a
# held-out slice of its own training data; either alone would have been
# read as a verdict on the recipe, and together they said the model had
# moved rather than broken (docs/plan.md 15.67). Every run should say
# both from now on.
#
#   runs/held.json,runs/nlpjournal.json
#
# The first is the one `best` follows, because a run has to keep
# something and only one of them can decide what.
def bundles_of(paths)
  paths = paths.split(",").map(&:strip).reject(&:empty?)
  names = paths.map { |path| File.basename(path, ".*") }
  # Two bundles under one name is one bundle, silently, and the whole
  # point of measuring two is that neither is dropped.
  repeated = names.tally.select { |_, n| n > 1 }.keys
  unless repeated.empty?
    abort "two evaluation bundles are both called #{repeated.join(", ")} " \
          "(#{paths.join(", ")}); a run reports them by name, so rename one"
  end

  names.zip(paths).to_h { |name, path| [name, JSON.parse(File.read(path))] }
end

def dot(a, b)
  total = 0.0
  a.each_index { |i| total += a[i] * b[i] }
  total
end

def main
  encoder, train_path, eval_path, dir, steps = ARGV
  unless encoder && train_path && eval_path && dir
    abort "usage: embed_retrieval.rb <encoder-dir> <train.jsonl> " \
          "<eval.json[,eval.json...]> <run-dir> [steps]"
  end

  config = Torobi::Models::ModernBERT.from_config_file(File.join(encoder, "config.json"))
  width = config.hidden_size
  pairs = rows_of(train_path).shuffle(random: Random.new(SEED))
  bundles = bundles_of(eval_path)
  steps = (steps || (pairs.size / PAIRS)).to_i
  FileUtils.mkdir_p(dir)

  Torobi::Memory.limit = CAP
  # Held open for the length of the run, so the block form is not what
  # this wants; it goes when the process does.
  journal = File.open(File.join(dir, "journal.jsonl"), "a") # rubocop:disable Style/FileOpen
                .tap { |io| io.sync = true }
  record = { started_at: Time.now.utc.iso8601, encoder:, train: train_path, eval: eval_path,
             seed: SEED, seq: SEQ, pairs: PAIRS, part: PART, lr: LEARNING_RATE,
             warmup: WARMUP, scale: SCALE, negatives: NEGATIVES, steps:,
             lora: LORA.zero? ? nil : { rank: LORA, alpha: LORA_ALPHA, on: LORA_ON },
             measurements: [] }.compact

  training = seeded(embedder(config))
  Torobi::Session.open(
    training,
    pretrained: { m: File.join(encoder, "model.safetensors") },
    # The adapter's matrices are in no published checkpoint, and saying
    # which parameters are meant to be new is what tells a fresh one from
    # a typo (`Torobi::LoRA#fresh`).
    fresh: ADAPTER ? ADAPTER.fresh(training) : [],
    optimizer: { kind: :adamw, lr: LEARNING_RATE }, seed: SEED, io: journal,
    dataset: { name: File.basename(train_path), pairs: pairs.size,
               tokenizer: File.basename(encoder), max_seq_length: SEQ }
  ) do |session|
    Torobi::Session.open(contrastive(width), weights: { params: {} }) do |loss|
      cache = Torobi::GradCache.new(session, loss:, tap: "m.embedding",
                                    into: :vectors, seed: :seed)
      # A gradient cache steps through `apply!`, which fires the same
      # hook a plain step does, so the schedule sees every step.
      session.use(Torobi::Policies::Schedule.new(peak: LEARNING_RATE, total: steps,
                                                 warmup: WARMUP), every: 10)

      best = { step: nil, ndcg: -1.0 }
      at = 0
      measure = lambda do |step|
        scores = bundles.transform_values { |b| ndcg(session, config, b, width) }
        record[:measurements] << { step:, ndcg: scores.transform_values { |v| v.round(4) },
                                   lr: session.lr, memory: Torobi::Memory.peak }
        session.observe(ndcg: scores)
        said = scores.map { |name, value| format("%s %.4f", name, value) }.join("  ")
        puts format("step %5d  %s  lr %.2e  peak %.1f GiB",
                    step, said, session.lr, Torobi::Memory.peak / (1024.0**3))
        # Kept rather than only reported: the last state of a run is not
        # the best one it passed through, and a run that ends worse than
        # it started should still leave what it reached. The first bundle
        # decides, because only one of them can.
        deciding = scores.values.first
        if deciding > best[:ndcg]
          best = { step:, ndcg: deciding }
          session.checkpoint!(File.join(dir, "best"), at: { at: }) if step.positive?
        end
        scores
      end

      # Where a run that was interrupted got to.
      #
      # A checkpoint holds the parameters, the optimizer and the RNG, and
      # `at:` holds the one thing the engine cannot know: how far into
      # the file this had read (docs/plan.md section 11.2). Together they
      # are the whole of the run, so a machine that slept or a connection
      # that dropped costs the steps since the last one and nothing else.
      checkpoint = File.join(dir, "checkpoint")
      if File.directory?(checkpoint)
        position = session.restore(checkpoint)
        at = position ? position.fetch("at", 0).to_i : 0
        record[:resumed] = { step: session.step, at: }
        puts format("resumed at step %d, row %d of %d", session.step, at, pairs.size)
      end

      record[:before] = measure.call(session.step)
      (session.step...steps).each do |i|
        batch = pairs[(at % pairs.size), PAIRS]
        at += PAIRS
        # The last rows of the file are fewer than a batch: start again
        # rather than train on a short one, which would change what the
        # loss is a mean of.
        next if batch.nil? || batch.size < PAIRS

        value = cache.step(parts_of(config, batch))
        puts format("step %5d  loss %.4f", i + 1, value) if ((i + 1) % 10).zero?
        next unless ((i + 1) % EVERY).zero?

        measure.call(i + 1)
        session.checkpoint!(checkpoint, at: { at: })
      end
      record[:after] = measure.call(steps)
      record[:best] = best
    end

    if ADAPTER.nil?
      session.export_model!(File.join(dir, "export"), from: encoder)
    else
      # An adapted run holds the base weights it started with and the
      # adapter beside them, and `export_model!` writes every parameter
      # it has. That directory would carry `lora_A` and `lora_B` tensors
      # nothing else knows, over base weights that never moved: a model
      # that looks trained and is not. Merging (W + scale * BA, and the
      # adapter dropped) is what makes one, and it is not built yet.
      puts "no export: this run trained an adapter, and merging it into " \
           "the weights is not implemented. The checkpoints hold both halves."
    end
  end

  record[:finished_at] = Time.now.utc.iso8601
  File.write(File.join(dir, "record.json"), "#{JSON.pretty_generate(record)}\n")
  bundles.each_key do |name|
    puts format("%-20s %.4f -> %.4f", name, record[:before].fetch(name),
                record[:after].fetch(name))
  end
  puts format("best %.4f at step %s (by %s), written to %s",
              record.dig(:best, :ndcg), record.dig(:best, :step),
              bundles.keys.first, dir)
end

main if $PROGRAM_NAME == __FILE__
