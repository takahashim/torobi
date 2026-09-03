#!/usr/bin/env ruby
# frozen_string_literal: true

# M4: a distillation that runs to the end and leaves a record.
#
# A cross-encoder reranker scores a (query, document) pair with one number.
# The teacher is cl-nagoya/ruri-v3-reranker-310m; the student is
# cl-nagoya/ruri-v3-130m's encoder with a scoring head it does not have,
# and the data is mteb/ESCIReranking's Japanese split. Kohagi tokenized and
# scored both halves ahead of time (its tools/pairs and tools/dataset), so
# nothing here touches text: a row is token ids, the teacher's score, and
# the dataset's own relevance label.
#
#   ruby experiments/distill_reranker.rb <encoder-dir> <train.jsonl> \
#     <validation.jsonl> <run-dir> [epochs]
#
# What it is not is a framework. docs/plan.md section 7.3 says experiments
# stay plain Ruby until dogfooding says what the shape should be, so this
# is a script: the graph, the loop, and the metric, written out.
#
# The metric is nDCG@10 over each query's candidates, which is what the
# dataset is labelled for, and the teacher's own nDCG is printed beside the
# student's because that is the number the student is being pulled towards.
# Distillation teaches the teacher's answers, including where they are
# wrong, so beating the teacher is not the goal and matching it closely is.

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "torobi"
require "json"

# The run's luck, in both places it enters: the head the encoder does not
# have is drawn from it at open, and the training rows are visited in an
# order shuffled from it. Two runs that differ only here differ only in
# their luck, which is the question a seed is varied to ask.
SEED = 20260903
SEQ = 128
BATCH = 8
EVALUATION_BATCH = 16
CAP = 8 * 1024 * 1024 * 1024
LEARNING_RATE = 2e-5
# Often enough to see the curve, rarely enough that the forwards it costs
# stay a small share of the run: one measurement is about a fifth of what
# the steps between two of them cost.
EVERY = 150

# The dataset this was made from, recorded so a checkpoint can say what it
# learned from (docs/plan.md section 11.2).
DATASET = {
  name: "mteb/ESCIReranking jp",
  revision: "dc2cfaf4fcbf238806a02ae8607786e88112463e",
  teacher: "cl-nagoya/ruri-v3-reranker-310m",
  tokenizer: "cl-nagoya/ruri-v3-reranker-310m",
  max_seq_length: SEQ
}.freeze

def rows_of(path)
  File.readlines(path).reject { |line| line.strip.empty? }.map { |line| JSON.parse(line) }
end

# One batch of rows as the graph's inputs, with the teacher's scores.
def batch_of(config, rows)
  Torobi::Models::ModernBERT
    .batch(config, rows.map { |row| row.fetch("input_ids") }, seq: SEQ)
    .merge(teacher: Torobi::TensorData.from_a([rows.size, 1],
                                              rows.map { |row| row.fetch("teacher") }))
end

# The validation set in one pass: the mean loss, and what the student said
# about every row.
#
# One pass rather than two, which is not tidiness. The validation set is
# 2864 rows and a forward over it costs the same as fifty training steps;
# asking for the loss and the scores separately would double that for
# nothing, since a tap reports what the same forward already computed.
#
# The tap goes on for the evaluation and comes off again: a standing tap
# costs a forward per training step, and this needs it a few times a run.
def measure(session, config, rows)
  session.tap("student.classifier", stat: :full)
  seen = rows.each_slice(EVALUATION_BATCH).map do |slice|
    loss = session.evaluate(batch_of(config, slice))
    [loss, slice.size, session.tapped.fetch("student.classifier").to_a]
  end
  session.untap("student.classifier")
  { loss: seen.sum { |loss, n, _| loss * n } / rows.size.to_f,
    ndcg: mean_ndcg(rows, seen.flat_map(&:last)) }
end

# nDCG@10 for one query: the ranking a score induces, against the labels.
def ndcg(scored, at: 10)
  gain = ->(order) { order.take(at).each_with_index.sum { |r, i| r / Math.log2(i + 2.0) } }
  ideal = gain.call(scored.map(&:last).sort.reverse)
  return 0.0 if ideal.zero?

  gain.call(scored.sort_by { |score, _| -score }.map(&:last)) / ideal
end

# The mean of that over queries. Rows carry the query they belong to, so
# the grouping is the dataset's rather than this script's.
def mean_ndcg(rows, scores)
  by_query = rows.zip(scores).group_by { |row, _| row.fetch("query_id") }
  per_query = by_query.map do |_, pairs|
    ndcg(pairs.map { |row, score| [score, row.fetch("relevance").to_f] })
  end
  per_query.sum / per_query.size
end

def child(run, encoder_dir, train, validation, epochs)
  run.cap!
  config = Torobi::Models::ModernBERT.from_config_file(File.join(encoder_dir, "config.json"))
  student = Torobi::Models::ModernBERT.classifier(config, seq: SEQ, encoder_prefix: "")
  objective = Torobi.objective(student:) do |g|
    # The teacher reports a sigmoid, so the student is pulled towards it
    # where the teacher spoke rather than where its logits happen to sit.
    g.output :loss, g.mse(g.from_model(:student, :logits).sigmoid,
                          g.from_batch(:teacher, [nil, 1]))
  end
  graph = Torobi::GraphConfig.new(models: { student: }, objective:)

  # The teacher's own ranking, which is the ceiling this run is aimed at,
  # and the dataset's order, which is where it starts from. Neither needs
  # the model, so both are known before anything opens.
  teacher = mean_ndcg(validation, validation.map { |row| row.fetch("teacher") })
  measurements = []

  Torobi::Session.open(graph,
                       pretrained: { student: File.join(encoder_dir, "model.safetensors") },
                       fresh: ["student.head.*", "student.classifier.*"],
                       optimizer: { kind: :adamw, lr: LEARNING_RATE }, seed: SEED,
                       io: run.journal, dataset: DATASET.merge(rows: train.size)) do |s|
    puts format("%d parameters, %d train rows, %d validation rows over %d queries",
                s.parameter_paths.size, train.size, validation.size,
                validation.map { |row| row.fetch("query_id") }.uniq.size)
    puts format("teacher nDCG@10: %.4f", teacher)

    record = lambda do
      measurement = { step: s.step, **measure(s, config, validation) }
      measurements << measurement
      puts format("step %4d: validation loss %.6f, nDCG@10 %.4f",
                  measurement[:step], measurement[:loss], measurement[:ndcg])
      # In the journal as well as on the screen: the record is the file.
      s.observe(event: "validation", **measurement)
    end

    record.call
    order = train.each_index.to_a.shuffle(random: Random.new(SEED))
    epochs.times do |epoch|
      order.each_slice(BATCH) do |slice|
        s.step!(batch_of(config, train.values_at(*slice)))
        break if run.stopping?

        record.call if (s.step % EVERY).zero?
      end
      break if run.stopping?

      puts format("epoch %d of %d done at step %d", epoch + 1, epochs, s.step)
    end
    record.call unless measurements.last[:step] == s.step
    s.checkpoint!(run.checkpoint)

    best = measurements.max_by { |m| m[:ndcg] }
    File.write(File.join(run.dir, "metrics.json"),
               JSON.pretty_generate(dataset: DATASET, epochs:, batch: BATCH,
                                    learning_rate: LEARNING_RATE,
                                    seed: s.seed,
                                    teacher_ndcg: teacher, measurements:))
    puts format("best: nDCG@10 %.4f at step %d (teacher %.4f, started %.4f)",
                best[:ndcg], best[:step], teacher, measurements.first[:ndcg])
  end
end

if ENV[Torobi::Runner::DIRECTORY_VARIABLE]
  encoder_dir, train, validation, _dir, epochs = ARGV
  Torobi::Runner.child! do |run|
    child(run, encoder_dir, rows_of(train), rows_of(validation), Integer(epochs || 1))
  end
end

encoder_dir, train, validation, dir, epochs = ARGV
unless encoder_dir && train && validation && dir
  abort "usage: distill_reranker.rb <encoder-dir> <train.jsonl> <validation.jsonl> " \
        "<run-dir> [epochs]"
end

runner = Torobi::Runner.new([RbConfig.ruby, __FILE__, encoder_dir, train, validation,
                             dir, (epochs || 1).to_s],
                            dir:, memory_limit: CAP).start
outcome = runner.wait
puts "journal:  #{runner.journal_path}"
puts "metrics:  #{File.join(runner.dir, "metrics.json")}"
abort "the run did not finish: #{outcome.inspect}" unless outcome.finished?
