#!/usr/bin/env ruby
# frozen_string_literal: true

# Fine-tune sbintuitions/modernbert-ja-130m as a Jev Local cross-encoder
# over the questions `data.py` renders (docs/plan.md section 15.74).
#
# The reference is jev_local's `modernbert/train.py`, and this follows it:
# a one-logit cross-encoder, the listwise cross-entropy over each
# question's candidates, AdamW (0.9, 0.98), a linear warmup and decay,
# gradient clipping at 1.0, and a validation report per task. What it does
# not follow is the batching order, which Torobi's shapes cannot make
# identical (see the adapter), and the numbers, which differ because MLX
# and torch are different kernels.
#
#   ruby experiments/modernbert_ja_choice.rb <base-dir> <train.jsonl> \
#     <validation.jsonl> <run-dir> [epochs]
#
# The data comes from `python -m tools.dump_jev_dataset` in jev_local;
# rendering and tokenization happen here, with the same reference code the
# tokenizer oracle holds to `prompting.py`.

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "torobi"
require "fileutils"
require "json"
require_relative "../tools/jev_graph"
require_relative "../tools/jev_candidates"
require_relative "../tools/jev_eval"

begin
  require "tokenizers"
rescue LoadError
  abort "this needs the tokenizers gem: gem install tokenizers"
end

# What the base is, recorded in the checkpoint's sidecar the way the
# reference records it, so the loader can name it rather than infer it.
BASE_MODEL = "sbintuitions/modernbert-ja-130m"
BASE_REVISION = "28c180b16463ba6f3fa79b48756fbf21586fe23e"

# The reference's hyperparameters (jev_local train.py, and the published
# modernbert-ja-310m-jev's `jev_modernbert.json`).
SEED = 0
PAIR_BUDGET = 128
EPOCHS = 2
LEARNING_RATE = 3e-5
WARMUP = 0.06
BETA1 = 0.9
BETA2 = 0.98
EPS = 1e-6
WEIGHT_DECAY = 0.01
CLIP = 1.0
MAX_LENGTH = 512
# Often enough to see the curve, rarely enough that the forwards stay a
# small share of the run.
EVAL_EVERY = 1000
CAP = 8 * 1024 * 1024 * 1024

# Recorded so a checkpoint can say what it learned from and how
# (docs/plan.md section 11.2); the tokenizer is the base's.
DATASET = {
  name: "JGLUE + JCoLA + commonsense-moral-ja + MASSIVE, as System One questions",
  seed: SEED,
  max_seq_length: MAX_LENGTH,
  pair_budget: PAIR_BUDGET
}.freeze

def rows_of(path)
  File.readlines(path).reject { |line| line.strip.empty? }.map { |line| JSON.parse(line) }
end

def total_steps(train)
  ((train.sum { |row| row.fetch("choices").size }.to_f / PAIR_BUDGET).ceil * EPOCHS)
end

# The checkpoint the loader will accept: weights, the base's tokenizer and
# config, and the sidecar that says which rendering it was trained with.
#
# `export_model!` writes sentence-transformers files too, which a
# cross-encoder is not; they are removed, and the config is corrected to
# say what the model now is, so the directory is what transformers and the
# Jev loader expect and nothing else.
def export!(session, dir, base_dir, config, meta)
  FileUtils.mkdir_p(dir)
  session.export_model!(dir, from: base_dir, pooling: :cls, pooling_dim: config.hidden_size)
  ["modules.json", "config_sentence_transformers.json", "1_Pooling"].each do |name|
    FileUtils.rm_rf(File.join(dir, name))
  end
  config_json = JSON.parse(File.read(File.join(dir, "config.json")))
  config_json["architectures"] = ["ModernBertForSequenceClassification"]
  config_json["num_labels"] = 1
  config_json["id2label"] = { "0" => "LABEL_0" }
  config_json["label2id"] = { "LABEL_0" => 0 }
  File.write(File.join(dir, "config.json"), "#{JSON.pretty_generate(config_json)}\n")
  File.write(File.join(dir, "jev_modernbert.json"), "#{JSON.pretty_generate(meta)}\n")
end

def child(run, base_dir, train, valid, epochs)
  run.cap!
  config = Torobi::Models::ModernBERT.from_config_file(File.join(base_dir, "config.json"))
  graph = Jev.graph(config)
  tokenizer = Tokenizers::Tokenizer.from_file(File.join(base_dir, "tokenizer.json"))
  candidates = Jev::Candidates.new(config, tokenizer:, pair_budget: PAIR_BUDGET,
                                            max_length: MAX_LENGTH)
  steps = total_steps(train) * epochs / EPOCHS
  meta = { "format_version" => Jev::FORMAT_VERSION, "base_model" => BASE_MODEL,
           "revision" => BASE_REVISION, "model_id" => "modernbert-ja-130m-jev",
           "max_length" => MAX_LENGTH,
           "trained" => true, "tasks" => "all", "epochs" => epochs, "lr" => LEARNING_RATE,
           "pair_budget" => PAIR_BUDGET, "seed" => SEED, "train_questions" => train.size,
           "total_steps" => steps }
  measurements = []
  best = nil

  Torobi::Session.open(graph,
                       pretrained: { Jev::MODEL => File.join(base_dir, "model.safetensors") },
                       # The base is a masked-LM: its encoder is under `model.`
                       # and its head is the one a classifier uses, so only the
                       # classifier is new.
                       fresh: ["student.classifier.*"],
                       optimizer: { kind: :adamw, lr: LEARNING_RATE, beta1: BETA1, beta2: BETA2,
                                    eps: EPS, weight_decay: WEIGHT_DECAY, clip: CLIP },
                       seed: SEED, io: run.journal, dataset: DATASET.merge(rows: train.size)) do |s|
    puts format("%d parameters, %d train questions, %d validation questions",
                s.parameter_paths.size, train.size, valid.size)
    s.use(Torobi::Policies::Linear.new(peak: LEARNING_RATE, total: steps, warmup: WARMUP))

    record = lambda do
      report = Jev::Eval.run(s, candidates, valid)
      measurement = { step: s.step, valid: report, elapsed_s: Time.now.to_i }
      measurements << measurement
      puts format("step %4d: macro accuracy %.4f, nll %.4f",
                  measurement[:step], report["_all"][:macro_accuracy], report["_all"][:nll])
      s.observe(event: "validation", **measurement)
      if best.nil? || report["_all"][:macro_accuracy] > best[:valid]["_all"][:macro_accuracy]
        best = measurement
        export!(s, File.join(run.dir, "model"), base_dir, config,
                meta.merge("step" => s.step, "valid" => report, "best" => true))
        puts format("saved best (macro accuracy %.4f) at step %d",
                    report["_all"][:macro_accuracy], s.step)
      end
    end

    record.call
    rng = Random.new(SEED)
    epochs.times do |epoch|
      order = (0...train.size).to_a
      order.shuffle!(random: rng)
      candidates.batches(train.values_at(*order)).each do |batch|
        s.step!(batch)
        break if run.stopping?

        record.call if (s.step % EVAL_EVERY).zero?
      end
      break if run.stopping?

      puts format("epoch %d of %d done at step %d", epoch + 1, epochs, s.step)
    end
    record.call unless measurements.last[:step] == s.step
    s.checkpoint!(run.checkpoint)

    File.write(File.join(run.dir, "metrics.json"),
               JSON.pretty_generate(dataset: DATASET, epochs:, pair_budget: PAIR_BUDGET,
                                    learning_rate: LEARNING_RATE, seed: s.seed,
                                    measurements:))
    puts format("best: macro accuracy %.4f at step %d",
                best[:valid]["_all"][:macro_accuracy], best[:step])
  end
end

if ENV[Torobi::Runner::DIRECTORY_VARIABLE]
  base_dir, train, valid, _dir, epochs = ARGV
  Torobi::Runner.child! do |run|
    child(run, base_dir, rows_of(train), rows_of(valid), Integer(epochs || EPOCHS))
  end
end

base_dir, train, valid, dir, epochs = ARGV
unless base_dir && train && valid && dir
  abort "usage: modernbert_ja_choice.rb <base-dir> <train.jsonl> <validation.jsonl> " \
        "<run-dir> [epochs]"
end

# The reference refuses a non-empty output directory, and so does this. A
# run always opens from the base weights -- it does not resume -- and the
# journal is opened for appending, so a run into a directory that already
# holds one would leave the two mixed and the record unreadable.
if File.directory?(dir) && !Dir.empty?(dir)
  abort "#{dir} is not empty; choose a new run directory (this run does not resume)"
end

runner = Torobi::Runner.new([RbConfig.ruby, __FILE__, base_dir, train, valid,
                             dir, (epochs || EPOCHS).to_s],
                            dir:, memory_limit: CAP).start
outcome = runner.wait
puts "journal:  #{runner.journal_path}"
puts "metrics:  #{File.join(runner.dir, "metrics.json")}"
puts "model:    #{File.join(runner.dir, "model")}"
abort "the run did not finish: #{outcome.inspect}" unless outcome.finished?
