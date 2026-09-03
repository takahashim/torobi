# frozen_string_literal: true

require "json"

module Torobi
  # Running a journal again.
  #
  # "Replay" was one word for two things (docs/plan.md section 8.6), and
  # they answer different questions:
  #
  #   action replay        applies what was recorded, without the policy
  #                        that decided it. Asks: does this training
  #                        reproduce? Used for CI and for making a run
  #                        again from its record.
  #
  #   deterministic rerun  runs the policy again, against the same data,
  #                        and holds it to what it observed the first
  #                        time. Asks: does this policy still decide what
  #                        it decided? Used to catch a regression in the
  #                        Ruby that drove the run.
  #
  # Neither replays the data: a journal names batches by digest rather
  # than holding them, so the caller supplies them and the digests say
  # whether they are the same ones.
  module Replay
    # What a replay found. `agrees?` answers the question the mode asked.
    Result = Data.define(:mode, :steps, :final_loss, :divergences) do
      def agrees? = divergences.empty?

      def to_s
        return "#{mode}: agrees after #{steps} steps" if agrees?

        "#{mode}: #{divergences.size} divergence(s), first at " \
          "step #{divergences.first.fetch(:step)}"
      end
    end

    module_function

    # Applies a journal's operations to a fresh session, in order, without
    # running any policy. `batches` supplies the data the journal names.
    #
    # `tolerance` is the agreement asked for on the loss at each step:
    # nil for none (only the operations are replayed), a number for the
    # dtype-scale agreement of docs/plan.md section 8.6, and :bitwise for
    # exact equality, which the same machine and build should give.
    def action(journal, config:, weights:, batches:, tolerance: :bitwise)
      entries = entries_of(journal)
      recorded = entries.select { |e| e["kind"] == "span" && e.key?("loss") }
      data = batches.to_a
      divergences = []

      final = Session.open(config, weights: weights, optimizer: optimizer_of(entries)) do |session|
        seen = 0
        entries.each do |entry|
          case entry["kind"]
          when "adjust" then apply_adjust(session, entry)
          when "put" then nil # a put's value is not in the journal, only its digest
          when "span"
            batch = data[seen] or
              raise ArgumentError,
                    "the journal has #{recorded.size} steps and only #{data.size} " \
                    "batches were given"
            loss = session.step!(batch)
            check(divergences, entry, loss, tolerance, seen + 1)
            seen += 1
          end
        end
        session.loss
      end

      Result.new(mode: :action, steps: recorded.size, final_loss: final, divergences:)
    end

    # Runs `program` again over the same data and holds it to what the
    # journal observed: every `observe` the first run recorded must be
    # observed again, with the same values.
    #
    # This is the mode for a policy: the program is the Ruby that drove the
    # run, and what is compared is its inputs and its decisions, not only
    # where the training ended up.
    def rerun(journal, config:, weights:, batches:, tolerance: 1e-6, &program)
      raise ArgumentError, "rerun needs the program that drove the run" unless program

      entries = entries_of(journal)
      expected = entries.select { |e| e["kind"] == "observe" }
      seen = []
      divergences = []

      final = Session.open(config, weights: weights, optimizer: optimizer_of(entries)) do |session|
        recorder = Watcher.new(session, seen)
        program.call(recorder, batches)
        session.loss
      end

      expected.zip(seen).each_with_index do |(want, got), i|
        if got.nil?
          divergences << { step: want["step"], expected: values_of(want), actual: nil,
                           why: "the rerun observed nothing here" }
          next
        end
        compare_observation(divergences, want, got, tolerance, i)
      end
      if seen.size > expected.size
        extra = seen[expected.size]
        divergences << { step: extra.fetch(:step), expected: nil, actual: extra,
                         why: "the rerun observed more than the journal did" }
      end

      Result.new(mode: :rerun, steps: seen.size, final_loss: final, divergences:)
    end

    # Wraps a session so that what the program observes is collected for
    # comparison. Everything else passes through untouched.
    class Watcher < BasicObject
      def initialize(session, into)
        @session = session
        @into = into
      end

      def observe(**values)
        @into << { step: @session.step, values: values.transform_keys(&:to_s) }
        @session.observe(**values)
      end

      def method_missing(name, ...)
        @session.public_send(name, ...)
      end

      def respond_to_missing?(name, include_private = false)
        @session.respond_to?(name, include_private)
      end
    end

    def entries_of(journal)
      case journal
      when Journal then journal.to_a
      when String then Journal.read(journal)
      when Array then journal
      else
        raise ArgumentError, "a journal is a Journal, its JSONL, or its entries"
      end
    end

    def optimizer_of(entries)
      opened = entries.find { |e| e["event"] == "opened" }
      opened&.dig("optimizer")&.transform_keys(&:to_sym) || Session::DEFAULT_OPTIMIZER
    end

    def apply_adjust(session, entry)
      knobs = {}
      knobs[:lr] = entry["lr"] if entry.key?("lr")
      knobs[:seed] = entry["seed"] if entry.key?("seed")
      session.adjust(**knobs) unless knobs.empty?
      session.freeze!(entry["freeze"]) if entry["freeze"]
      session.unfreeze!(entry["unfreeze"]) if entry["unfreeze"]
    end

    def check(divergences, entry, loss, tolerance, step)
      return if tolerance.nil?

      want = entry.fetch("loss")
      agrees = tolerance == :bitwise ? want == loss : (want - loss).abs <= tolerance
      return if agrees

      divergences << { step:, expected: want, actual: loss,
                       why: "the loss differs by #{(want - loss).abs}" }
    end

    def compare_observation(divergences, want, got, tolerance, index)
      wanted = values_of(want)
      actual = got.fetch(:values)
      if wanted.keys.sort != actual.keys.sort
        divergences << { step: want["step"], expected: wanted, actual:,
                         why: "the rerun observed different things" }
        return
      end

      wanted.each do |key, value|
        other = actual.fetch(key)
        next if agrees?(value, other, tolerance)

        divergences << { step: want["step"], expected: { key => value },
                         actual: { key => other },
                         why: "observation #{index} differs on #{key}" }
      end
    end

    def agrees?(a, b, tolerance)
      return a == b unless a.is_a?(Numeric) && b.is_a?(Numeric)
      return a == b if tolerance == :bitwise

      (a - b).abs <= tolerance
    end

    def values_of(entry)
      entry.except("kind", "step", "at")
    end
  end
end
