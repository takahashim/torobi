# frozen_string_literal: true

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
  # than holding them, so the caller supplies them. Action replay holds
  # each one to the digest that was recorded (`Batch#digest`); a rerun
  # holds the run to the observations instead, and the losses those
  # observations came from move if the data did. Both open the session
  # the way the run was opened, with the optimizer and seed its
  # provenance recorded.
  module Replay
    # One place the replay and the record disagree.
    Divergence = Data.define(:step, :expected, :actual, :why)

    # What a replay found. `agrees?` answers the question the mode asked.
    Result = Data.define(:mode, :steps, :final_loss, :divergences) do
      def agrees? = divergences.empty?

      def to_s
        return "#{mode}: agrees after #{steps} steps" if agrees?

        "#{mode}: #{divergences.size} divergence(s), first at step #{divergences.first.step}"
      end
    end

    # `Replay::Action.new(...).call(batches)`, said shorter.
    def self.action(journal, config:, weights:, batches:, tolerance: :bitwise)
      Action.new(journal, config:, weights:, tolerance:).call(batches)
    end

    # `Replay::Rerun.new(...).call(batches, &program)`, said shorter.
    def self.rerun(journal, config:, weights:, batches:, tolerance: 1e-6, &program)
      Rerun.new(journal, config:, weights:, tolerance:).call(batches, &program)
    end

    # What both modes share: the record, how to open a session the way the
    # run was opened, and what counts as agreeing.
    class Mode
      # `journal` is a Journal, its JSONL, or what `Journal.read` returned.
      #
      # `tolerance` is the agreement asked for: nil for none (only the
      # operations are replayed), a number for the dtype-scale agreement of
      # docs/plan.md section 8.6, and :bitwise for exact equality, which the
      # same machine and build should give.
      def initialize(journal, config:, weights:, tolerance:)
        @record = record_of(journal)
        @config = config
        @weights = weights
        @tolerance = tolerance
        @divergences = []
      end

      private

      def open(&)
        provenance = @record.header.provenance
        optimizer = provenance.optimizer&.transform_keys(&:to_sym) || Session::DEFAULT_OPTIMIZER
        Session.open(@config, weights: @weights, optimizer:,
                              seed: provenance.seed || Session::DEFAULT_SEED, &)
      end

      def agrees?(a, b)
        return true if @tolerance.nil?
        return a == b unless a.is_a?(Numeric) && b.is_a?(Numeric)
        return a == b if @tolerance == :bitwise

        (a - b).abs <= @tolerance
      end

      def diverged(step:, expected:, actual:, why:)
        @divergences << Divergence.new(step:, expected:, actual:, why:)
      end

      def record_of(journal)
        case journal
        when Journal then journal.to_record
        when String then Journal.read(journal)
        when Journal::Record then journal
        else
          raise ArgumentError, "a journal is a Journal, its JSONL, or what Journal.read returned"
        end
      end
    end

    # Applies a journal's operations to a fresh session, in order, without
    # running any policy. `batches` supplies the data the journal names,
    # one per step taken or part accumulated.
    class Action < Mode
      def call(batches)
        @data = batches.to_a
        @fed = 0
        steps = @record.of(Journal::Span).size
        final = open do |session|
          @record.entries.each { |entry| apply(session, entry) }
          session.loss
        end
        Result.new(mode: :action, steps:, final_loss: final, divergences: @divergences.dup)
      end

      private

      def apply(session, entry)
        case entry
        when Journal::Adjust
          session.adjust(**entry.knobs.transform_keys(&:to_sym))
        when Journal::Freezing
          entry.frozen ? session.freeze!(entry.pattern) : session.unfreeze!(entry.pattern)
        when Journal::Accumulate
          session.accumulate(checked(entry, next_batch))
        when Journal::Span
          if entry.accumulated?
            compare(entry, session.apply!)
          else
            compare(entry, session.step!(checked(entry, next_batch)))
          end
        end
        # A put's value is not in the journal, only its digest; observations
        # and notes change nothing.
      end

      # The batch to feed, after holding it to the digest the journal
      # recorded. It is fed either way: a mismatch is reported and the
      # replay goes on, so the losses can be compared too.
      def checked(entry, batch)
        batch = Batch.of(batch)
        recorded = entry.batches_digest
        return batch if recorded.nil?

        actual = batch.digest
        return batch if actual == recorded

        diverged(step: entry.step, expected: recorded, actual:,
                 why: "the batch differs from the one that was recorded")
        batch
      end

      def compare(entry, loss)
        return if agrees?(entry.loss, loss)

        diverged(step: entry.step, expected: entry.loss, actual: loss,
                 why: "the loss differs by #{(entry.loss - loss).abs}")
      end

      # The batch the next step or part was fed.
      def next_batch
        batch = @data.fetch(@fed) do
          raise ArgumentError,
                "the journal was fed #{fed_in_all} batches and only #{@data.size} " \
                "batches were given"
        end
        @fed += 1
        batch
      end

      # One per step taken on a batch, and one per part accumulated.
      def fed_in_all
        @record.of(Journal::Span).count { |span| !span.accumulated? } +
          @record.of(Journal::Accumulate).size
      end
    end

    # Runs `program` again over the same data and holds it to what the
    # journal observed: every `observe` the first run recorded must be
    # observed again, with the same values.
    #
    # This is the mode for a policy: the program is the Ruby that drove the
    # run, and what is compared is its inputs and its decisions, not only
    # where the training ended up.
    class Rerun < Mode
      def call(batches, &program)
        raise ArgumentError, "rerun needs the program that drove the run" unless program

        seen = []
        final = open do |session|
          program.call(Watcher.new(session, seen), batches)
          session.loss
        end
        expected = @record.of(Journal::Observe)
        expected.zip(seen).each_with_index { |(want, got), i| compare(want, got, i) }
        if seen.size > expected.size
          extra = seen[expected.size]
          diverged(step: extra.step, expected: nil, actual: extra.values,
                   why: "the rerun observed more than the journal did")
        end
        Result.new(mode: :rerun, steps: seen.size, final_loss: final, divergences: @divergences.dup)
      end

      private

      def compare(want, got, index)
        if got.nil?
          diverged(step: want.step, expected: want.values, actual: nil,
                   why: "the rerun observed nothing here")
        elsif want.values.keys.sort != got.values.keys.sort
          diverged(step: want.step, expected: want.values, actual: got.values,
                   why: "the rerun observed different things")
        else
          want.values.each_pair do |key, value|
            other = got.values.fetch(key)
            next if agrees?(value, other)

            diverged(step: want.step, expected: { key => value }, actual: { key => other },
                     why: "observation #{index} differs on #{key}")
          end
        end
      end
    end

    # Wraps a session so that what the program observes is collected for
    # comparison. Everything else passes through untouched.
    class Watcher < BasicObject
      def initialize(session, into)
        @session = session
        @into = into
      end

      def observe(**values)
        @into << ::Torobi::Journal::Observe.new(step: @session.step, values:)
        @session.observe(**values)
      end

      def method_missing(name, ...)
        @session.public_send(name, ...)
      end

      def respond_to_missing?(name, include_private = false)
        @session.respond_to?(name, include_private)
      end
    end
  end
end
