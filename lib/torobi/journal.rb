# frozen_string_literal: true

require "json"
# For Time#iso8601, which is not on Time until somebody asks: a run
# started by `Torobi::Runner` is a process where nothing else has.
require "time"

module Torobi
  # What a run recorded, so that it can be replayed.
  #
  # A journal is JSONL: one header, then one entry per thing that happened.
  # It is written as it goes, so an interrupted run leaves a readable file.
  #
  # The schema is the one docs/plan.md section 8.5 lists, and it exists for
  # the two modes of section 8.6:
  #
  #   action replay        applies the recorded operations without running
  #                        the policy that produced them
  #   deterministic rerun  runs the policy again and holds it to the
  #                        observations recorded here
  #
  # The second is why observations are recorded at all. v3.1 said reading
  # needed no record; that was wrong, because a policy that reads and then
  # decides has the reading as its input.
  #
  # Every entry is a value of its own kind (`Journal::Span`, `Adjust`, ...),
  # so what a kind holds is written once, here, and the session that writes
  # one and the replay that reads it back agree by construction rather than
  # by string keys.
  class Journal
    SCHEMA_VERSION = 2

    def self.now = Time.now.utc.iso8601

    # What every entry has: its kind, the step it happened at, and when.
    # The rest is the kind's own, and is written beside those three.
    module Entry
      def self.included(kind)
        kind.extend(ClassMethods)
      end

      module ClassMethods
        # The name the kind is written under.
        attr_accessor :kind
      end

      # The entry a serialized line holds.
      def self.from_h(h)
        kind = KINDS.fetch(h.fetch("kind")) do
          raise ArgumentError, "#{h["kind"].inspect} is not a journal entry kind " \
                               "(#{KINDS.keys.join(", ")})"
        end
        kind.read(h.except("kind", "step", "at"), step: h["step"], at: h["at"])
      end

      def initialize(at: nil, **fields)
        super(at: at || Journal.now, **fields)
      end

      def kind = self.class.kind

      def to_h
        { "kind" => kind, "step" => step, "at" => at }.compact.merge(payload)
      end

      def to_json(*) = JSON.generate(to_h)
    end

    # A step was taken: on one batch, or on the `parts` accumulated before
    # it. `batches_digest` names the batch without holding it.
    Span = Data.define(:step, :at, :steps, :loss, :parts, :batches_digest) do
      include Entry

      def self.read(h, step:, at:)
        new(step:, at:, steps: h["steps"], loss: h["loss"], parts: h["parts"],
            batches_digest: h["batches_digest"])
      end

      def initialize(step:, steps: 1, loss: nil, parts: nil, batches_digest: nil, at: nil)
        super
      end

      def accumulated? = !parts.nil?

      def payload
        { "steps" => steps, "loss" => loss, "parts" => parts,
          "batches_digest" => batches_digest }.compact
      end
    end

    # One batch's gradients were added to what is waiting; `parts` is how
    # many are waiting now.
    Accumulate = Data.define(:step, :at, :parts, :loss) do
      include Entry

      def self.read(h, step:, at:) = new(step:, at:, parts: h.fetch("parts"), loss: h["loss"])

      def payload = { "parts" => parts, "loss" => loss }
    end

    # Knobs turned, by name, to the value the window set. A knob set to nil
    # (`clip: nil`, no cap) is held as nil rather than dropped, because
    # "lifted" and "not touched" are different decisions.
    Adjust = Data.define(:step, :at, :knobs) do
      include Entry

      def self.read(h, step:, at:) = new(step:, at:, knobs: h)

      def initialize(step:, knobs:, at: nil)
        super(step:, at:, knobs: Freeze.deep(knobs.transform_keys(&:to_s)))
      end

      def payload = knobs
    end

    # What `pattern` matched was frozen (or thawed), and which paths moved.
    Freezing = Data.define(:step, :at, :pattern, :frozen, :moved) do
      include Entry

      def self.read(h, step:, at:)
        new(step:, at:, pattern: h.fetch("pattern"), frozen: h.fetch("frozen"),
            moved: h.fetch("moved"))
      end

      def payload = { "pattern" => pattern, "frozen" => frozen, "moved" => moved }
    end

    # What the window read, and therefore what a policy could have decided
    # on. Recorded so a rerun can be held to it.
    Observe = Data.define(:step, :at, :values) do
      include Entry

      def self.read(h, step:, at:) = new(step:, at:, values: h)

      def initialize(step:, values:, at: nil)
        super(step:, at:, values: Freeze.deep(values.transform_keys(&:to_s)))
      end

      def payload = values
    end

    # A parameter written from the window, named by digest.
    Put = Data.define(:step, :at, :path, :digest) do
      include Entry

      def self.read(h, step:, at:)
        new(step:, at:, path: h.fetch("path"), digest: h.fetch("digest"))
      end

      def payload = { "path" => path, "digest" => digest }
    end

    # A checkpoint was written.
    Checkpoint = Data.define(:step, :at, :path) do
      include Entry

      def self.read(h, step:, at:) = new(step:, at:, path: h.fetch("path"))

      def payload = { "path" => path }
    end

    # Something happened that a replay does not apply: the run opened,
    # finished a span, exported, restored, closed, ended. `event` says
    # which; the rest is what the one who noted it wanted kept.
    Note = Data.define(:step, :at, :event, :details) do
      include Entry

      def self.read(h, step:, at:)
        new(step:, at:, event: h.fetch("event"), details: h.except("event"))
      end

      def initialize(event:, step: nil, details: {}, at: nil)
        details = details.transform_keys(&:to_s).transform_values do |v|
          v.is_a?(Symbol) ? v.to_s : v
        end
        super(step:, at:, event: event.to_s, details: Freeze.deep(details))
      end

      def [](key) = details[key.to_s]

      def payload = { "event" => event }.merge(details)
    end

    KINDS = {
      "span" => Span, "accumulate" => Accumulate, "adjust" => Adjust, "freeze" => Freezing,
      "observe" => Observe, "put" => Put, "checkpoint" => Checkpoint, "note" => Note
    }.each { |name, kind| kind.kind = name }.freeze

    # The first line: which schema, when, and what the run was.
    Header = Data.define(:schema_version, :started_at, :provenance) do
      def self.from_h(h)
        new(schema_version: h.fetch("schema_version"), started_at: h["started_at"],
            provenance: Provenance.from_h(h.fetch("provenance")))
      end

      def to_h
        { "schema_version" => schema_version, "started_at" => started_at,
          "provenance" => provenance.to_h }
      end

      def to_json(*) = JSON.generate(to_h)
    end

    # A journal as it was read back: its header and its entries.
    Record = Data.define(:header, :entries) do
      # The entries of one kind, in order.
      def of(kind) = entries.grep(kind)
    end

    attr_reader :header, :entries

    # `provenance` is everything about the run that is not an event
    # (`Torobi::Provenance`).
    def initialize(provenance, io: nil)
      @header = Header.new(schema_version: SCHEMA_VERSION, started_at: Journal.now, provenance:)
      @entries = []
      @io = io
      write(@header)
    end

    def span(step:, loss:, parts: nil, batches_digest: nil)
      add(Span.new(step:, loss:, parts:, batches_digest:))
    end

    def accumulate(step:, parts:, loss:) = add(Accumulate.new(step:, parts:, loss:))

    # The value is what the window set, not what it was.
    def adjust(step:, **knobs) = add(Adjust.new(step:, knobs:))

    def freezing(step:, pattern:, frozen:, moved:)
      add(Freezing.new(step:, pattern:, frozen:, moved:))
    end

    def observe(step:, **values) = add(Observe.new(step:, values:))

    def put(step:, path:, digest:) = add(Put.new(step:, path:, digest:))

    def checkpoint(step:, path:) = add(Checkpoint.new(step:, path:))

    # Anything a caller wants in the record; not interpreted.
    def note(event:, step: nil, **details) = add(Note.new(event:, step:, details:))

    def close
      @io&.flush
      self
    end

    # The journal as it would be re-read.
    def to_record = Record.new(header: @header, entries: @entries.dup.freeze)

    def to_jsonl = [@header, *@entries].map(&:to_json).join("\n") << "\n"

    # Parses a journal back.
    #
    # A run that was killed leaves whole lines and, possibly, a last one
    # that never finished: entries are flushed one at a time, but a write
    # is not atomic. So an unterminated final line is dropped, and only
    # that one. A broken line anywhere else means the file is damaged
    # rather than truncated, and saying so is better than reading past it.
    def self.read(text)
      lines = text.lines
      lines.pop if lines.last && !lines.last.end_with?("\n")
      header, *entries = lines.reject { |line| line.strip.empty? }.map { |line| JSON.parse(line) }
      Record.new(header: Header.from_h(header), entries: entries.map { |h| Entry.from_h(h) })
    end

    private

    def add(entry)
      @entries << entry
      write(entry)
      entry
    end

    # Flushed per entry, because the promise is that an interrupted run
    # leaves a readable file, and a buffer that never reached the disk is
    # not one.
    def write(line)
      return unless @io

      @io.puts(line.to_json)
      @io.flush
    end
  end
end
