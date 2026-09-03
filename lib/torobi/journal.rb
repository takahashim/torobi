# frozen_string_literal: true

require "json"
require "digest"
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
  class Journal
    SCHEMA_VERSION = 1

    # The kinds of entry. Named, small, and closed: a journal a later
    # version cannot read is worse than one that records less.
    KINDS = %i[
      span
      adjust
      observe
      put
      checkpoint
      note
    ].freeze

    attr_reader :header, :entries

    # `provenance` is everything about the run that is not an event: the
    # config's digest, the runtime, the dataset. Built by `Provenance`.
    def initialize(provenance, io: nil)
      @header = Freeze.deep({
        "schema_version" => SCHEMA_VERSION,
        "started_at" => Time.now.utc.iso8601,
        "provenance" => provenance
      })
      @entries = []
      @io = io
      write(@header)
    end

    # A span ran: how many steps, on what, and what came out.
    def span(steps:, loss:, batches_digest: nil, **rest)
      record(:span, steps:, loss:, batches_digest:, **rest)
    end

    # A knob turned. The value is what the window set, not what it was.
    def adjust(**knobs) = record(:adjust, **knobs)

    # What the window read, and therefore what a policy could have decided
    # on. Recorded so a rerun can be held to it.
    def observe(**values) = record(:observe, **values)

    # A parameter written from the window.
    def put(path:, digest:) = record(:put, path:, digest:)

    def checkpoint(path:, step:) = record(:checkpoint, path:, step:)

    # Anything a caller wants in the record; not interpreted.
    def note(**fields) = record(:note, **fields)

    def close
      @io&.flush
      self
    end

    # The journal as it would be re-read: header first, then entries.
    def to_a = [@header, *@entries]

    def to_jsonl = to_a.map { |entry| JSON.generate(entry) }.join("\n") << "\n"

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
      lines.reject { |line| line.strip.empty? }.map { |line| JSON.parse(line) }
    end

    private

    def record(kind, **fields)
      unless KINDS.include?(kind)
        raise ArgumentError, "#{kind.inspect} is not a journal entry kind (#{KINDS.join(", ")})"
      end

      entry = {
        "kind" => kind.to_s,
        "step" => fields.delete(:step),
        "at" => Time.now.utc.iso8601
      }.merge(fields.transform_keys(&:to_s)).compact
      @entries << Freeze.deep(entry)
      write(entry)
      entry
    end

    # Flushed per entry, because the promise is that an interrupted run
    # leaves a readable file, and a buffer that never reached the disk is
    # not one.
    def write(entry)
      return unless @io

      @io.puts(JSON.generate(entry))
      @io.flush
    end
  end

  # Everything about a run that is not an event: what was trained, with
  # what, on what, and by which build. Gathered once, at the top of a
  # journal (docs/plan.md section 8.5).
  module Provenance
    module_function

    def of(config, dataset: nil, extra: {})
      {
        "config" => {
          "digest" => config.digest,
          "schema_version" => config.schema_version,
          "semantics_version" => config.semantics_version,
          "models" => config.models.keys,
          "train" => config.train,
          "parameters" => config.parameters.map(&:qualified_path)
        },
        "dataset" => dataset,
        "runtime" => runtime
      }.merge(extra.transform_keys(&:to_s)).compact
    end

    # The build this ran on. The engine's half is asked for only when the
    # extension is there, so a pure-Ruby run still produces a journal.
    def runtime
      info = {
        "torobi" => Torobi::VERSION,
        "ruby" => RUBY_VERSION,
        "platform" => RUBY_PLATFORM
      }
      info["engine"] = Torobi::Native.build_info if Torobi.const_defined?(:Native) &&
                                                    Torobi::Native.respond_to?(:build_info)
      info
    end

    # A digest of data, so that a journal names what it was fed without
    # holding it.
    #
    # Framed: each part contributes its length before its bytes, so that
    # ("ab", "c") and ("a", "bc") do not agree, which plain concatenation
    # let them do. Hashes are serialized with their keys sorted, so a
    # digest depends on the data and not on the order it was written in.
    def digest_of(*parts)
      digest = Digest::SHA256.new
      digest << "torobi/1\n"
      parts.each do |part|
        bytes = part.is_a?(String) ? part : JSON.generate(IR::Json.canonical(part))
        digest << "#{bytes.bytesize}:"
        digest << bytes
      end
      digest.hexdigest
    end
  end
end
