# frozen_string_literal: true

require_relative "test_helper"
require "stringio"

# The journal schema of docs/plan.md section 8.5, and what the two replay
# modes of 8.6 need from it.
class JournalTest < Minitest::Test
  def config
    model = Torobi.graph do |g|
      x = g.input :x, [nil, 2]
      g.output :loss, g.mean(g.linear(x, 1, name: "l"))
    end
    Torobi::GraphConfig.new(models: { "m" => model })
  end

  def test_the_header_says_what_the_run_was
    c = config
    provenance = Torobi::Provenance.of(c, dataset: { "digest" => "abc" },
                                          optimizer: { kind: :sgd, lr: 0.1 }, seed: 7)
    journal = Torobi::Journal.new(provenance)

    assert_same provenance, journal.header.provenance
    # Which description was trained, exactly.
    assert_equal c.digest, provenance.config.fetch("digest")
    assert_equal ["m"], provenance.config.fetch("models")
    assert_equal %w[m.l.weight m.l.bias], provenance.config.fetch("parameters")
    # On what.
    assert_equal({ "digest" => "abc" }, provenance.dataset)
    # By which build.
    assert_equal Torobi::VERSION, provenance.runtime.fetch("torobi")
    assert_equal RUBY_PLATFORM, provenance.runtime.fetch("platform")
    # With what, and from where.
    assert_equal({ "kind" => "sgd", "lr" => 0.1 }, provenance.optimizer)
    assert_equal 7, provenance.seed
  end

  def test_the_engine_reports_what_it_was_built_from
    skip "extension not compiled" unless defined?(Torobi::Session)
    info = Torobi::Provenance.runtime.fetch("engine")

    assert_equal Torobi::VERSION, info.fetch("torobi_engine")
    assert_includes %w[debug release], info.fetch("profile")
    refute_empty info.fetch("mlx")
    refute_empty info.fetch("mlx_c")
  end

  def test_entries_carry_their_kind_and_step
    journal = Torobi::Journal.new(Torobi::Provenance.of(config))
    journal.span(loss: 0.5, step: 100)
    journal.observe(step: 100, loss: 0.5, grad_norm: 1.25)
    journal.adjust(step: 100, lr: 0.05)
    journal.checkpoint(path: "checkpoint/000001", step: 100)

    assert_equal %w[span observe adjust checkpoint], journal.entries.map(&:kind)
    assert(journal.entries.all? { |e| e.step == 100 })
    assert(journal.entries.all?(&:at))
    assert_in_delta(0.05, journal.entries[2].knobs.fetch("lr"))
  end

  # An observation is recorded because a policy that reads and then decides
  # has the reading as its input: a deterministic rerun is held to it.
  def test_what_the_window_read_is_recorded
    journal = Torobi::Journal.new(Torobi::Provenance.of(config))
    journal.observe(step: 10, loss: 0.9)
    journal.adjust(step: 10, lr: 0.01)

    read, decided = journal.entries

    assert_in_delta(0.9, read.values.fetch("loss"))
    assert_equal read.step, decided.step, "the decision and its input are at the same step"
  end

  def test_an_unknown_kind_is_refused_when_read
    e = assert_raises(ArgumentError) do
      Torobi::Journal::Entry.from_h({ "kind" => "improvise", "step" => 1 })
    end
    assert_match(/not a journal entry kind/, e.message)
  end

  # Every kind reads back as what was written, so a writer and a reader
  # agree by construction. A knob set to nil is a decision (`clip: nil`
  # lifts the cap) and is held rather than dropped.
  def test_every_kind_round_trips
    journal = Torobi::Journal.new(Torobi::Provenance.of(config))
    journal.span(step: 1, loss: 0.5, batches_digest: "d")
    journal.accumulate(step: 1, parts: 2, loss: 0.25)
    journal.span(step: 2, loss: 0.25, parts: 2)
    journal.adjust(step: 2, lr: 0.1, clip: nil)
    journal.freezing(step: 2, pattern: "m.*", frozen: true, moved: ["m.w"])
    journal.observe(step: 2, loss: 0.25)
    journal.put(step: 2, path: "m.w", digest: "e")
    journal.checkpoint(step: 2, path: "c")
    journal.note(step: 2, event: "exported", model: "m")

    read = Torobi::Journal.read(journal.to_jsonl)

    assert_equal journal.entries, read.entries
    assert_equal({ "lr" => 0.1, "clip" => nil }, read.entries[3].knobs)
    assert_equal Torobi::Journal::KINDS.keys.sort, read.entries.map(&:kind).uniq.sort
  end

  # Written as it goes, so an interrupted run leaves a readable file.
  def test_it_streams_and_round_trips
    io = StringIO.new
    journal = Torobi::Journal.new(Torobi::Provenance.of(config), io:)
    journal.span(loss: 1.0, step: 5)
    journal.close

    record = Torobi::Journal.read(io.string)

    assert_equal journal.to_record, record
    assert_equal Torobi::Journal::SCHEMA_VERSION, record.header.schema_version
    assert_equal "span", record.entries.last.kind
    assert_equal journal.to_jsonl, io.string
  end

  def test_a_data_digest_names_what_was_fed_without_holding_it
    a = Torobi::Provenance.digest_of({ "x" => [1.0, 2.0] })
    b = Torobi::Provenance.digest_of({ "x" => [1.0, 2.0] })
    c = Torobi::Provenance.digest_of({ "x" => [1.0, 2.5] })

    assert_equal a, b
    refute_equal a, c
    assert_match(/\A[0-9a-f]{64}\z/, a)
  end

  # Plain concatenation let ("ab", "c") and ("a", "bc") agree; each part
  # now contributes its length before its bytes.
  def test_the_digest_frames_its_parts
    refute_equal Torobi::Provenance.digest_of("ab", "c"),
                 Torobi::Provenance.digest_of("a", "bc")
    refute_equal Torobi::Provenance.digest_of("a", "b"),
                 Torobi::Provenance.digest_of("ab")
  end

  # And it depends on the data, not on the order a hash was written in.
  def test_the_digest_ignores_key_order
    assert_equal Torobi::Provenance.digest_of({ "a" => 1, "b" => { "x" => 1, "y" => 2 } }),
                 Torobi::Provenance.digest_of({ "b" => { "y" => 2, "x" => 1 }, "a" => 1 })
  end

  def test_entries_cannot_be_changed_after_they_are_written
    journal = Torobi::Journal.new(Torobi::Provenance.of(config))
    journal.note(event: "x", detail: { "nested" => [1, 2] })
    entry = journal.entries.last
    assert_raises(FrozenError) { entry.details["other"] = 1 }
    assert_raises(FrozenError) { entry["detail"]["nested"] << 3 }
  end

  # A run that was killed leaves whole lines and, possibly, a last one that
  # never finished: entries are flushed one at a time, but a write is not
  # atomic. Reading that back has to work, because "an interrupted run
  # leaves a readable file" is the promise the flushing is for.
  def test_a_truncated_last_line_is_dropped_and_the_rest_reads
    whole = <<~JSONL
      {"schema_version":2,"provenance":{"config":{},"runtime":{}}}
      {"kind":"span","step":1,"loss":0.5}
    JSONL
    text = "#{whole}{\"kind\":\"span\",\"step\":2,\"lo"

    entries = Torobi::Journal.read(text).entries

    assert_equal 1, entries.size
    assert_equal 1, entries.last.step
  end

  # A broken line in the middle is damage, not truncation, and reading past
  # it would be inventing a record.
  def test_a_broken_line_in_the_middle_is_an_error
    text = <<~JSONL
      {"schema_version":2,"provenance":{"config":{},"runtime":{}}}
      {"kind":"span","ste
      {"kind":"span","step":2,"loss":0.4}
    JSONL

    assert_raises(JSON::ParserError) { Torobi::Journal.read(text) }
  end

  def test_a_complete_file_is_unaffected
    journal = Torobi::Journal.new(Torobi::Provenance.new(config: {}, runtime: {}))
    journal.span(loss: 0.5, step: 1)

    assert_equal journal.to_record, Torobi::Journal.read(journal.to_jsonl)
  end

  # A file asks for what it uses.
  #
  # Nothing in a run's own process can be relied on to have loaded
  # `time`: a child started by `Torobi::Runner` has this library and the
  # standard library it named, and none of the parent's rake, bundler or
  # minitest. `Time#iso8601` is not on Time until somebody requires it.
  #
  # Checked by reading rather than by running, because a process that has
  # already loaded `time` cannot notice: this cost an hour of CI on Ruby
  # 3.2, where the parent's environment happened not to have loaded it
  # either, and it passed everywhere else.
  def test_every_file_that_formats_a_time_asks_for_time
    lib = File.expand_path("../lib", __dir__)
    forgot = Dir[File.join(lib, "**", "*.rb")].select do |path|
      source = File.read(path)
      source.include?("iso8601") && !source.include?(%(require "time"))
    end

    assert_empty(forgot.map { |path| path.delete_prefix("#{lib}/") })
  end
end
