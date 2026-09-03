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
    journal = Torobi::Journal.new(Torobi::Provenance.of(c, dataset: { "digest" => "abc" }))
    provenance = journal.header.fetch("provenance")

    # Which description was trained, exactly.
    assert_equal c.digest, provenance.dig("config", "digest")
    assert_equal ["m"], provenance.dig("config", "models")
    assert_equal %w[m.l.weight m.l.bias], provenance.dig("config", "parameters")
    # On what.
    assert_equal({ "digest" => "abc" }, provenance.fetch("dataset"))
    # By which build.
    assert_equal Torobi::VERSION, provenance.dig("runtime", "torobi")
    assert_equal RUBY_PLATFORM, provenance.dig("runtime", "platform")
  end

  def test_the_engine_reports_what_it_was_built_from
    skip "extension not compiled" unless defined?(Torobi::Session)
    info = Torobi::Provenance.runtime.fetch("engine")

    assert_equal Torobi::VERSION, info.fetch("torobi_engine")
    assert_includes %w[debug release], info.fetch("profile")
    refute_empty info.fetch("mlx_rs")
  end

  def test_entries_carry_their_kind_and_step
    journal = Torobi::Journal.new(Torobi::Provenance.of(config))
    journal.span(steps: 100, loss: 0.5, step: 100)
    journal.observe(step: 100, loss: 0.5, grad_norm: 1.25)
    journal.adjust(step: 100, lr: 0.05)
    journal.checkpoint(path: "checkpoint/000001", step: 100)

    kinds = journal.entries.map { |e| e["kind"] }

    assert_equal %w[span observe adjust checkpoint], kinds
    assert(journal.entries.all? { |e| e["step"] == 100 })
    assert(journal.entries.all? { |e| e.key?("at") })
    assert_in_delta(0.05, journal.entries[2].fetch("lr"))
  end

  # An observation is recorded because a policy that reads and then decides
  # has the reading as its input: a deterministic rerun is held to it.
  def test_what_the_window_read_is_recorded
    journal = Torobi::Journal.new(Torobi::Provenance.of(config))
    journal.observe(step: 10, loss: 0.9)
    journal.adjust(step: 10, lr: 0.01)

    read = journal.entries.find { |e| e["kind"] == "observe" }
    decided = journal.entries.find { |e| e["kind"] == "adjust" }

    assert_in_delta(0.9, read.fetch("loss"))
    assert_equal read["step"], decided["step"], "the decision and its input are at the same step"
  end

  def test_an_unknown_kind_is_refused
    journal = Torobi::Journal.new(Torobi::Provenance.of(config))
    assert_raises(ArgumentError) { journal.send(:record, :improvise, whatever: 1) }
  end

  # Written as it goes, so an interrupted run leaves a readable file.
  def test_it_streams_and_round_trips
    io = StringIO.new
    journal = Torobi::Journal.new(Torobi::Provenance.of(config), io:)
    journal.span(steps: 5, loss: 1.0, step: 5)
    journal.close

    lines = Torobi::Journal.read(io.string)

    assert_equal journal.to_a, lines
    assert_equal 1, lines.first.fetch("schema_version")
    assert_equal "span", lines.last.fetch("kind")
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
    journal.note(detail: { "nested" => [1, 2] })
    entry = journal.entries.last
    assert_raises(FrozenError) { entry["kind"] = "other" }
    assert_raises(FrozenError) { entry.fetch("detail")["nested"] << 3 }
  end

  # A run that was killed leaves whole lines and, possibly, a last one that
  # never finished: entries are flushed one at a time, but a write is not
  # atomic. Reading that back has to work, because "an interrupted run
  # leaves a readable file" is the promise the flushing is for.
  def test_a_truncated_last_line_is_dropped_and_the_rest_reads
    whole = <<~JSONL
      {"schema_version":1,"provenance":{}}
      {"kind":"span","step":1,"loss":0.5}
    JSONL
    text = "#{whole}{\"kind\":\"span\",\"step\":2,\"lo"

    entries = Torobi::Journal.read(text)

    assert_equal 2, entries.size
    assert_equal 1, entries.last["step"]
  end

  # A broken line in the middle is damage, not truncation, and reading past
  # it would be inventing a record.
  def test_a_broken_line_in_the_middle_is_an_error
    text = <<~JSONL
      {"schema_version":1,"provenance":{}}
      {"kind":"span","ste
      {"kind":"span","step":2,"loss":0.4}
    JSONL

    assert_raises(JSON::ParserError) { Torobi::Journal.read(text) }
  end

  def test_a_complete_file_is_unaffected
    journal = Torobi::Journal.new({ "config" => {} })
    journal.span(steps: 1, loss: 0.5, step: 1)

    assert_equal journal.to_a, Torobi::Journal.read(journal.to_jsonl)
  end
end
