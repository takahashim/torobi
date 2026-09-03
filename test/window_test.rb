# frozen_string_literal: true

require_relative "test_helper"
require "stringio"
require "tmpdir"

# The window: what a caller may do between spans, and the record it leaves
# (docs/plan.md sections 8.3 and 8.5). Knobs, freezing, writing a
# parameter, and a journal that holds enough to replay from.
class WindowTest < Minitest::Test
  DIM = 4

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  # Two layers, so freezing one is a thing that can be observed.
  def config
    model = Torobi.graph do |g|
      x = g.input :x, [nil, DIM]
      h = g.linear(x, DIM, name: "first")
      g.output :loss, g.mean(g.linear(h, 1, name: "second"))
    end
    Torobi::GraphConfig.new(models: { "m" => model })
  end

  def weights
    { params: {
      "m.first.weight" => { shape: [DIM, DIM], data: Array.new(DIM * DIM, 0.1) },
      "m.first.bias" => { shape: [DIM], data: Array.new(DIM, 0.0) },
      "m.second.weight" => { shape: [1, DIM], data: Array.new(DIM, 0.1) },
      "m.second.bias" => { shape: [1], data: [0.0] }
    } }
  end

  def batch = { x: { shape: [8, DIM], data: Array.new(8 * DIM, 1.0) } }

  def test_freezing_stops_training_what_it_names
    Torobi::Session.open(config, weights, optimizer: { kind: :sgd, lr: 0.1 }) do |s|
      assert_equal 4, s.trainable.size

      moved = s.freeze!("m.first.*")
      assert_equal %w[m.first.weight m.first.bias], moved
      assert_equal %w[m.second.weight m.second.bias], s.trainable

      before = s.fetch("m.first.weight")[:data]
      s.run([batch] * 5)
      assert_equal before, s.fetch("m.first.weight")[:data], "a frozen layer must not move"
      refute_equal Array.new(DIM, 0.1), s.fetch("m.second.weight")[:data]
    end
  end

  def test_unfreezing_lets_it_move_again
    Torobi::Session.open(config, weights, optimizer: { kind: :sgd, lr: 0.1 }) do |s|
      s.freeze!("m.first.*")
      s.run([batch] * 3)
      frozen = s.fetch("m.first.weight")[:data]

      assert_equal %w[m.first.weight m.first.bias], s.unfreeze!("m.first.*")
      s.run([batch] * 3)
      refute_equal frozen, s.fetch("m.first.weight")[:data]
    end
  end

  # AdamW's slots have to follow the set: kept for what stays, started at
  # zero for what thaws. If they did not, the next step would index a slot
  # that is not there.
  def test_freezing_moves_the_optimizer_slots_with_it
    Torobi::Session.open(config, weights, optimizer: { kind: :adamw, lr: 0.05 }) do |s|
      s.run([batch] * 3)
      s.freeze!("m.first.*")
      s.run([batch] * 3)
      s.unfreeze!("m.first.*")
      s.run([batch] * 3)
      assert_equal 9, s.step
      assert_predicate s.loss, :finite?
    end
  end

  def test_freezing_everything_is_refused
    Torobi::Session.open(config, weights) do |s|
      e = assert_raises(Torobi::StepError) { s.freeze!("m.*") }
      assert_match(/would leave nothing to train/, e.message)
      assert_equal 4, s.trainable.size, "the refusal changed nothing"
    end
  end

  def test_a_pattern_that_matches_nothing_is_refused
    Torobi::Session.open(config, weights) do |s|
      e = assert_raises(Torobi::StepError) { s.freeze!("m.third.*") }
      assert_match(/no parameter matches/, e.message)
    end
  end

  def test_a_parameter_can_be_written_from_the_window
    Torobi::Session.open(config, weights) do |s|
      s.put("m.second.weight", { shape: [1, DIM], data: [1.0, 2.0, 3.0, 4.0] })
      assert_equal [1.0, 2.0, 3.0, 4.0], s.fetch("m.second.weight")[:data]

      e = assert_raises(Torobi::StepError) do
        s.put("m.second.weight", { shape: [DIM], data: Array.new(DIM, 0.0) })
      end
      assert_match(/is not \[1, 4\]/, e.message)
      assert_raises(Torobi::StepError) { s.put("nope", { shape: [1], data: [0.0] }) }
    end
  end

  # The journal is what a replay reads: it must hold the decisions and
  # their inputs, not just the fact that steps happened.
  def test_the_journal_records_the_window
    io = StringIO.new
    Torobi::Session.open(config, weights, io:, dataset: { "digest" => "abc" }) do |s|
      s.adjust(lr: 0.25)
      s.run([batch] * 2)
      s.observe(loss: s.loss)
      s.freeze!("m.first.*")
      s.put("m.second.bias", { shape: [1], data: [0.5] })
      s.run([batch])
    end

    entries = Torobi::Journal.read(io.string)
    header = entries.first
    assert_equal Torobi::Journal::SCHEMA_VERSION, header.fetch("schema_version")
    assert_equal config.digest, header.dig("provenance", "config", "digest")
    assert_equal({ "digest" => "abc" }, header.dig("provenance", "dataset"))

    kinds = entries.drop(1).map { |e| e["kind"] }
    assert_equal %w[note adjust span span note observe adjust put span note note], kinds

    adjusts = entries.select { |e| e["kind"] == "adjust" }
    assert_equal 0.25, adjusts.first.fetch("lr")
    assert_equal "m.first.*", adjusts.last.fetch("freeze")
    assert_equal %w[m.first.weight m.first.bias], adjusts.last.fetch("moved")

    # A put names what was written without holding it.
    put = entries.find { |e| e["kind"] == "put" }
    assert_equal "m.second.bias", put.fetch("path")
    assert_match(/\A[0-9a-f]{64}\z/, put.fetch("digest"))

    # And what the window read is there, because a policy reading it would
    # have decided on it.
    observed = entries.find { |e| e["kind"] == "observe" }
    assert_in_delta entries.select { |e| e["kind"] == "span" }[1].fetch("loss"),
                    observed.fetch("loss"), 1e-9
  end

  def test_a_session_without_a_journal_records_nothing_and_still_runs
    Torobi::Session.open(config, weights) do |s|
      assert_nil s.journal
      s.adjust(lr: 0.1)
      s.run([batch])
      assert_equal 1, s.step
    end
  end

  # A journal written to a file survives the run that wrote it.
  def test_a_journal_can_be_written_to_a_file_as_it_goes
    Dir.mktmpdir("torobi-journal") do |dir|
      path = File.join(dir, "run.jsonl")
      File.open(path, "w") do |file|
        Torobi::Session.open(config, weights, io: file) do |s|
          s.run([batch] * 2)
          # Readable before the session closes, because each entry is
          # flushed as it is written.
          assert_operator Torobi::Journal.read(File.read(path)).size, :>=, 3
        end
      end
      entries = Torobi::Journal.read(File.read(path))
      assert_equal "closed", entries.last.fetch("event")
    end
  end
end
