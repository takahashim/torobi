# frozen_string_literal: true

require_relative "test_helper"
require "stringio"

# The two replay modes (docs/plan.md 8.6). They answer different questions,
# and the tests below are written to make that difference visible: action
# replay asks whether the training reproduces, deterministic rerun asks
# whether the policy still decides what it decided.
class ReplayTest < Minitest::Test
  DIM = 2

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  def config
    model = Torobi.graph do |g|
      x = g.input :x, [nil, DIM]
      y = g.input :y, [nil, 1]
      g.output :loss, g.mse(g.linear(x, 1, name: "l"), y)
    end
    Torobi::GraphConfig.new(models: { "m" => model })
  end

  def weights
    { params: { "m.l.weight" => { shape: [1, DIM], data: [0.0, 0.0] },
                "m.l.bias" => { shape: [1], data: [0.0] } } }
  end

  # Different data each step, so a replay that fed the wrong batches would
  # be found out.
  def batches(count = 8, seed: 3)
    rng = Random.new(seed)
    Array.new(count) do
      xs = Array.new(4) { [rng.rand(-1.0..1.0), rng.rand(-1.0..1.0)] }
      { x: { shape: [4, DIM], data: xs.flatten },
        y: { shape: [4, 1], data: xs.map { |a, b| [a + b] }.flatten } }
    end
  end

  def optimizer = { kind: :adamw, lr: 0.05 }

  # A run that turns a knob partway through, so the journal has something
  # to replay beyond the steps themselves.
  def record(data)
    io = StringIO.new
    Torobi::Session.open(config, weights: weights, io:, optimizer:) do |s|
      s.run(data.first(3))
      s.adjust(lr: 0.01)
      s.run(data.drop(3))
    end
    io.string
  end

  # Same machine, same build, same data: the loss agrees bit for bit.
  def test_action_replay_reproduces_the_run_bitwise
    data = batches
    journal = record(data)

    result = Torobi::Replay.action(journal, config:, weights:, batches: data)

    assert_predicate result, :agrees?, result.to_s
    assert_equal :action, result.mode
    assert_equal data.size, result.steps
  end

  # The knob is in the journal, so a replay that ignored it would end
  # somewhere else: this is what makes the replay of operations, not just
  # of steps, worth having.
  def test_the_replay_applies_the_knobs_the_journal_records
    data = batches
    journal = record(data)
    replayed = Torobi::Replay.action(journal, config:, weights:, batches: data)

    # The same steps without the adjustment reach a different loss.
    without = Torobi::Session.open(config, weights: weights, optimizer:) do |s|
      s.run(data)
      s.loss
    end

    refute_in_delta without, replayed.final_loss, 1e-9,
                    "ignoring the knob should not land in the same place"
  end

  # Different data is the thing a replay cannot paper over.
  def test_a_replay_over_other_data_diverges_and_says_where
    data = batches
    journal = record(data)
    other = batches(8, seed: 99)

    result = Torobi::Replay.action(journal, config:, weights:, batches: other)

    refute_predicate result, :agrees?
    assert_equal 1, result.divergences.first.step, "it should diverge at the first step"
    assert_match(/differs/, result.divergences.first.why)
    assert_match(/divergence/, result.to_s)
  end

  # The digest names the data, not only the shape of the batch, so a
  # replay can tell one batch from another and the record is not a lie.
  def test_a_batch_digest_depends_on_the_data
    one = Torobi::Batch.of(x: { shape: [1, DIM], data: [1.0, 2.0] })
    same = Torobi::Batch.of(x: { shape: [1, DIM], data: [1.0, 2.0] })
    other = Torobi::Batch.of(x: { shape: [1, DIM], data: [1.0, 2.5] })

    assert_equal one.digest, same.digest
    refute_equal one.digest, other.digest
  end

  # Other data is caught by the digest the journal recorded, not only by
  # whatever loss it happens to produce.
  def test_action_replay_checks_the_batch_digest
    data = batches
    journal = record(data)
    other = batches(8, seed: 99)

    result = Torobi::Replay.action(journal, config:, weights:, batches: other)

    divergence = result.divergences.find { |d| d.why =~ /batch differs/ }

    refute_nil divergence, result.to_s
    assert_equal 1, divergence.step
  end

  def test_action_replay_checks_the_accumulated_part_digest
    data = batches(6)
    program = lambda do |session, all|
      all.each_slice(3) do |parts|
        parts.each { |part| session.accumulate(part) }
        session.apply!
      end
    end
    io = StringIO.new
    Torobi::Session.open(config, weights: weights, io:, optimizer:) { |s| program.call(s, data) }

    result = Torobi::Replay.action(io.string, config:, weights:, batches: batches(6, seed: 99))

    data_divergences = result.divergences.select { |d| d.why =~ /batch differs/ }

    refute_empty(data_divergences, result.to_s)
  end

  def test_too_few_batches_is_refused_by_count
    data = batches
    journal = record(data)
    e = assert_raises(ArgumentError) do
      Torobi::Replay.action(journal, config:, weights:, batches: data.first(3))
    end
    assert_match(/only 3 batches/, e.message)
  end

  # A rerun runs the policy again and holds it to what it observed. Same
  # policy, same data: it observes the same things.
  def test_a_deterministic_rerun_holds_the_policy_to_what_it_observed
    data = batches
    io = StringIO.new
    policy = lambda do |session, all|
      all.each do |batch|
        loss = session.step!(batch)
        session.observe(loss:, lr: session.lr) if (session.step % 2).zero?
      end
    end

    Torobi::Session.open(config, weights: weights, io:, optimizer:) { |s| policy.call(s, data) }
    result = Torobi::Replay.rerun(io.string, config:, weights:, batches: data, &policy)

    assert_predicate result, :agrees?, result.to_s
    assert_equal :rerun, result.mode
    assert_equal 4, result.steps, "it observed on every second step of eight"
  end

  # And a policy that changed its mind is caught, which is the point.
  def test_a_rerun_catches_a_policy_that_decides_differently
    data = batches
    io = StringIO.new
    original = lambda do |session, all|
      all.each do |batch|
        loss = session.step!(batch)
        session.observe(loss:) if (session.step % 2).zero?
      end
    end
    changed = lambda do |session, all|
      all.each do |batch|
        session.step!(batch)
        # Observes something else: a regression in the driving Ruby.
        session.observe(loss: 0.0) if (session.step % 2).zero?
      end
    end

    Torobi::Session.open(config, weights: weights, io:, optimizer:) { |s| original.call(s, data) }
    result = Torobi::Replay.rerun(io.string, config:, weights:, batches: data, &changed)

    refute_predicate result, :agrees?
    assert_match(/differs on loss/, result.divergences.first.why)
  end

  def test_a_rerun_notices_a_policy_that_stopped_observing
    data = batches(4)
    io = StringIO.new
    Torobi::Session.open(config, weights: weights, io:, optimizer:) do |s|
      data.each { |b| s.observe(loss: s.step!(b)) }
    end

    silent = ->(session, all) { all.each { |b| session.step!(b) } }
    result = Torobi::Replay.rerun(io.string, config:, weights:, batches: data, &silent)

    refute_predicate result, :agrees?
    assert_equal data.size, result.divergences.size
    assert_match(/observed nothing/, result.divergences.first.why)
  end

  def test_a_journal_is_accepted_as_an_object_its_jsonl_or_its_entries
    data = batches(3)
    io = StringIO.new
    Torobi::Session.open(config, weights: weights, io:, optimizer:) { |s| s.run(data) }

    text = io.string
    [text, Torobi::Journal.read(text)].each do |form|
      result = Torobi::Replay.action(form, config:, weights:, batches: data)

      assert_predicate result, :agrees?, "#{form.class}: #{result}"
    end
    assert_raises(ArgumentError) { Torobi::Replay.action(42, config:, weights:, batches: data) }
  end

  # The optimizer a run used is in its journal, so a replay does not have
  # to be told again.
  def test_the_replay_takes_the_optimizer_from_the_journal
    data = batches(4)
    io = StringIO.new
    Torobi::Session.open(config, weights: weights, io:, optimizer: { kind: :sgd, lr: 0.4 }) do |s|
      s.run(data)
    end
    result = Torobi::Replay.action(io.string, config:, weights:, batches: data)

    assert_predicate result, :agrees?, result.to_s
  end

  # A cap lifted partway (`clip: nil`) is a decision the replay has to
  # make again, not a knob it may skip.
  def test_action_replay_turns_the_clip_as_the_run_did
    data = batches(6)
    io = StringIO.new
    # SGD, whose step is as long as the gradient: AdamW's would hardly
    # notice a cap, and the test would pass with the clip ignored.
    Torobi::Session.open(config, weights: weights, io:, optimizer: { kind: :sgd, lr: 0.5 }) do |s|
      s.adjust(clip: 1e-3)
      s.run(data.first(3))
      s.adjust(clip: nil)
      s.run(data.drop(3))
    end
    result = Torobi::Replay.action(io.string, config:, weights:, batches: data)

    assert_predicate result, :agrees?, result.to_s
  end

  # Accumulated parts are fed and applied as they were, and they are not
  # observations: the same program over the same data agrees with itself.
  def test_accumulated_steps_replay_and_rerun
    data = batches(6)
    program = lambda do |session, all|
      all.each_slice(3) do |parts|
        parts.each { |part| session.accumulate(part) }
        session.observe(loss: session.apply!)
      end
    end
    io = StringIO.new
    Torobi::Session.open(config, weights: weights, io:, optimizer:) { |s| program.call(s, data) }

    action = Torobi::Replay.action(io.string, config:, weights:, batches: data)
    rerun = Torobi::Replay.rerun(io.string, config:, weights:, batches: data, &program)

    assert_predicate action, :agrees?, action.to_s
    assert_equal 2, action.steps
    assert_predicate rerun, :agrees?, rerun.to_s
  end

  # The seed a run was opened with is in its provenance, so a replay opens
  # with it rather than with the default.
  def test_the_replay_takes_the_seed_from_the_journal
    data = batches(2)
    io = StringIO.new
    Torobi::Session.open(config, weights: weights, io:, optimizer:, seed: 42) { |s| s.run(data) }
    replayed = nil
    Torobi::Replay::Action.new(io.string, config:, weights:, tolerance: :bitwise)
                          .send(:open) { |s| replayed = s.seed }

    assert_equal 42, replayed
  end
end
