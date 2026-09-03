# frozen_string_literal: true

require_relative "test_helper"
require "timeout"
require "stringio"

# What an asynchronous interrupt does to a session while the engine is
# running (notes/SESSION_CONCURRENCY_SPEC.md section 0 and 10).
#
# These are not the same path as raising from the block a span yields to.
# That raise happens in Ruby, between steps, and never reaches the boundary
# where the GVL is released. `Timeout`, `Thread#raise` and a signal do:
# CRuby delivers them around the blocking region, and a binding that left
# anything to finish after that region lost it. It used to leave a
# `MutexGuard`, which was never dropped, and the session was busy for good.
class InterruptTest < Minitest::Test
  DIM = 256

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  def config
    model = Torobi.graph do |g|
      x = g.input :x, [nil, DIM]
      g.output :loss, g.mean(g.linear(x, DIM, name: "l"))
    end
    Torobi::GraphConfig.new(models: { "m" => model })
  end

  def weights
    { params: { "m.l.weight" => { shape: [DIM, DIM], data: Array.new(DIM * DIM, 0.001) },
                "m.l.bias" => { shape: [DIM], data: Array.new(DIM, 0.0) } } }
  end

  def batch(rows: 256)
    { x: { shape: [rows, DIM], data: Array.new(rows * DIM, 0.5) } }
  end

  def endless = Enumerator.new { |y| loop { y << batch } }

  def test_a_timeout_during_a_span_leaves_the_session_usable
    session = Torobi::Session.open(config, weights)
    assert_raises(Timeout::Error) { Timeout.timeout(0.3) { session.run(endless) } }

    took = session.step
    assert_operator took, :>, 0, "the span should have taken some steps"

    session.step!(batch)
    assert_equal took + 1, session.step, "the session still trains"
    assert session.close, "and it can still be closed"
  end

  def test_a_thread_raise_during_a_span_surfaces_the_callers_own_error
    session = Torobi::Session.open(config, weights)
    target = Thread.current
    killer = Thread.new { sleep 0.2; target.raise(ArgumentError, "the caller's own error") }

    error = assert_raises(ArgumentError) { session.run(endless) }
    killer.join

    assert_equal "the caller's own error", error.message,
                 "the caller's exception, not one the binding invented"
    took = session.step
    session.step!(batch)
    assert_equal took + 1, session.step
    session.close
  end

  # The window's record and the engine must point at the same place. If a
  # step were taken but the return to Ruby were skipped, the journal would
  # be one step behind what the engine did and a replay would diverge.
  def test_the_journal_and_the_engine_agree_after_an_interruption
    io = StringIO.new
    session = Torobi::Session.open(config, weights, io:)
    assert_raises(Timeout::Error) { Timeout.timeout(0.3) { session.run(endless) } }

    spans = session.journal.entries.select { |e| e["kind"] == "span" }
    refute_empty spans, "the interrupted span should have recorded its steps"

    assert_equal session.step, spans.size, "one entry per step the engine took"
    assert_equal session.step, spans.last["step"]
    session.close
  end

  # The numbers a watcher reads are a state the run passed through, never a
  # step in progress and never one that was rolled back.
  def test_a_watcher_never_sees_a_step_that_did_not_happen
    session = Torobi::Session.open(config, weights)
    readings = Queue.new
    # Sampled rather than spun: the point is to catch the run at many
    # moments, not to see how fast a read is.
    watcher = Thread.new { loop { readings << [session.step, session.loss]; sleep 0.0005 } }
    steps = 30
    steps.times { session.step!(batch) }
    watcher.kill
    watcher.join

    seen = Array.new(readings.size) { readings.pop }
    refute_empty seen
    seen.each do |step, loss|
      assert_operator step, :<=, steps, "no step the run has not reached"
      # NaN belongs to a session that has taken no step at all; after that
      # every published loss is one some step produced.
      assert(step.zero? || !loss.nan?, "step #{step} published a loss of #{loss}")
    end
    # And the readings only ever move forward.
    steps_seen = seen.map(&:first)

    assert_equal steps_seen.sort, steps_seen
    session.close
  end

  # A signal is the case a caller actually meets: Ctrl-C during training.
  # In a subprocess, because an unhandled INT would end this one.
  def test_a_signal_during_a_span_leaves_the_session_usable
    script = <<~SCRIPT
      $LOAD_PATH.unshift(#{File.expand_path("../lib", __dir__).inspect})
      require "torobi"
      model = Torobi.graph do |g|
        x = g.input :x, [nil, #{DIM}]
        g.output :loss, g.mean(g.linear(x, #{DIM}, name: "l"))
      end
      config = Torobi::GraphConfig.new(models: { "m" => model })
      weights = #{weights.inspect}
      batch = #{batch.inspect}
      session = Torobi::Session.open(config, weights)
      Thread.new { sleep 0.3; Process.kill("INT", Process.pid) }
      begin
        session.run(Enumerator.new { |y| loop { y << batch } })
      rescue Interrupt
        puts "INTERRUPTED after \#{session.step} steps"
      end
      session.step!(batch)
      puts "STILL USABLE at \#{session.step}"
      puts "CLOSED \#{session.close}"
    SCRIPT
    output = IO.popen([RbConfig.ruby, "-e", script], err: %i[child out], &:read)

    assert_match(/INTERRUPTED after [1-9]/, output, output)
    assert_match(/STILL USABLE/, output, output)
    assert_match(/CLOSED true/, output, output)
  end
end
