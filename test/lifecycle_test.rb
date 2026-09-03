# frozen_string_literal: true

require_relative "test_helper"

# How a session behaves under the things a long-running Ruby process does
# to it: a second thread, a panic, a close, a fork. Each of these used to
# end the process; none of them should.
class LifecycleTest < Minitest::Test
  DIM = 64

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
    { params: { "m.l.weight" => { shape: [DIM, DIM], data: Array.new(DIM * DIM, 0.01) },
                "m.l.bias" => { shape: [DIM], data: Array.new(DIM, 0.0) } } }
  end

  def batch(rows: 128)
    { x: { shape: [rows, DIM], data: Array.new(rows * DIM, 1.0) } }
  end

  # A session serves one thread. A second one used to panic inside a
  # RefCell, and a panic that escapes becomes a Ruby fatal: the process
  # ended. Now a watcher reads the last completed step's numbers, and both
  # threads survive (notes/SESSION_CONCURRENCY_SPEC.md section 4).
  def test_a_second_thread_watching_reads_the_last_step_rather_than_refusing
    session = Torobi::Session.open(config, weights)
    seen = []
    refused = []
    reader = Thread.new do
      400.times do
        seen << [session.step, session.loss, session.lr, session.seed]
      rescue StandardError => e
        refused << e.class
      end
    end
    session.run([batch] * 40)
    reader.join
    session.close

    assert_empty refused, "watching should never refuse"
    refute_empty seen
    # Only steps the run actually passed through, and never going backwards.
    steps = seen.map(&:first)

    assert_equal steps, steps.sort
    assert_operator steps.max, :<=, 40
    assert(seen.none? { |_, loss, _, _| loss.nil? })
    assert_equal 40, Torobi::Session.open(config, weights) { |s| s.run([batch] * 40); s.step }
  end

  # Watching is not serving. A second thread that tries to *use* the engine
  # while a step runs is still told it is busy.
  def test_a_second_thread_that_uses_the_engine_is_told_it_is_busy
    session = Torobi::Session.open(config, weights)
    busy = 0
    other = Thread.new do
      200.times do
        session.fetch("m.l.bias")
      rescue Torobi::Busy
        busy += 1
      end
    end
    session.run([batch] * 40)
    other.join
    session.close

    assert_operator busy, :>, 0, "using the engine from a second thread should say busy"
  end

  # Closing releases the engine, and everything afterwards refuses rather
  # than pretending. Idempotent, so an ensure can always call it.
  def test_close_is_idempotent_and_refuses_afterwards
    session = Torobi::Session.open(config, weights)
    session.step!(batch)
    refute_predicate session, :closed?

    assert session.close, "the first close reports that it closed it"
    assert_predicate session, :closed?
    refute session.close, "a second close is a no-op"

    e = assert_raises(Torobi::SessionClosed) { session.step!(batch) }
    assert_match(/closed/, e.message)
    # Not a StepError: a step error means the session is still yours.
    refute_kind_of Torobi::StepError, e

    # Watching still answers. A closed session says where it got to, which
    # is what a caller reporting on a finished run needs.
    assert_equal 1, session.step
    refute_predicate session.loss, :nan?
  end

  # The block form owns the lifetime, so the device memory goes when the
  # block does rather than when the GC gets to it.
  def test_the_block_form_closes_even_when_the_block_raises
    session = nil
    assert_raises(RuntimeError) do
      Torobi::Session.open(config, weights) do |s|
        session = s
        s.step!(batch)
        raise "something went wrong in the caller"
      end
    end
    assert_predicate session, :closed?
  end

  # A span is driven per step from Ruby, so it can be interrupted between
  # steps rather than after the last one.
  def test_a_span_is_interruptible_between_steps
    Torobi::Session.open(config, weights) do |s|
      seen = 0
      assert_raises(RuntimeError) do
        s.run(Array.new(1000) { batch }) do |_loss|
          seen += 1
          raise "stop here" if seen == 3
        end
      end
      assert_equal 3, s.step, "the span stopped where the caller stopped it"
      # And the session is still usable.
      s.step!(batch)
      assert_equal 4, s.step
    end
  end

  # An endless enumerable trains, rather than being materialized first.
  def test_a_span_consumes_its_batches_as_it_goes
    Torobi::Session.open(config, weights) do |s|
      endless = Enumerator.new { |y| loop { y << batch } }
      taken = 0
      assert_raises(RuntimeError) do
        s.run(endless) do |_loss|
          taken += 1
          raise "enough" if taken == 5
        end
      end
      assert_equal 5, s.step
    end
  end

  def test_the_bulk_form_is_bounded
    Torobi::Session.open(config, weights) do |s|
      e = assert_raises(ArgumentError) { s.run_uninterruptible(Array.new(5) { batch }, limit: 3) }
      assert_match(/at most 3 batches/, e.message)
      assert_equal 0, s.step

      s.run_uninterruptible(Array.new(3) { batch }, limit: 3)
      assert_equal 3, s.step
    end
  end

  # Device memory is not something Ruby's GC can see, so it is something to
  # measure. These tests are about the numbers being real, not about a
  # particular figure.
  def test_device_memory_is_observable
    before = Torobi::Memory.report
    assert_operator before.fetch(:limit), :>, 0, "this machine reports a cap"

    Torobi::Session.open(config, weights) do |s|
      s.run([batch(rows: 512)] * 5)
      during = Torobi::Memory.report
      assert_operator during.fetch(:peak), :>, 0, "training should have allocated"
      assert_operator during.fetch(:peak), :>=, during.fetch(:active)
    end

    freed = Torobi::Memory.clear_cache!
    assert_operator freed, :>=, 0
    assert_equal 0, Torobi::Memory.cache, "the cache is empty after clearing it"
  end

  def test_the_peak_can_be_forgotten_and_the_limit_set
    Torobi::Session.open(config, weights) { |s| s.step!(batch) }
    Torobi::Memory.reset_peak!
    assert_operator Torobi::Memory.peak, :<=, Torobi::Memory.active + 1

    original = Torobi::Memory.limit
    Torobi::Memory.limit = 1 << 30
    assert_equal 1 << 30, Torobi::Memory.limit
  ensure
    Torobi::Memory.limit = original if original
  end

  # Sessions that are closed give their memory back, which is the point of
  # close existing at all.
  def test_closing_sessions_returns_their_memory
    Torobi::Memory.clear_cache!
    settled = Torobi::Memory.active

    20.times do
      Torobi::Session.open(config, weights) { |s| s.step!(batch(rows: 256)) }
    end
    Torobi::Memory.clear_cache!

    grown = Torobi::Memory.active - settled
    assert_operator grown, :<, 8 * 1024 * 1024,
                    "twenty opened and closed sessions should not leave megabytes behind"
  end

  # A Metal device does not survive fork, so a child is refused here rather
  # than at the GPU.
  def test_a_forked_child_is_refused
    reader, writer = IO.pipe
    pid = fork do
      reader.close
      begin
        Torobi::Session.open(config, weights) { |s| s.step!(batch) }
        writer.puts "OPENED"
      rescue Torobi::EngineUnavailable => e
        writer.puts "REFUSED #{e.message[0, 60]}"
      rescue StandardError => e
        writer.puts "OTHER #{e.class}"
      end
      writer.close
      exit!(0)
    end
    writer.close
    output = reader.read
    reader.close
    Process.wait(pid)

    assert_match(/REFUSED/, output)
    assert_match(/fork/, output)
  end
end
