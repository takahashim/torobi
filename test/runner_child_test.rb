# frozen_string_literal: true

require_relative "test_helper"
require "tmpdir"
require "json"

# The child's half of `Torobi::Runner`: the run directory it owns, the stop
# flag, and how it says on the way out which way it went. The parent's half
# is in `RunnerTest`; this is the code that runs in the other process, held
# to directly rather than only through one.
class RunnerChildTest < Minitest::Test
  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
    @dir = Dir.mktmpdir("torobi-child")
    # `listen` replaces the process's TERM and INT handlers, and `cap!`
    # moves the allocator's cap; both are process-global, so both go back.
    @signals = %w[TERM INT].to_h { |name| [name, Signal.trap(name, "DEFAULT")] }
    @limit = Torobi::Memory.limit
  end

  def teardown
    return unless @dir

    @signals.each { |name, handler| Signal.trap(name, handler) }
    Torobi::Memory.limit = @limit
    FileUtils.rm_rf(@dir)
  end

  def child(memory_limit: nil) = Torobi::Runner::Child.new(@dir, memory_limit:)

  def journal_lines
    File.readlines(File.join(@dir, Torobi::Runner::JOURNAL)).map { |line| JSON.parse(line) }
  end

  def test_a_child_owns_its_run_directory
    c = child

    assert_equal @dir, c.dir
    assert_nil c.memory_limit
    refute_predicate c, :stopping?
    assert_equal File.join(@dir, Torobi::Runner::CHECKPOINT), c.checkpoint.dir
    refute_predicate c.checkpoint, :exist?
  end

  def test_the_journal_is_one_appending_stream
    File.write(File.join(@dir, Torobi::Runner::JOURNAL), "first\n")
    c = child

    assert_same c.journal, c.journal, "held open for the run, not reopened each write"
    assert_predicate c.journal, :sync, "flushed per entry, so a killed run leaves a readable file"
    c.journal.puts "second"

    assert_equal %w[first second],
                 File.readlines(File.join(@dir, Torobi::Runner::JOURNAL)).map(&:chomp)
  end

  def test_closing_is_idempotent
    c = child
    c.journal.puts "x"
    c.close
    c.close

    assert_equal ["x"], File.readlines(File.join(@dir, Torobi::Runner::JOURNAL)).map(&:chomp)
  end

  def test_capping_tells_the_allocator_what_the_run_may_hold
    c = child(memory_limit: 1 << 20)

    assert_equal 1 << 20, c.cap!
    assert_equal 1 << 20, Torobi::Memory.limit
  end

  def test_no_limit_leaves_the_allocator_alone
    before = Torobi::Memory.limit

    assert_nil child.cap!
    assert_equal before, Torobi::Memory.limit
  end

  def test_listen_arms_the_stop_flag_and_returns_itself
    c = child

    assert_same c, c.listen
    refute_predicate c, :stopping?
  end

  def test_a_run_that_ends_writes_that_it_finished
    ran = false
    code = Torobi::Runner.child(@dir) { |_run| ran = true }

    assert ran, "the block should have run"
    assert_equal Torobi::Runner::EXIT_OK, code
    assert_equal "finished", journal_lines.last.fetch("event")
  end

  def test_a_child_says_when_the_engine_cannot_run_here
    code = Torobi::Runner.child(@dir) { raise Torobi::EngineUnavailable, "no device" }
    note = journal_lines.last

    assert_equal Torobi::Runner::EXIT_UNAVAILABLE, code
    assert_equal "unavailable", note.fetch("event")
    assert_equal "no device", note.fetch("message")
  end

  def test_a_child_says_when_the_run_raised
    code = Torobi::Runner.child(@dir) { raise "boom" }
    note = journal_lines.last

    assert_equal Torobi::Runner::EXIT_FAILED, code
    assert_equal "failed", note.fetch("event")
    assert_equal "RuntimeError: boom", note.fetch("message")
  end

  def test_a_child_needs_a_run_directory
    e = assert_raises(ArgumentError) { Torobi::Runner.child(nil) }

    assert_match(/no run directory/, e.message)
  end

  # A journal that cannot be written to is not worth failing a run over:
  # the note is the last thing written, and losing it loses only the note.
  def test_a_child_whose_journal_is_gone_still_ends
    code = Torobi::Runner.child(@dir) { |run| run.journal.close }

    assert_equal Torobi::Runner::EXIT_OK, code
  end

  def test_child_bang_exits_with_what_child_returns
    e = assert_raises(SystemExit) { Torobi::Runner.child!(@dir) { |_run| nil } }

    assert_equal Torobi::Runner::EXIT_OK, e.status
  end
end
