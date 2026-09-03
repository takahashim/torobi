# frozen_string_literal: true

require_relative "test_helper"
require "tmpdir"

# A long run belongs in a process of its own, and the contract for that is
# a directory: the child writes its journal and its checkpoints there, the
# parent reads both (notes/SESSION_CONCURRENCY_SPEC.md section 11).
class RunnerTest < Minitest::Test
  SCRIPT = File.expand_path("support/train.rb", __dir__)

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
    @dir = Dir.mktmpdir("torobi-run")
  end

  def teardown
    FileUtils.remove_entry(@dir) if @dir && File.exist?(@dir)
  end

  def runner(env = {})
    Torobi::Runner.new([RbConfig.ruby, SCRIPT], dir: @dir, env:)
  end

  # Waits for the child to say something, and gives up.
  #
  # A test that waits forever cannot report anything: a child that dies
  # before its first step leaves the parent spinning, which reads as a
  # hung machine rather than as the failure it is. Ten seconds is long
  # against a run that starts in tenths of one, and short against a CI
  # job nobody is watching.
  def moving(runner, within: 10)
    deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + within
    sleep(0.05) while runner.running? && runner.progress.nil? &&
                      Process.clock_gettime(Process::CLOCK_MONOTONIC) < deadline
    return runner if runner.progress

    flunk "the run wrote no progress in #{within}s (#{runner.outcome&.to_s || "still going"})"
  end

  def test_a_run_finishes_and_says_so
    r = runner("STEPS" => "20").start
    outcome = r.wait

    assert_predicate outcome, :finished?, outcome.to_s
    refute_predicate outcome, :crashed?
    assert_equal 20, r.progress.fetch(:step)
    assert_equal 20, r.checkpoint_manifest.fetch("step")
  end

  # The parent reads the journal rather than asking the child, so progress
  # is answerable while the child is inside a step and after it is gone.
  def test_progress_is_readable_while_the_run_is_going
    r = runner("STEPS" => "100000", "CHECKPOINT_EVERY" => "1000").start
    seen = []
    # The child has an engine to open first, so wait for the run to move
    # before sampling it.
    moving(r)
    30.times do
      break unless r.running?

      seen << r.progress
      sleep 0.01
    end
    r.stop

    reported = seen.compact.map { _1[:step] }

    refute_empty reported, "the parent should have seen the run move"
    assert_equal reported.sort, reported, "progress only goes forward"
  end

  # Progress is asked for over and over, so what it reads is what has been
  # appended since it last read: a poll costs a step's worth of journal
  # rather than the whole of it. Written here rather than run, because the
  # only way to see it is to write the journal by hand.
  def test_progress_reads_what_was_appended_and_no_more
    r = runner
    path = r.journal_path
    write = ->(step) { File.open(path, "a") { |f| f.puts JSON.generate(kind: "span", step:) } }

    assert_nil r.progress, "an empty directory has no progress to report"
    write.call(1)

    assert_equal 1, r.progress.fetch(:step)
    write.call(2)

    assert_equal 2, r.progress.fetch(:step)

    # A poll can land between an entry and its newline. The half-written
    # line is held rather than skipped, so the entry is not lost.
    File.open(path, "a") { |f| f.write(JSON.generate(kind: "span", step: 3)) }

    assert_equal 2, r.progress.fetch(:step), "an unfinished line is not an entry yet"
    File.open(path, "a", &:puts)

    assert_equal 3, r.progress.fetch(:step)
  end

  # A journal is only ever appended to, so a file that got shorter is a
  # different journal and the position kept from the last one means
  # nothing. Reading it from the start is the only answer that is right.
  def test_a_replaced_journal_is_read_again_rather_than_continued
    r = runner
    File.open(r.journal_path, "a") do |f|
      3.times { |i| f.puts JSON.generate(kind: "span", step: 100 + i) }
    end

    assert_equal 102, r.progress.fetch(:step)

    File.write(r.journal_path, "#{JSON.generate(kind: "span", step: 7)}\n")

    assert_equal 7, r.progress.fetch(:step)
  end

  # TERM asks; it does not insist. The child finishes the step it is in,
  # writes what it reached, and exits cleanly.
  def test_stopping_is_asked_for_and_leaves_a_usable_checkpoint
    r = runner("STEPS" => "100000", "CHECKPOINT_EVERY" => "1000").start
    moving(r)
    outcome = r.stop(grace: 20)

    assert_predicate outcome, :stopped?, outcome.to_s
    refute_predicate outcome, :crashed?
    manifest = r.checkpoint_manifest

    refute_nil manifest, "a stopped run leaves the state it reached"
    assert_operator manifest.fetch("step"), :>, 0
  end

  # The point of the arrangement: a run that fails is an exit status here,
  # not the end of this process.
  def test_a_failed_run_is_an_exit_status_and_a_reason
    outcome = runner("RAISE" => "the data was not there").start.wait

    assert_predicate outcome, :failed?, outcome.to_s
    refute_predicate outcome, :finished?
    assert_match(/the data was not there/, outcome.message)
    assert_equal "failed", outcome.event
  end

  # The case the arrangement exists for. A process that is ended rather
  # than ending leaves no exception behind, and the parent has to learn it
  # from the exit status. What it left on disk is still whole, because a
  # checkpoint is renamed into place rather than written in place.
  def test_a_crashed_run_is_reported_and_its_last_checkpoint_stands
    r = runner("STEPS" => "1000", "CHECKPOINT_EVERY" => "5", "ABORT_AT" => "20").start
    outcome = r.wait

    assert_predicate outcome, :crashed?, outcome.to_s
    refute_predicate outcome, :finished?
    refute_predicate outcome, :failed?, "a signal is not an exit status"
    assert_match(/SIGABRT/, outcome.to_s)

    manifest = r.checkpoint_manifest

    refute_nil manifest, "the checkpoint written before the crash is still there"
    assert_operator manifest.fetch("step"), :>, 0
    assert_operator manifest.fetch("step"), :<=, 20
    # And it is a checkpoint, not a directory of fragments.
    assert_equal manifest.fetch("step"), manifest.dig("run", "position", "step")
  end

  # A directory that already holds a checkpoint is how a run resumes, so
  # starting against one must not wipe it.
  def test_a_second_run_resumes_from_what_the_first_left
    first = runner("STEPS" => "20").start.wait

    assert_predicate first, :finished?, first.to_s
    reached = Torobi::Checkpoint.manifest(File.join(@dir, "checkpoint")).fetch("step")

    second = runner("STEPS" => "20").start

    assert_predicate second.wait, :finished?, second.outcome.to_s
    assert_equal reached + 20, second.checkpoint_manifest.fetch("step"),
                 "the second run carried on from the first"
  end

  # MLX draws from the same memory the machine runs on, so an uncapped run
  # that asks for too much does not fail: the machine stops answering and
  # its watchdog ends the session. A cap turns that into an exception the
  # run can report, which is much of why a long run lives in a process of
  # its own.
  def test_a_cap_reaches_the_child_and_is_in_force
    skip "extension not compiled" unless defined?(Torobi::Session)

    script = <<~SCRIPT
      $LOAD_PATH.unshift(#{File.expand_path("../lib", __dir__).inspect})
      require "torobi"
      Torobi::Runner.child! do |run|
        File.write(File.join(run.dir, "limit.txt"), Torobi::Memory.report[:limit])
      end
    SCRIPT
    file = File.join(@dir, "capped.rb")
    File.write(file, script)
    outcome = Torobi::Runner.new([RbConfig.ruby, file], dir: @dir,
                                 memory_limit: 3_000_000_000).start.wait

    assert_predicate outcome, :finished?, outcome.to_s
    assert_equal 3_000_000_000, Integer(File.read(File.join(@dir, "limit.txt")).strip)
  end

  # And without one, nothing is imposed: a spike should not have to ask.
  def test_no_cap_is_the_default
    skip "extension not compiled" unless defined?(Torobi::Session)

    script = <<~SCRIPT
      $LOAD_PATH.unshift(#{File.expand_path("../lib", __dir__).inspect})
      require "torobi"
      Torobi::Runner.child! do |run|
        File.write(File.join(run.dir, "limit.txt"), Torobi::Memory.report[:limit])
      end
    SCRIPT
    file = File.join(@dir, "uncapped.rb")
    File.write(file, script)
    Torobi::Runner.new([RbConfig.ruby, file], dir: @dir).start.wait
    reported = Integer(File.read(File.join(@dir, "limit.txt")).strip)

    refute_equal 3_000_000_000, reported
  end

  def test_a_runner_refuses_to_start_twice
    r = runner("STEPS" => "5").start
    assert_raises(Torobi::Error) { r.start }
    r.wait
  end

  def test_the_outcome_of_a_run_that_never_started
    r = runner

    assert_nil r.outcome
    assert_raises(Torobi::Error) { r.wait }
  end
end
