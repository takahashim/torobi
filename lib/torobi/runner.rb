# frozen_string_literal: true

require "json"

module Torobi
  # A training run in a process of its own.
  #
  # Torobi asks for this rather than offering it as an option
  # (notes/SESSION_CONCURRENCY_SPEC.md sections 2 and 9, docs/plan.md 11.4).
  # A mutex inside a process is not fault isolation: MLX can end the process
  # it runs in, and no `rescue` sees that happen. A long run therefore
  # belongs somewhere its death is an exit status rather than the end of the
  # caller. Which also settles the fork question: the child is exec'd, so it
  # brings no Metal device with it.
  #
  # The two sides share a directory and nothing else. The child writes its
  # journal and its checkpoints there; the parent reads both. No pipe to
  # keep open, no protocol to version, and an interrupted run leaves exactly
  # what a finished one would, minus the last entry.
  #
  #   runner = Torobi::Runner.new(["ruby", "train.rb"], dir: "runs/001")
  #   runner.start
  #   runner.progress          # => {step: 1400, loss: 0.21, at: "..."}
  #   runner.stop              # asks it to stop at the next step boundary
  #   runner.wait.finished?    # => true
  #
  # And in train.rb:
  #
  #   Torobi::Runner.child do |run|
  #     Torobi::Session.open(config, weights, io: run.journal) do |s|
  #       s.run(batches) do
  #         s.checkpoint!(run.checkpoint) if (s.step % 200).zero?
  #         break if run.stopping?
  #       end
  #     end
  #   end
  class Runner
    # The run directory as both sides see it.
    JOURNAL = "journal.jsonl"
    CHECKPOINT = "checkpoint"

    # How the child says which way it ended, in the variable it is started
    # with. A directory rather than a pipe means the parent can also read
    # this after the fact, from a run it did not start.
    DIRECTORY_VARIABLE = "TOROBI_RUN_DIR"

    # What the child exits with. A signal is not in here on purpose: that is
    # the case this whole arrangement exists for, and it arrives as a signal
    # rather than as a number.
    EXIT_OK = 0
    EXIT_FAILED = 70          # EX_SOFTWARE: the run raised
    EXIT_UNAVAILABLE = 69     # EX_UNAVAILABLE: the engine cannot run here

    def initialize(command, dir:, env: {})
      @command = Array(command).map(&:to_s)
      raise ArgumentError, "a runner needs a command" if @command.empty?

      @dir = File.expand_path(dir.to_s)
      @env = env.transform_keys(&:to_s).transform_values(&:to_s)
      @pid = nil
      @outcome = nil
    end

    attr_reader :dir, :pid

    def journal_path = File.join(@dir, JOURNAL)
    def checkpoint = File.join(@dir, CHECKPOINT)

    # Starts the run. The directory is made, but nothing in it is removed:
    # a runner started against a directory that already holds a checkpoint
    # is how a run resumes, and deleting that would be the wrong default.
    def start
      raise Error, "this runner has already been started" if @pid

      require "fileutils"
      FileUtils.mkdir_p(@dir)
      @pid = Process.spawn(@env.merge(DIRECTORY_VARIABLE => @dir), *@command,
                           chdir: Dir.pwd, unsetenv_others: false)
      self
    end

    # Whether the child is still going.
    #
    # Reaping is not free of consequence: asking this is the only chance to
    # learn the exit status, so it is kept when it arrives rather than
    # thrown away. A `running?` that discarded it left `wait` with nothing
    # to report.
    def running?
      return false unless @pid && @outcome.nil?

      pid, status = Process.waitpid2(@pid, Process::WNOHANG)
      return true if pid.nil?

      reap(status)
      false
    rescue Errno::ECHILD
      false
    end

    # What the child has written down so far: the last step it recorded,
    # and what that step cost. Nil before the first one.
    #
    # Read from the journal rather than asked of the child, because the
    # child may be inside a step, or gone. A record on disk answers either
    # way (the journal flushes per entry for exactly this).
    def progress
      last = last_entry { |e| e["kind"] == "span" }
      return nil unless last

      { step: last["step"], loss: last["loss"], at: last["at"] }
    end

    # Whether a checkpoint is there to resume from, and what it says.
    def checkpoint_manifest
      Checkpoint.manifest(checkpoint) if Checkpoint.exist?(checkpoint)
    end

    # Asks the run to stop at the next step boundary and waits for it.
    #
    # TERM rather than KILL: the child finishes the step it is in, writes a
    # last checkpoint if it was asked to, and exits. `grace` is how long to
    # allow for that before insisting, because a step that has wedged the
    # GPU will not answer.
    def stop(grace: 30)
      return @outcome unless running?

      signal("TERM")
      deadline = now + grace
      sleep(0.02) while running? && now < deadline
      if running?
        signal("KILL")
        sleep(0.02) while running?
      end
      wait
    end

    # Blocks until the run ends, and says how.
    def wait
      return @outcome if @outcome
      raise Error, "this runner has not been started" unless @pid

      _, status = Process.waitpid2(@pid)
      reap(status)
    rescue Errno::ECHILD
      @outcome ||= Outcome.new(status: nil, note: nil)
    end

    def outcome = @outcome

    private

    # The child has ended; read what it said on the way out.
    def reap(status)
      @outcome = Outcome.new(status:, note: last_entry { |e| e["kind"] == "note" })
    end

    def signal(name)
      Process.kill(name, @pid)
    rescue Errno::ESRCH
      nil
    end

    def now = Process.clock_gettime(Process::CLOCK_MONOTONIC)

    # The last journal entry a caller cares about. Written a line at a time
    # and flushed, so the only line that can be incomplete is the last one;
    # anything that does not parse is skipped rather than raised on.
    def last_entry
      return nil unless File.exist?(journal_path)

      File.readlines(journal_path).reverse_each do |line|
        entry = begin
          JSON.parse(line)
        rescue JSON::ParserError
          next
        end
        return entry if yield(entry)
      end
      nil
    end

    # How a run ended, and what it said on the way out.
    class Outcome
      def initialize(status:, note:)
        @status = status
        @note = note
      end

      attr_reader :status, :note

      # It ran to the end of what it was given.
      def finished? = ok? && event == "finished"

      # It was asked to stop and did.
      def stopped? = ok? && event == "stopped"

      # It raised. The journal's last note says what.
      def failed? = !@status.nil? && !ok? && @status.exitstatus == EXIT_FAILED

      # The engine could not run there at all.
      def unavailable? = !@status.nil? && @status.exitstatus == EXIT_UNAVAILABLE

      # The process was ended rather than ending: a signal, which for this
      # library usually means MLX aborted. The last checkpoint still stands,
      # because a checkpoint is written atomically.
      def crashed? = !@status.nil? && !@status.signaled?.nil? && @status.signaled?

      def ok? = !@status.nil? && @status.success?

      # What the run said as it ended, if it said anything.
      def message = @note && @note["message"]

      def event = @note && @note["event"]

      def to_s
        return "not started" if @status.nil?
        return "crashed (SIG#{Signal.signame(@status.termsig)})" if crashed?

        "#{event || "exited"} (#{@status.exitstatus})"
      end
    end

    # The child's half: where to write, and whether to stop.
    #
    # It owns two things and no more. The run directory, so both sides agree
    # without being told twice. And the stop signals, because the parent's
    # only way to ask is a signal, and a signal handler must set a flag
    # rather than do work.
    class Child
      def initialize(dir)
        @dir = dir
        @stopping = false
        @journal = nil
      end

      attr_reader :dir

      def checkpoint = File.join(@dir, CHECKPOINT)

      # The journal to hand `Session.open(io:)`. Appended to, so a resumed
      # run adds to the record rather than replacing it.
      def journal
        @journal ||= File.open(File.join(@dir, JOURNAL), "a").tap { |io| io.sync = true }
      end

      # Whether the parent has asked this run to stop. The loop checks it
      # between steps, which is the only place stopping is well defined
      # (the spec, section 7).
      def stopping? = @stopping

      # Installs the handlers. A trap runs at an unspecified moment, so it
      # does the smallest thing that can be done: set a flag.
      def listen
        %w[TERM INT].each { |name| Signal.trap(name) { @stopping = true } }
        self
      end

      def close
        @journal&.close
        @journal = nil
      end
    end

    class << self
      # The child side. Runs `block` with a `Child`, and turns however it
      # ends into an exit status the parent can read.
      #
      # The last thing written is always a note saying which way it went, so
      # a parent that arrives after the fact learns the same thing a parent
      # that waited would.
      def child(dir = ENV.fetch(DIRECTORY_VARIABLE, nil))
        raise ArgumentError, "no run directory (#{DIRECTORY_VARIABLE})" unless dir

        run = Child.new(dir).listen
        begin
          yield run
          finish(run, run.stopping? ? "stopped" : "finished")
          EXIT_OK
        rescue EngineUnavailable => e
          finish(run, "unavailable", e.message)
          EXIT_UNAVAILABLE
        rescue StandardError => e
          finish(run, "failed", "#{e.class}: #{e.message}")
          EXIT_FAILED
        ensure
          run.close
        end
      end

      # The same, and exits with what it returns. What a training script
      # ends with.
      def child!(dir = ENV.fetch(DIRECTORY_VARIABLE, nil), &)
        exit(child(dir, &))
      end

      private

      # A note of its own, appended by hand: the session's journal may
      # already be closed by the time a run ends, and this has to be the
      # last line either way.
      def finish(run, event, message = nil)
        entry = { "kind" => "note", "step" => nil, "at" => Time.now.utc.iso8601,
                  "event" => event, "message" => message }.compact
        run.journal.puts(JSON.generate(entry))
        run.journal.flush
      rescue IOError, Errno::EBADF
        nil
      end
    end
  end
end
