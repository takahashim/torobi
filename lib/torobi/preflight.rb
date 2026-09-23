# frozen_string_literal: true

module Torobi
  # What must hold before the engine is asked to do anything.
  #
  # Not defensive programming for its own sake: some MLX failures do not
  # come back as errors. A machine with no working Metal device raises a
  # C++ exception during device initialization, which Rust cannot catch;
  # the process exits with nothing for Ruby to rescue.
  #
  # This file holds the checks only Ruby can make. The missing-metallib
  # refusal moved into the engine's runtime, where it also covers callers
  # that never pass through here (`Torobi::Native` used directly, the
  # engine's own CLI and tests). What stays is the probe: an external
  # review reached a machine where the metallib was present and
  # initialization still aborted, and the only honest way to ask "will the
  # device start" is in a subprocess, where an abort is an exit status
  # rather than the end of this process (docs/plan.md section 4.1).
  module Preflight
    # Where MLX will look for its Metal kernels: beside the library holding
    # the MLX symbols, found through dladdr. The engine refuses when the
    # file is not there; this constant remains for the tests that hide it.
    METALLIB = File.expand_path("mlx.metallib", __dir__)

    # The process that loaded the extension. A Metal device and its command
    # queues do not survive fork, so a child that inherited them cannot use
    # them, and finding out at the GPU is finding out by aborting.
    ORIGIN_PID = Process.pid

    # Refuses, before MLX is touched, what would otherwise fail at the GPU
    # or end the process: a forked child, and a device that cannot start.
    def self.check!
      unless Process.pid == ORIGIN_PID
        raise Torobi::EngineUnavailable,
              "this process (#{Process.pid}) inherited Torobi from a fork of " \
              "#{ORIGIN_PID}. A Metal device does not survive fork, so a session " \
              "here would fail at the GPU rather than here. Run training in a " \
              "process started with Process.spawn or exec, not in a prefork " \
              "worker (Puma clustered, Sidekiq, Spring)."
      end

      probe = Probe.current
      return if probe.ok?

      raise Torobi::EngineUnavailable, "MLX cannot start on this machine: #{probe.reason}"
    end

    # Whether MLX could initialize, as one process found out: its pid, and
    # nil or the reason it could not.
    #
    # Asked in a subprocess, because the failure it looks for is an abort:
    # asking in this process is the thing we are trying to avoid. The answer
    # is kept against the pid that learned it (`current`), so a forked
    # child does not inherit its parent's answer: it would be answering
    # about the parent's device, not its own.
    class Probe < Data.define(:pid, :reason)
      # Enough to make MLX build a device and run a kernel.
      SCRIPT = <<~RUBY.freeze
        $LOAD_PATH.unshift(#{File.expand_path("..", __dir__).inspect})
        require "torobi"
        model = Torobi.graph do |g|
          x = g.input :x, [nil, 1]
          g.output :loss, g.mean(g.linear(x, 1, name: "probe"))
        end
        config = Torobi::GraphConfig.new(models: { "m" => model })
        weights = { params: { "m.probe.weight" => { shape: [1, 1], data: [0.0] },
                              "m.probe.bias" => { shape: [1], data: [0.0] } } }
        # The engine directly, and nothing of the session around it: what
        # this asks is whether the device starts, and a failure anywhere
        # else would be reported as that.
        native = Torobi::Native::Session.open(
          config.canonical_json, JSON.generate(weights),
          JSON.generate(Torobi::Session::DEFAULT_OPTIMIZER),
          Torobi::Session::DEFAULT_SEED
        )
        native.run_step({ "x" => ["f32", [1, 1], [1.0].pack("f*")] })
        print "ok"
      RUBY

      # This process's answer, asked for the first time if it has not been.
      def self.current
        @current = run unless @current&.pid == Process.pid
        @current
      end

      # Forgets the answer, for tests that need the probe run again.
      def self.forget! = @current = nil

      # Asks, in a subprocess, now.
      def self.run
        require "open3"
        output, status = Open3.capture2e(RbConfig.ruby, "-e", SCRIPT)
        ok = status.success? && output.end_with?("ok")
        new(pid: Process.pid, reason: ok ? nil : reason_from(output))
      rescue StandardError => e
        new(pid: Process.pid, reason: e.message)
      end

      # The child's own words, when it had any. An engine refusal arrives as
      # an exception message; an abort leaves whatever MLX printed on the way
      # down; a silent death leaves nothing, and then the generic truth is
      # all there is to say.
      def self.reason_from(output)
        line = output.lines.map(&:strip).reject(&:empty?).first
        unless line
          return "initializing its device ended a probe process rather than raising. " \
                 "Torobi needs a working GPU for the backend it was built against: " \
                 "Metal on Apple silicon, or an NVIDIA card and its driver " \
                 "elsewhere; see docs/vendoring.md."
        end

        # An unrescued Ruby exception prints as "-e:12:in '<main>': message".
        line.sub(/\A.*?:\d+:in '.*?': /, "")
      end

      def ok? = reason.nil?
    end
  end
end
