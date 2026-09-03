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

    module_function

    def check!
      unless Process.pid == ORIGIN_PID
        raise Torobi::EngineUnavailable,
              "this process (#{Process.pid}) inherited Torobi from a fork of " \
              "#{ORIGIN_PID}. A Metal device does not survive fork, so a session " \
              "here would fail at the GPU rather than here. Run training in a " \
              "process started with Process.spawn or exec, not in a prefork " \
              "worker (Puma clustered, Sidekiq, Spring)."
      end

      return if probe_result

      raise Torobi::EngineUnavailable, "MLX cannot start on this machine: #{@probe_reason}"
    end

    # Whether MLX can actually initialize, asked once per process.
    #
    # Asked in a subprocess, because the failure it looks for is an abort:
    # asking in this process is the thing we are trying to avoid. The answer
    # is memoized against the pid that learned it, so a forked child does
    # not inherit its parent's answer (it would be answering about the
    # parent's device, not its own).
    def probe_result
      return @probe_result if defined?(@probe_pid) && @probe_pid == Process.pid

      @probe_pid = Process.pid
      @probe_result = probe!
    end

    def probe!
      require "open3"
      script = <<~RUBY
        $LOAD_PATH.unshift(#{File.expand_path("..", __dir__).inspect})
        require "torobi"
        # Enough to make MLX build a device and run a kernel.
        model = Torobi.graph do |g|
          x = g.input :x, [nil, 1]
          g.output :loss, g.mean(g.linear(x, 1, name: "probe"))
        end
        config = Torobi::GraphConfig.new(models: { "m" => model })
        weights = { params: { "m.probe.weight" => { shape: [1, 1], data: [0.0] },
                              "m.probe.bias" => { shape: [1], data: [0.0] } } }
        native = Torobi::Native::Session.open(
          config.canonical_json, JSON.generate(weights),
          JSON.generate(Torobi::Session::DEFAULT_OPTIMIZER)
        )
        native.run_step({ "x" => ["f32", [1, 1], [1.0].pack("f*")] })
        print "ok"
      RUBY
      output, status = Open3.capture2e(RbConfig.ruby, "-e", script)
      ok = status.success? && output.end_with?("ok")
      @probe_reason = ok ? nil : reason_from(output)
      ok
    rescue StandardError => e
      @probe_reason = e.message
      false
    end

    # The child's own words, when it had any. An engine refusal arrives as
    # an exception message; an abort leaves whatever MLX printed on the way
    # down; a silent death leaves nothing, and then the generic truth is
    # all there is to say.
    def reason_from(output)
      line = output.lines.map(&:strip).reject(&:empty?).first
      return "initializing its device ended a probe process rather than raising. " \
             "Torobi needs Apple silicon with a working Metal device; " \
             "see docs/vendoring.md." unless line

      # An unrescued Ruby exception prints as "-e:12:in '<main>': message".
      line.sub(/\A.*?:\d+:in '.*?': /, "")
    end

    # For tests that need the probe run again.
    def forget_probe!
      @probe_pid = nil
      @probe_result = nil
      @probe_reason = nil
    end
  end
end
