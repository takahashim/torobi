# frozen_string_literal: true

module Torobi
  # Hooks: the window's sugar, and nothing more.
  #
  # A hook can do exactly what the window can do (docs/plan.md section
  # 8.4), because it *is* the window: it fires between steps, never inside
  # one, and it holds the session it was given. What it buys is not
  # capability but separation - the experiment's main line stays a program
  # you read top to bottom, while the crosscutting concerns (watching for
  # NaN, keeping the best checkpoint, drawing a bar) live beside it.
  #
  # The fences, from the plan:
  #
  # - hooks fire in registration order, so a run is deterministic
  # - a hook may not call `run` (no reentry)
  # - an exception in a hook stops the span, which is safe because a window
  #   is a point where the state is consistent
  # - what a hook adjusts goes through the same knobs, and is journalled
  class Hooks
    # Where a hook may fire. Closed on purpose: a name here is a promise.
    EVENTS = %i[step span_end plateau nan checkpoint_written].freeze

    # What a hook is handed. A reading of the run, plus the session, whose
    # window capabilities are the hook's capabilities.
    Event = Data.define(:name, :session, :step, :loss, :history) do
      # Whether the loss has stopped falling: nothing in the last
      # `patience` readings improved on the best before them, by more than
      # `by`. Readings that are not numbers do not count as improvement.
      def plateaued?(patience: 3, by: 1e-4)
        return false if history.size <= patience

        recent = history.last(patience)
        best_before = history[0..-(patience + 1)].select { |l| l.is_a?(Numeric) && l.finite? }.min
        return false unless best_before

        recent.none? { |loss| loss.is_a?(Numeric) && loss.finite? && loss < best_before - by }
      end

      def finite? = loss&.finite?
    end

    def initialize(session)
      @session = session
      @registered = []
      @history = []
      @firing = false
    end

    def any? = !@registered.empty?

    # `on(:step, every: 200) { |event| ... }`
    def on(event, every: 1, &block)
      unless EVENTS.include?(event)
        raise ArgumentError, "#{event.inspect} is not a hook event (#{EVENTS.join(", ")})"
      end
      raise ArgumentError, "on(#{event.inspect}) needs a block" unless block

      @registered << { event:, every:, call: block }
      self
    end

    # A policy object: anything answering `call(event)`. The standard ones
    # are in Torobi::Policies.
    def use(policy, event: :step, every: 1)
      unless policy.respond_to?(:call)
        raise ArgumentError, "a policy must respond to #call, and #{policy.class} does not"
      end

      @registered << { event:, every:, call: policy }
      self
    end

    # Called by the session at each window. Not part of the public API.
    #
    # Hooks do not nest: a hook that writes a checkpoint would otherwise
    # fire :checkpoint_written inside itself, and the fence against
    # reentry is about spans, not about that. The inner firing is skipped.
    def fire(name, step:, loss:)
      return if @registered.empty? || @firing

      @history << loss if name == :step && loss
      event = Event.new(name:, session: @session, step:, loss:, history: @history.dup)
      @firing = true
      @registered.each do |hook|
        next unless hook[:event] == name
        next unless (step % hook[:every]).zero?

        hook[:call].call(event)
      end
    ensure
      @firing = false
    end

    # Whether a span is currently inside a hook, which is how the session
    # refuses reentry.
    def firing? = @firing
  end

  # The standard policies: the crosscutting concerns worth writing once.
  # Each is a plain object answering `call(event)`, so a caller's own is
  # exactly as good.
  module Policies
    # Answers a loss that has stopped being a number.
    #
    # There are two of these and they want different answers.
    #
    # A bad batch: one step's numbers overflow, the parameters are fine.
    # The engine does not take a step whose loss is not finite, so this
    # costs that step and nothing else. Carrying on is the whole answer,
    # and it works with no checkpoint at all.
    #
    # A divergence: the parameters themselves are somewhere the forward
    # overflows. Then no step is taken either, which means lowering the
    # rate moves nothing and the run is stuck rather than corrupt. Only
    # going back somewhere else answers this, which is why `rollback` is
    # still worth giving.
    #
    # So: lower the rate, go back if there is somewhere to go back to, and
    # give up after `patience` consecutive non-finite steps, because by
    # then whatever is being tried is not working.
    class NaNGuard
      def initialize(rollback: nil, lr_factor: 0.5, patience: 5)
        @rollback = rollback
        @lr_factor = lr_factor
        @patience = patience
        @consecutive = 0
      end

      # How many non-finite steps have arrived in a row.
      attr_reader :consecutive

      def call(event)
        if event.finite?
          @consecutive = 0
          return
        end

        @consecutive += 1
        session = event.session
        session.observe(nan_at: event.step, consecutive: @consecutive)
        if @consecutive >= @patience
          raise Torobi::StepError,
                "the loss has not been finite for #{@consecutive} steps " \
                "(since step #{event.step - @consecutive + 1}); lowering the rate " \
                "is not answering it"
        end

        session.restore(@rollback) if @rollback && File.exist?(@rollback.to_s)
        session.adjust(lr: session.lr * @lr_factor)
      end
    end

    # Stops a run before the machine notices it.
    #
    # `Torobi::Memory.limit=` is not a refusal, which was measured rather
    # than assumed: a 336 MB peak ran to completion under a 67 MB limit,
    # and a step whose peak reached 14.5 GB ran under a 9 GB one (it took
    # 41s where it usually takes 3, because the allocator was reclaiming
    # rather than failing). MLX treats the limit as pressure on its cache,
    # not as a ceiling.
    #
    # So the ceiling has to be Ruby's. This reads what the device is
    # holding between steps and raises when it passes `bytes`, which the
    # run's process reports as a failure and the parent reads from the
    # journal. On unified memory the alternative is the machine stopping
    # (docs/plan.md section 15.22), which no rescue sees.
    #
    #   s.use(Torobi::Policies::MemoryGuard.new(10 * 1024**3), every: 10)
    #
    # `every:` is worth using: reading costs a call through the runtime's
    # gate, and device memory does not change between steps that are not
    # taking any.
    class MemoryGuard
      # What the device was holding when this last looked.
      attr_reader :seen

      def initialize(bytes, on_limit: nil)
        raise ArgumentError, "a guard needs a positive limit" unless bytes.positive?

        @bytes = Integer(bytes)
        @on_limit = on_limit
        @seen = 0
      end

      def call(event)
        @seen = Torobi::Memory.active
        return if @seen <= @bytes

        event.session.observe(memory_at: event.step, active: @seen, limit: @bytes)
        return @on_limit.call(event) if @on_limit

        raise Torobi::StepError,
              "the device is holding #{@seen / 1024**2} MB at step #{event.step}, "               "past the #{@bytes / 1024**2} MB this run was given. Stopping here "               "rather than letting the machine find out"
      end
    end

    # Keeps a checkpoint of the best the run has been.
    class BestCheckpoint
      attr_reader :best, :path

      def initialize(dir, metric: nil)
        @dir = dir
        @metric = metric
        @best = nil
        @path = nil
      end

      def call(event)
        value = @metric ? @metric.call(event) : event.loss
        return if value.nil? || !value.finite?
        return if @best && value >= @best

        @best = value
        @path = event.session.checkpoint!(@dir)
      end
    end

    # Halves the learning rate when the loss stops falling.
    class LrOnPlateau
      def initialize(factor: 0.5, patience: 3, by: 1e-4, min_lr: 1e-6)
        @factor = factor
        @patience = patience
        @by = by
        @min_lr = min_lr
        @last_change = 0
      end

      def call(event)
        return unless event.plateaued?(patience: @patience, by: @by)
        # One change per plateau, not one per step of it.
        return if @last_change.positive? && event.step - @last_change < @patience

        session = event.session
        next_lr = [session.lr * @factor, @min_lr].max
        return if next_lr >= session.lr

        @last_change = event.step
        session.observe(plateau_at: event.step, loss: event.loss)
        session.adjust(lr: next_lr)
      end
    end

    # Stops the run when it stops improving. Raises, which stops the span
    # where the plan says a hook's exception stops it.
    class EarlyStopping
      class Stop < Torobi::Error; end

      def initialize(patience: 5, by: 1e-4)
        @patience = patience
        @by = by
      end

      def call(event)
        return unless event.plateaued?(patience: @patience, by: @by)

        event.session.observe(early_stop_at: event.step, loss: event.loss)
        raise Stop, "the loss has not improved for #{@patience} steps; stopping at #{event.step}"
      end
    end

    # Anything that wants the numbers: a bar, a log, a tracker.
    class Progress
      def initialize(&block)
        @block = block or raise ArgumentError, "Progress needs a block"
      end

      def call(event) = @block.call(event.step, event.loss)
    end
  end
end
