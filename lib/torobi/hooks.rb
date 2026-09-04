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

    # The losses a run has seen, kept the way anything asks about them.
    #
    # `plateaued?` asks two things: the last few readings, and the best of
    # everything before them. An Array answers both and grows for the
    # length of the run, which was fine until you notice that every window
    # carried a copy of it: measured at 40,000 windows that was 10 us a
    # step and rising. Nothing beside a training step, and unbounded all
    # the same.
    #
    # So the recent readings are kept and the rest are folded into their
    # minimum as they fall out, which is all anything asked of them.
    class History
      # How many readings are kept whole. Far past any patience anybody
      # sets, so what falls out is only ever read as "the best before".
      KEEP = 10_000

      def initialize(keep: KEEP)
        @keep = keep
        @recent = []
        @dropped = nil
        @size = 0
      end

      # How many readings there have been, not how many are kept.
      attr_reader :size

      def <<(loss)
        @size += 1
        @recent << loss
        return self if @recent.size <= @keep

        gone = @recent.shift
        @dropped = gone if number?(gone) && (@dropped.nil? || gone < @dropped)
        self
      end

      # The last `count` readings, oldest first.
      def last(count) = @recent.last(count)

      # The best reading before the last `count`, or nil if there is none.
      def best_before(count)
        kept = @recent[0..-(count + 1)] || []
        [@dropped, *kept.select { |loss| number?(loss) }].compact.min
      end

      def to_a = @recent.dup

      private

      def number?(loss) = loss.is_a?(Numeric) && loss.finite?
    end

    # What a hook is handed. A reading of the run, plus the session, whose
    # window capabilities are the hook's capabilities.
    Event = Data.define(:name, :session, :step, :loss, :readings) do
      # The losses so far, as an Array. Built when something asks, which
      # most events never are.
      def history = readings.to_a

      # Whether the loss has stopped falling: nothing in the last
      # `patience` readings improved on the best before them, by more than
      # `by`. Readings that are not numbers do not count as improvement.
      def plateaued?(patience: 3, by: 1e-4)
        return false if readings.size <= patience

        best_before = readings.best_before(patience)
        return false unless best_before

        readings.last(patience)
                .none? { |loss| loss.is_a?(Numeric) && loss.finite? && loss < best_before - by }
      end

      def finite? = loss&.finite?
    end

    def initialize(session)
      @session = session
      @registered = []
      @history = History.new
      @firing = false
    end

    def any? = !@registered.empty?

    # `on(:step, every: 200) { |event| ... }`
    def on(event, every: 1, &block)
      unless EVENTS.include?(event)
        raise ArgumentError, "#{event.inspect} is not a hook event (#{EVENTS.join(", ")})"
      end
      raise ArgumentError, "on(#{event.inspect}) needs a block" unless block

      @registered << { event:, every: cadence(every), call: block }
      self
    end

    # A policy object: anything answering `call(event)`. The standard ones
    # are in Torobi::Policies.
    def use(policy, event: :step, every: 1)
      unless policy.respond_to?(:call)
        raise ArgumentError, "a policy must respond to #call, and #{policy.class} does not"
      end

      @registered << { event:, every: cadence(every), call: policy }
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
      event = Event.new(name:, session: @session, step:, loss:, readings: @history)
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

    private

    # How many windows apart a hook fires.
    #
    # Checked where it is given rather than where it is divided by: a 0
    # is a ZeroDivisionError at the first step, which arrives far from
    # the call that caused it and says nothing about which knob it was.
    def cadence(every)
      every = Integer(every)
      return every if every.positive?

      raise ArgumentError,
            "every: is how many windows apart a hook fires, so it is at least 1 " \
            "(got #{every})"
    end
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

    # The rate over the run, which every fine-tune has an opinion about.
    #
    # Warmup, then decay. Both halves are there for a reason a run shows
    # rather than argues: the first steps of a fine-tune move a model
    # that is already good, and moving it at full rate before AdamW has
    # any idea of the gradient's scale is how a good model is spent. The
    # decay is what lets the last steps settle instead of bouncing.
    #
    #   s.use(Torobi::Policies::Schedule.new(peak: 1e-5, total: 3000), every: 10)
    #
    # `every:` is worth using and costs nothing in accuracy: over ten
    # steps a cosine moves by a thousandth of its range, and the journal
    # records a knob turn each time it is turned.
    #
    # It is not a scheduler in the framework's sense: it is an object
    # that turns the same knob a caller could turn, which is what every
    # policy here is (docs/plan.md section 8.4).
    class Schedule
      # `warmup` is a share of the run when it is below one, and a number
      # of steps when it is not.
      def initialize(peak:, total:, warmup: 0.05, floor: 0.0)
        raise ArgumentError, "a schedule needs a positive peak" unless peak.positive?
        raise ArgumentError, "a schedule needs a positive length" unless total.positive?

        @peak = Float(peak)
        @total = Integer(total)
        @warmup = warmup < 1 ? (total * warmup).round : Integer(warmup)
        @floor = Float(floor)
      end

      attr_reader :peak, :total, :warmup, :floor

      # The rate at a step. Public so it can be looked at without a run:
      # a schedule nobody can plot is a schedule nobody can check.
      def at(step)
        return @peak * (step + 1) / (@warmup + 1).to_f if step < @warmup

        over = (step - @warmup).to_f / [@total - @warmup, 1].max
        cosine = 0.5 * (1 + Math.cos(Math::PI * [over, 1.0].min))
        @floor + ((@peak - @floor) * cosine)
      end

      def call(event)
        event.session.adjust(lr: at(event.step))
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
    # So the ceiling has to be Ruby's. This reads the high-water mark
    # between steps and raises when it passes `bytes`, which the run's
    # process reports as a failure and the parent reads from the journal.
    # On unified memory the alternative is the machine stopping
    # (docs/plan.md section 15.22), which no rescue sees.
    #
    #   s.use(Torobi::Policies::MemoryGuard.new(10 * 1024**3), every: 10)
    #
    # **The peak, not what is held now.** Between steps a run holds its
    # parameters and its optimizer; what it passes through in the middle
    # of a step is several times that, and that is the number the machine
    # reacts to. Measured on a small model: 101 MB held between steps
    # against a 369 MB peak, and at ModernBERT sizes the gap is wider. A
    # guard reading what is held would have watched a step touch 14.5 GB
    # and said nothing.
    #
    # **It cannot stop a step in the middle.** No Ruby runs during one, so
    # this fires at the first hook after the overshoot. Whether a shape
    # fits at all is answered by running one step of it and reading the
    # peak (bench/overfit.rb), not by this.
    #
    # `every:` is worth using: reading costs a call through the runtime's
    # gate. The peak is a high-water mark and is not reset here, so a
    # sparser check still sees an overshoot between two of them.
    class MemoryGuard
      # The high-water mark when this last looked.
      attr_reader :seen

      def initialize(bytes, on_limit: nil)
        raise ArgumentError, "a guard needs a positive limit" unless bytes.positive?

        @bytes = Integer(bytes)
        @on_limit = on_limit
        @seen = 0
      end

      def call(event)
        @seen = Torobi::Memory.peak
        return if @seen <= @bytes

        event.session.observe(memory_at: event.step, peak: @seen, limit: @bytes)
        return @on_limit.call(event) if @on_limit

        raise Torobi::StepError,
              "this run has passed through #{@seen / (1024**2)} MB on the device by " \
              "step #{event.step}, past the #{@bytes / (1024**2)} MB it was given. " \
              "Stopping here rather than letting the machine find out"
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
