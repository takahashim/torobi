# frozen_string_literal: true

require "json"

module Torobi
  # A loaded graph and its state, held by the engine, driven from here.
  #
  # The division of labour is the one in docs/plan.md: the engine owns the
  # step, Ruby owns the data and the decisions between spans. Every step is
  # given its batch (section 5A.2) - the session holds parameters and
  # counters, never a dataset - and between spans the window is open, where
  # the listed operations apply (read metrics, turn knobs, copy tensors by
  # name). No tensor handle and no Ruby block ever crosses into a
  # computation.
  #
  # A session is a single conversation, so it is not thread-safe: one
  # session per thread, or a pool.
  #
  # What it raises: ArgumentError for a call that is malformed here,
  # Torobi::StepError for what the engine refused (the session survives),
  # and Torobi::EngineUnavailable before opening if the engine cannot run
  # at all (docs/plan.md section 5A.4).
  class Session
    # The update rule a session takes when none is named: plain SGD, which
    # has no state to restore and is the right default for a spike.
    DEFAULT_OPTIMIZER = { kind: :sgd, lr: 0.1 }.freeze

    # Opens a session over `config` (a GraphConfig) with initial `weights`,
    # shaped {params: {path => {shape:, data:}}}.
    #
    # `optimizer` is data, so that a journal and a checkpoint can record
    # exactly what ran: {kind: :adamw, lr: 1e-3, weight_decay: 0.01}.
    #
    # `journal:` records what happened, for the replay of docs/plan.md
    # section 8.6. Pass a Journal, or `io:` to have one made against this
    # config's provenance; without either, nothing is recorded and the
    # session is exactly as fast.
    def self.open(config, weights, optimizer: DEFAULT_OPTIMIZER, journal: nil, io: nil,
                  dataset: nil)
      Preflight.check!
      native = Native::Session.open(config.canonical_json, JSON.generate(weights),
                                    JSON.generate(optimizer))
      journal ||= Journal.new(Provenance.of(config, dataset:, extra: { "optimizer" => optimizer }),
                              io:) if io
      session = new(native, journal:)
      journal&.note(step: 0, event: "opened", optimizer: optimizer.transform_keys(&:to_s))
      return session unless block_given?

      begin
        yield session
      ensure
        # The block form owns the session's lifetime, so its device memory
        # goes when the block does, rather than when the GC gets to it.
        session.close
      end
    end

    def initialize(native, journal: nil)
      @native = native
      @journal = journal
      @hooks = Hooks.new(self)
    end

    # What this session is recording into, if anything.
    attr_reader :journal

    # Registers a hook. Sugar for the window and nothing more: a hook can
    # do what the window can do, because it fires in one (docs/plan.md
    # section 8.4).
    #
    #   s.on(:step, every: 200) { |e| bar.update(e.step, e.loss) }
    def on(event, every: 1, &block)
      @hooks.on(event, every:, &block)
      self
    end

    # Registers a policy object: anything answering #call(event). The
    # standard ones are in Torobi::Policies.
    def use(policy, event: :step, every: 1)
      @hooks.use(policy, event:, every:)
      self
    end

    # One step on `batch`, a Hash of input name => {shape:, data:}. Returns
    # the loss. The GVL is released for the step, so other Ruby threads
    # proceed.
    def step!(batch)
      loss = @native.run_step(Batch.pack(batch))
      @journal&.span(steps: 1, loss:, step: @native.step,
                     batches_digest: Provenance.digest_of(batch.keys.map(&:to_s)))
      @hooks.fire(:step, step: @native.step, loss:)
      loss
    end

    # Hands several batches to the engine at once, which runs them without
    # returning to Ruby in between.
    #
    # Faster by a fraction of a percent, and worth it only where a span is
    # short and known: it cannot be interrupted, and every batch is held in
    # memory (as Ruby objects, as packed strings, and again inside the
    # engine) before the first step runs. `run` is the normal way.
    def run_uninterruptible(batches, limit: 1024)
      packed = batches.first(limit + 1).map { |b| Batch.pack(b) }
      raise ArgumentError, "a span needs at least one batch" if packed.empty?
      if packed.size > limit
        raise ArgumentError,
              "run_uninterruptible takes at most #{limit} batches at a time; " \
              "use run for a span this long"
      end

      @native.run_steps(packed)
    end

    # A span: one step per batch, driven from here.
    #
    # One native call per step, not one per span. The boundary costs a
    # fraction of a percent of a step (bench/boundary.rb), and driving it
    # from Ruby is what makes the span interruptible: Ctrl-C, Thread#raise
    # and Timeout all land between steps rather than after the last one.
    # It also means an enumerable is consumed as it goes, so an endless one
    # trains rather than exhausting memory first.
    #
    # Returns the last loss. Yields it per step if a block is given, which
    # is where a caller records or decides.
    def run(batches)
      raise ArgumentError, "this session is closed" if closed?
      empty = true

      if @hooks.firing?
        raise Error, "a hook cannot run a span (hooks fire in the window, not around it)"
      end

      started = @native.step
      batches.each do |batch|
        empty = false
        loss = step!(batch)
        yield loss if block_given?
      end
      raise ArgumentError, "a span needs at least one batch" if empty

      @journal&.note(step: @native.step, event: "span", steps: @native.step - started,
                     loss: @native.loss)
      @hooks.fire(:span_end, step: @native.step, loss: @native.loss)
      @native.loss
    end

    # The same batch for `steps` steps. For fixed-data spikes and tests;
    # real training passes different batches.
    def repeat(batch, steps:)
      raise ArgumentError, "steps must be positive" unless steps.positive?

      packed = Batch.pack(batch)
      last = nil
      steps.times { last = @native.run_step(packed) }
      last
    end

    def step = @native.step
    def loss = @native.loss
    def lr = @native.lr

    # The seed the RNG runs from. Explicit state, not a global: the draws a
    # step makes are a function of this and the step count, which is what
    # lets a resumed run take the same ones (docs/plan.md section 11.1).
    def seed = @native.seed

    # Knobs take effect from the next step, and are recorded: a journal
    # that holds the decisions is what a replay applies (docs/plan.md 8.6).
    def adjust(lr: nil, seed: nil)
      turned = {}
      if lr
        @native.lr = lr
        turned[:lr] = lr
      end
      if seed
        @native.seed = seed
        turned[:seed] = seed
      end
      @journal&.adjust(step: @native.step, **turned) unless turned.empty?
      self
    end

    # Records what the window read. A policy that reads and then decides
    # has the reading as its input, so a deterministic rerun is held to it
    # (docs/plan.md section 8.5).
    def observe(**values)
      @journal&.observe(step: @native.step, **values)
      values
    end

    # Writes the run's state to `dir`, atomically: parameters, optimizer
    # slots, the step counts, and the digest of the description they belong
    # to. Returns the path. Interrupting it leaves no half-checkpoint.
    def checkpoint!(dir)
      path = @native.save(dir.to_s)
      @journal&.checkpoint(path:, step: @native.step)
      @hooks.fire(:checkpoint_written, step: @native.step, loss: @native.loss)
      path
    end

    # Restores what `checkpoint!` wrote. Refuses a checkpoint from another
    # description, another optimizer, or another shape, rather than
    # absorbing the difference.
    def restore(dir)
      @native.restore(dir.to_s)
      @journal&.note(step: @native.step, event: "restored", path: dir.to_s)
      self
    end

    # Releases the engine and its device memory. Idempotent; afterwards
    # every call refuses rather than pretending. Returns whether this call
    # was the one that closed it.
    def close
      # Read the counter before closing: afterwards the session refuses,
      # which is the point of closing.
      last_step = @native.closed? ? nil : @native.step
      closed = @native.close
      if closed
        @journal&.note(step: last_step, event: "closed")
        @journal&.close
      end
      closed
    end

    def closed? = @native.closed?

    # Whether the engine panicked under this session. A poisoned session
    # cannot be trusted to hold consistent state, so nothing more is
    # attempted with it; open a new one, from a checkpoint if there is one.
    def poisoned? = @native.poisoned?

    # Stops training what matches `pattern`: a path, or a prefix ending in
    # `*` ("student.layers.3.*"). Returns the paths that moved.
    #
    # Not a scalar knob (docs/plan.md 5A.1): it changes what autodiff
    # differentiates, so the optimizer's slots follow - kept for what
    # stays, dropped for what freezes, started at zero for what thaws. It
    # takes effect from the next step, like every knob, and is recorded.
    def freeze!(pattern)
      moved = @native.set_frozen(pattern.to_s, true)
      @journal&.adjust(step: @native.step, freeze: pattern.to_s, moved:) unless moved.empty?
      moved
    end

    # The reverse: gradual unfreezing is the usual reason.
    def unfreeze!(pattern)
      moved = @native.set_frozen(pattern.to_s, false)
      @journal&.adjust(step: @native.step, unfreeze: pattern.to_s, moved:) unless moved.empty?
      moved
    end

    # What is being trained right now, by qualified path.
    def trainable = @native.trainable

    # Writes one parameter from a copy: {shape:, data:, dtype:}. The
    # window's writing half, and recorded by digest rather than by value -
    # a journal names what was written without holding it.
    def put(path, tensor)
      packed = Batch.pack({ path.to_s => tensor }).fetch(path.to_s)
      @native.put(path.to_s, packed)
      @journal&.put(path: path.to_s, digest: Provenance.digest_of(packed[2]))
      self
    end

    def parameter_paths = @native.parameter_paths
    def input_names = @native.input_names

    # A copy of one parameter, as {shape:, data:}. Copies, never handles.
    def fetch(path)
      shape, data = @native.fetch(path.to_s)
      { shape:, data: }
    end

    # The gradients for `batch`, by parameter path. Does not update anything.
    def gradients(batch)
      @native.gradients(Batch.pack(batch)).to_h { |path, shape, data| [path, { shape:, data: }] }
    end
  end
end
