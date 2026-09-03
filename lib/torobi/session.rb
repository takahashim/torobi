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
    def self.open(config, weights, optimizer: DEFAULT_OPTIMIZER)
      Preflight.check!
      native = Native::Session.open(config.canonical_json, JSON.generate(weights),
                                    JSON.generate(optimizer))
      session = new(native)
      return session unless block_given?

      begin
        yield session
      ensure
        # The block form owns the session's lifetime, so its device memory
        # goes when the block does, rather than when the GC gets to it.
        session.close
      end
    end

    def initialize(native)
      @native = native
    end

    # One step on `batch`, a Hash of input name => {shape:, data:}. Returns
    # the loss. The GVL is released for the step, so other Ruby threads
    # proceed.
    def step!(batch)
      @native.run_step(Batch.pack(batch))
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

      batches.each do |batch|
        empty = false
        loss = step!(batch)
        yield loss if block_given?
      end
      raise ArgumentError, "a span needs at least one batch" if empty

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

    # Knobs take effect from the next step.
    def adjust(lr: nil, seed: nil)
      @native.lr = lr if lr
      @native.seed = seed if seed
      self
    end

    # Writes the run's state to `dir`, atomically: parameters, optimizer
    # slots, the step counts, and the digest of the description they belong
    # to. Returns the path. Interrupting it leaves no half-checkpoint.
    def checkpoint!(dir)
      @native.save(dir.to_s)
    end

    # Restores what `checkpoint!` wrote. Refuses a checkpoint from another
    # description, another optimizer, or another shape, rather than
    # absorbing the difference.
    def restore(dir)
      @native.restore(dir.to_s)
      self
    end

    # Releases the engine and its device memory. Idempotent; afterwards
    # every call refuses rather than pretending. Returns whether this call
    # was the one that closed it.
    def close
      @native.close
    end

    def closed? = @native.closed?

    # Whether the engine panicked under this session. A poisoned session
    # cannot be trusted to hold consistent state, so nothing more is
    # attempted with it; open a new one, from a checkpoint if there is one.
    def poisoned? = @native.poisoned?

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
