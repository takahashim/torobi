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

      yield session
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

    # A span: one step per batch. The batches are handed over before the GVL
    # is released, so the engine never asks Ruby for data mid-span. Takes an
    # array, or an enumerable of batches.
    def run(batches)
      packed = batches.map { |b| Batch.pack(b) }
      raise ArgumentError, "a span needs at least one batch" if packed.empty?

      @native.run_steps(packed)
    end

    # The same batch for `steps` steps. For fixed-data spikes and tests;
    # real training passes different batches.
    def repeat(batch, steps:)
      raise ArgumentError, "steps must be positive" unless steps.positive?

      run([batch] * steps)
    end

    def step = @native.step
    def loss = @native.loss
    def lr = @native.lr

    # Knobs take effect from the next step.
    def adjust(lr: nil)
      @native.lr = lr if lr
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
