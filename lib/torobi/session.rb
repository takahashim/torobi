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
  class Session
    # Opens a session over `config` (a GraphConfig) with initial `weights`,
    # shaped {params: {path => {shape:, data:}}}.
    def self.open(config, weights)
      Preflight.check!
      session = new(Native::Session.open(config.canonical_json, JSON.generate(weights)))
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
      @native.run_step(JSON.generate(batch))
    end

    # A span: one step per batch. The batches are handed over before the GVL
    # is released, so the engine never asks Ruby for data mid-span. Takes an
    # array, or an enumerable of batches.
    def run(batches)
      batches = batches.to_a
      raise ArgumentError, "a span needs at least one batch" if batches.empty?

      @native.run_steps(JSON.generate(batches))
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

    def parameter_paths = @native.parameter_paths
    def input_names = @native.input_names

    # A copy of one parameter, as {shape:, data:}. Copies, never handles.
    def fetch(path)
      shape, data = @native.fetch(path.to_s)
      { shape:, data: }
    end

    # The gradients for `batch`, by parameter path. Does not update anything.
    def gradients(batch)
      @native.gradients(JSON.generate(batch)).to_h { |path, shape, data| [path, { shape:, data: }] }
    end
  end
end
