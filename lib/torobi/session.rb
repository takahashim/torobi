# frozen_string_literal: true

require "json"

module Torobi
  # A loaded graph and its state, held by the engine, driven from here.
  #
  # The division of labour is the one in docs/plan.md: the engine owns the
  # step, Ruby owns the ordering and the decisions between spans. A span is
  # one `run` call; between spans the window is open, and what may happen in
  # it is the listed set (read metrics, turn knobs, copy tensors by name).
  # No tensor handle and no Ruby block ever crosses into a computation.
  #
  # A session is a single conversation, so it is not thread-safe: one
  # session per thread, or a pool.
  class Session
    # Opens a session over `config` (a GraphConfig) with `bindings`, whose
    # shape is {inputs: {name => {shape:, data:}}, params: {path => ...}}.
    def self.open(config, bindings)
      session = new(Native::Session.open(config.canonical_json, JSON.generate(bindings)))
      return session unless block_given?

      yield session
    end

    def initialize(native)
      @native = native
    end

    # Runs `steps` steps and returns the loss at the end. The GVL is
    # released for the duration, so other Ruby threads proceed.
    def run(steps: 1)
      raise ArgumentError, "steps must be positive" unless steps.positive?

      @native.run_steps(steps)
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

    # A copy of one parameter, as {shape:, data:}. Copies, never handles.
    def fetch(path)
      shape, data = @native.fetch(path.to_s)
      { shape:, data: }
    end

    # The gradients for the bound inputs, by parameter path.
    def gradients
      @native.gradients.to_h { |path, shape, data| [path, { shape:, data: }] }
    end
  end
end
