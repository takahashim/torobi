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

    # Opens a session over `config` (a GraphConfig).
    #
    # Where the parameters come from is said by keyword, because there is
    # more than one answer:
    #
    #   weights:       values, {params: {path => {shape:, data:}}}. For
    #                  spikes and tests. Not for a real model: 130M
    #                  parameters as JSON numbers is gigabytes of text.
    #   weights_file:  a safetensors file, read by the engine. Its tensors
    #                  are named the way the graph names them, which is
    #                  also how a checkpoint writes them, so a run can
    #                  start from what another run reached. A file in
    #                  another precision is converted: importing is
    #                  starting somewhere, not resuming.
    #
    # A third will follow: the GraphConfig already declares an initializer
    # per parameter, so a run that starts from those and a seed needs no
    # weights at all.
    #
    # `optimizer` is data, so that a journal and a checkpoint can record
    # exactly what ran: {kind: :adamw, lr: 1e-3, weight_decay: 0.01}.
    #
    # `journal:` records what happened, for the replay of docs/plan.md
    # section 8.6. Pass a Journal, or `io:` to have one made against this
    # config's provenance; without either, nothing is recorded and the
    # session is exactly as fast.
    def self.open(config, weights: nil, weights_file: nil, optimizer: DEFAULT_OPTIMIZER,
                  journal: nil, io: nil, dataset: nil)
      if weights.nil? == weights_file.nil?
        raise ArgumentError,
              "a session needs its parameters, from exactly one place: " \
              "weights: {params: {path => {shape:, data:}}}, or " \
              "weights_file: a safetensors path"
      end

      Preflight.check!
      native =
        if weights_file
          Native::Session.open_from_file(config.canonical_json, weights_file.to_s,
                                         JSON.generate(optimizer))
        else
          Native::Session.open(config.canonical_json, JSON.generate(weights),
                               JSON.generate(optimizer))
        end
      # Gathered whether or not anything is journalling: a checkpoint
      # records it too, so that what it holds can be identified later
      # (docs/plan.md section 11.2).
      provenance = Provenance.of(config, dataset:, extra: { "optimizer" => optimizer })
      journal ||= Journal.new(provenance, io:) if io
      session = new(native, journal:, provenance:)
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

    def initialize(native, journal: nil, provenance: nil)
      @native = native
      @journal = journal
      @provenance = provenance
      @hooks = Hooks.new(self)
    end

    # What this session is recording into, if anything.
    attr_reader :journal

    # What this run is: the config's digest, the dataset it was opened
    # against, the build. Written into every checkpoint.
    attr_reader :provenance

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
      packed = Batch.pack(batch)
      loss = atomically do
        value = @native.run_step(packed)
        @journal&.span(steps: 1, loss: value, step: @native.step,
                       batches_digest: Provenance.digest_of(batch.keys.map(&:to_s)))
        value
      end
      # Outside the fence: a hook runs the caller's own code, and that
      # should be interruptible like any other Ruby.
      @hooks.fire(:step, step: @native.step, loss:)
      loss
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

    # The loss for `batch` without taking a step.
    #
    # What a validation set is read with. It costs a forward rather than a
    # forward and a backward, and it runs with the random ops standing
    # aside, so the number is the model's rather than a sample of it.
    # Nothing about the run moves: not the parameters, not the counters, not
    # what `loss` reports.
    #
    #   validation.sum { |b| s.evaluate(b) } / validation.size
    #
    # Not journalled. An evaluation changes nothing, so a replay has nothing
    # to apply; `observe` is how a decision made on one gets recorded.
    def evaluate(batch)
      @native.evaluate(Batch.pack(batch))
    end

    # The same batch for `steps` steps. For fixed-data spikes and tests;
    # real training passes different batches.
    #
    # One `step!` per step, not a shortcut past it: a step that the journal
    # does not hold is a step a replay cannot follow, and hooks fire here
    # like anywhere else.
    def repeat(batch, steps:)
      raise ArgumentError, "steps must be positive" unless steps.positive?

      last = nil
      steps.times { last = step!(batch) }
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
      atomically do
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
      end
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
    # slots, the step counts, the description they belong to, and this
    # run's provenance. Returns the path. Interrupting it leaves no
    # half-checkpoint.
    #
    # `at:` is where in the data the run is - epoch, batch, whatever a
    # sampler needs to carry on. Torobi is handed batches and never fetches
    # them (docs/plan.md section 5A.2), so it cannot work this out; it
    # records what it is told, verbatim, and hands it back on restore.
    #
    #   s.checkpoint!("run/000200", at: { epoch: 2, batch: 1_400 })
    def checkpoint!(dir, at: nil)
      record = { "provenance" => @provenance, "position" => stringify(at) }.compact
      path = atomically do
        written = @native.save(dir.to_s, JSON.generate(record))
        @journal&.checkpoint(path: written, step: @native.step)
        written
      end
      @hooks.fire(:checkpoint_written, step: @native.step, loss: @native.loss)
      path
    end

    # Restores what `checkpoint!` wrote. Refuses a checkpoint from another
    # description, another optimizer, or another shape, rather than
    # absorbing the difference.
    #
    # Returns the position that was recorded with it (or nil), so that
    # whoever owns the data can put its sampler back where it was. The
    # whole record, provenance included, is `Torobi::Checkpoint.manifest`.
    def restore(dir)
      recorded = atomically do
        state = JSON.parse(@native.restore(dir.to_s))
        @journal&.note(step: @native.step, event: "restored", path: dir.to_s)
        state
      end
      recorded.is_a?(Hash) ? recorded["position"] : nil
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
      atomically do
        moved = @native.set_frozen(pattern.to_s, true)
        @journal&.adjust(step: @native.step, freeze: pattern.to_s, moved:) unless moved.empty?
        moved
      end
    end

    # The reverse: gradual unfreezing is the usual reason.
    def unfreeze!(pattern)
      atomically do
        moved = @native.set_frozen(pattern.to_s, false)
        @journal&.adjust(step: @native.step, unfreeze: pattern.to_s, moved:) unless moved.empty?
        moved
      end
    end

    # What is being trained right now, by qualified path.
    def trainable = @native.trainable

    # Writes one parameter from a copy: {shape:, data:, dtype:}. The
    # window's writing half, and recorded by digest rather than by value -
    # a journal names what was written without holding it.
    def put(path, tensor)
      packed = Batch.pack({ path.to_s => tensor }).fetch(path.to_s)
      atomically do
        @native.put(path.to_s, packed)
        @journal&.put(path: path.to_s, digest: Provenance.digest_of(packed[2]))
      end
      self
    end

    # Watches a named value inside the graph, read-only (docs/plan.md
    # section 8.3, capability B+). `stat` reduces it on the device:
    #
    #   :norm    the L2 norm, which is how an activation is usually watched
    #   :mean    its mean
    #   :extent  its smallest and largest
    #   :full    the tensor itself, which is for debugging and costs it
    #
    # A tap adds an output; it never changes one. What it costs is a
    # forward pass per step (a traced pass produces values that belong to
    # the trace, not to the caller), so taps are opt-in and a standing one
    # should reduce.
    def tap(name, stat: :norm)
      @native.tap_node(name.to_s, stat.to_s)
      self
    end

    def untap(name) = @native.untap(name.to_s)

    # What is being watched.
    def taps = @native.taps

    # Every name a tap could ask for.
    def node_names = @native.node_names

    # What the last step's taps saw, by name. Read-only, so it needs no
    # journal entry of its own; `observe` is how a decision made on one
    # gets recorded.
    def tapped
      @native.tapped.to_h do |name, shape, data|
        [name, shape.empty? ? data.first : { shape:, data: }]
      end
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

    private

    # Runs an engine change and the journal's record of it as one, so an
    # asynchronous interrupt cannot land between them.
    #
    # Without this the engine takes a step and Ruby is interrupted before
    # writing it down, leaving a record one operation behind what actually
    # happened. A replay follows the record, so that difference is not
    # cosmetic (notes/SESSION_CONCURRENCY_SPEC.md section 10).
    #
    # What it costs in responsiveness is bounded by one step, which is the
    # granularity a span is interruptible at anyway.
    def atomically(&)
      Thread.handle_interrupt(Object => :never, &)
    end

    # A position travels as JSON, so its keys are strings by the time it
    # comes back. Convert on the way in, so that what a caller reads after
    # a restore is what it wrote.
    def stringify(value)
      case value
      when nil then nil
      when Hash then value.to_h { |k, v| [k.to_s, stringify(v)] }
      when Array then value.map { |v| stringify(v) }
      when Symbol then value.to_s
      else value
      end
    end
  end
end
