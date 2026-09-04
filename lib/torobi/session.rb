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
    #
    # This is the one a Ruby caller gets. The engine has its own for its
    # command-line tool, and Ruby never reads it: every open from here
    # states the rule and the seed, so the two cannot disagree about a
    # run that actually happened.
    DEFAULT_OPTIMIZER = { kind: :sgd, lr: 0.1 }.freeze

    # Where a run's randomness starts when the caller names no seed. Zero
    # rather than something drawn: a run that did not choose its seed
    # should still be the same run twice.
    DEFAULT_SEED = 0

    # Opens a session over `config` (a GraphConfig).
    #
    # Where the parameters come from is said by keyword, because there is
    # more than one answer:
    #
    #   weights:       values, {params: {path => {shape:, data:}}}. For
    #                  spikes and tests. Not for a real model: 130M
    #                  parameters as JSON numbers is gigabytes of text.
    #   weights_file:  one safetensors file holding every model's
    #                  parameters under their qualified paths, which is how
    #                  a checkpoint writes them, so a run can start from
    #                  what another run reached.
    #   pretrained:    {model_name => path}, one published checkpoint per
    #                  model. A model published on its own does not know
    #                  what this run will call it, so its tensors carry no
    #                  model prefix; naming a file per model is what lets
    #                  ruri-v3 or a reranker be imported with no renaming,
    #                  and it is the shape a distillation wants anyway:
    #
    #                    pretrained: { student: "ruri-130m/model.safetensors",
    #                                  teacher: "reranker-310m/model.safetensors" }
    #
    # `fresh:` goes with `pretrained:` and names the parameters that are
    # meant to be in no file, built from the initializers the graph
    # declares instead. A classification head put on a published encoder
    # is the case: it exists in no checkpoint. Patterns, as freezing uses
    # them. Everything they do not name must be in its file, so a mistyped
    # parameter is still an error rather than a quietly random one:
    #
    #   pretrained: { student: "ruri-130m/model.safetensors" },
    #   fresh: ["student.head.*", "student.classifier.*"]
    #
    # A file in another precision is converted: importing is starting
    # somewhere, not resuming.
    #
    # A third will follow: the GraphConfig already declares an initializer
    # per parameter, so a run that starts from those and a seed needs no
    # weights at all.
    #
    # `optimizer` is data, so that a journal and a checkpoint can record
    # exactly what ran: {kind: :adamw, lr: 1e-3, weight_decay: 0.01}.
    #
    # `seed:` is where every draw this run makes comes from: the parameters
    # a `fresh:` pattern builds here, and the ops that draw at each step.
    # It is the axis an experiment varies to ask whether a result was the
    # method or the luck, so it belongs at open, where the first draw
    # happens. `adjust(seed:)` moves it afterwards, which is a different
    # question (docs/plan.md section 11.1).
    #
    # `journal:` records what happened, for the replay of docs/plan.md
    # section 8.6. Pass a Journal, or `io:` to have one made against this
    # config's provenance; without either, nothing is recorded and the
    # session is exactly as fast.
    #
    # `dataset:` is anything about the data that a later reader would need
    # and cannot work out from here. It goes verbatim into the journal's
    # provenance and into every checkpoint's record, and nothing
    # interprets it.
    #
    # **What belongs there above all is the tokenizer.** Torobi is handed
    # token ids and does not know what produced them; a run whose record
    # does not say cannot be repeated, however exactly its parameters are
    # restored. Convention rather than a requirement, because a spike has
    # nothing to say and should not be made to say it:
    #
    #   dataset: { name: "jaqket-train", rows: 12_000,
    #              tokenizer: "cl-nagoya/ruri-v3-130m",
    #              tokenizer_revision: "e3114c6...", max_seq_length: 512 }
    def self.open(config, weights: nil, weights_file: nil, pretrained: nil, fresh: [],
                  optimizer: DEFAULT_OPTIMIZER, seed: DEFAULT_SEED,
                  journal: nil, io: nil, dataset: nil)
      source = Weights.of(weights:, weights_file:, pretrained:, fresh:)
      Preflight.check!
      native = source.open(config, optimizer, seed)
      # Gathered whether or not anything is journalling: a checkpoint
      # records it too, so that what it holds can be identified later
      # (docs/plan.md section 11.2).
      provenance = Provenance.of(config, dataset:,
                                 extra: { "optimizer" => optimizer, "seed" => seed })
      journal ||= Journal.new(provenance, io:) if io
      session = new(native, journal:, provenance:, loss: config.loss?)
      journal&.note(step: 0, event: "opened", seed:,
                    optimizer: optimizer.transform_keys(&:to_s))
      return session unless block_given?

      begin
        yield session
      ensure
        # The block form owns the session's lifetime, so its device memory
        # goes when the block does, rather than when the GC gets to it.
        session.close
      end
    end

    def initialize(native, journal: nil, provenance: nil, loss: true)
      @native = native
      @journal = journal
      @provenance = provenance
      @loss = loss
      @hooks = Hooks.new(self)
    end

    # Whether this run has a loss. A config that trains nothing and
    # declares no objective has none, and everything that would move
    # something refuses rather than inventing one (`GraphConfig#loss?`).
    def loss? = @loss

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
    def on(event, every: 1, &)
      @hooks.on(event, every:, &)
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
      needs_loss!("a step")
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

    # Adds one batch's gradients to what is waiting, and returns its loss.
    # No step is taken and no counter moves.
    #
    # What this is for is a batch too large to hold: train it as several
    # that fit, then `apply!`. The gradients of a sum are the sum of the
    # gradients, so the step lands where one over the whole batch would
    # have. It is also what a gradient cache is built on, where the parts
    # are re-run with a cotangent the caller worked out in between.
    #
    # **The weighting is the caller's.** A loss that is a mean over its
    # rows means each part carries the share of the rows it holds; the
    # engine sums what it is given and does not guess what the loss was a
    # mean of.
    #
    #   parts.each { |part| s.accumulate(part) }
    #   s.apply!
    #
    # Freezing and `checkpoint!` refuse while parts are waiting, because
    # one moves what a gradient is for and the other does not hold them.
    def accumulate(batch)
      needs_loss!("accumulating")
      atomically do
        loss = @native.accumulate(Batch.pack(batch))
        @journal&.observe(step: @native.step, accumulated: @native.accumulated, loss:)
        loss
      end
    end

    # Takes the step the accumulated gradients ask for, and returns the
    # mean of the losses they came from. Refuses when nothing is waiting.
    def apply!
      parts = @native.accumulated
      loss = atomically do
        value = @native.apply
        @journal&.span(steps: 1, loss: value, step: @native.step, parts:,
                       batches_digest: nil)
        value
      end
      @hooks.fire(:step, step: @native.step, loss:)
      loss
    end

    # How many parts are waiting for a step.
    def accumulated = @native.accumulated

    # Throws away what is waiting. Returns how many parts went.
    def discard = @native.discard

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
      needs_loss!("evaluate")
      @native.evaluate(Batch.pack(batch))
    end

    # What the models produce for `batch`, by qualified output name.
    #
    # The forward `evaluate` runs, stopping before the objective: what
    # comes back is what a model produced rather than the loss over it.
    # This is how a fine-tuned model is used to look at something, which
    # is what it was fine-tuned for.
    #
    #   s.forward(batch)                         # => {"m.embedding" => TensorData}
    #   s.forward(batch, outputs: ["m.logits"])  # only that one
    #
    # No gradients, no randomness (dropout stands aside), and nothing
    # about the run moves: not the parameters, not the counters, not what
    # `loss` reports. Not journalled, for the same reason `evaluate` is
    # not: nothing happened that a replay would apply.
    #
    # The batch needs what the models read and no more. A run that trains
    # against labels does not need them to be asked what it thinks.
    #
    # The taps report it, as they report any pass, except any on the
    # objective: it does not run here, so it has nothing to say.
    #
    # Every value is a TensorData, whatever its shape. What an output is
    # for is its shape, and a caller that wants the number out of a
    # scalar can say `.to_a.first`.
    def forward(batch = nil, outputs: nil, **fields)
      # `forward(x: ids)` is the batch written the way it reads, and in
      # Ruby 3 it is keywords, so the batch arrives as nothing at all.
      # Caught here because what Ruby says about it ("unknown keyword",
      # or an argument count) is true and no help.
      if batch.nil? || !fields.empty?
        raise ArgumentError,
              "forward takes its batch as one argument: forward({x: ...}), not " \
              "forward(x: ...). The braces are what keep it from being keywords."
      end

      wanted = Array(outputs).map(&:to_s)
      values_of(@native.forward(Batch.pack(batch), wanted))
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

    # Writes one model as a checkpoint somebody else can load: its
    # parameters as fp32 safetensors, and the files that say what they are.
    #
    # `from:` is the published model this run started from. Everything
    # except its weights is still true of the result (the tokenizer, the
    # vocabulary, the sequence length, how the pooling was configured), so
    # it is copied rather than reinvented. **Without it the export holds no
    # tokenizer, and sentence-transformers cannot load a directory that has
    # none**, whatever else is in it. Fine-tuning always has a source; a
    # model trained from nothing does not, and has to be given a tokenizer
    # by hand.
    #
    # `model:` is which model of the GraphConfig to write, needed only when
    # the run has more than one. The model's own name is stripped from each
    # parameter path, so a model declared with `encoder_prefix:` keeps that
    # prefix and the keys match the published layout.
    #
    # `pooling:` (:cls or :mean) and `pooling_dim:` override what the
    # source said, for an export that pools differently from what it
    # started as. Left out, the source's own pooling configuration travels
    # unchanged, including keys this code knows nothing about.
    #
    #   s.export_model!("out/ruri-ft", from: "ruri-v3-130m")
    #   s.export_model!("out/mean", from: "ruri-v3-130m", pooling: :mean)
    #
    # This is not a checkpoint: no optimizer state, no RNG, no counters.
    # It is the shape a run produces for serving, and `checkpoint!` is the
    # shape it produces for itself.
    def export_model!(dir, from: nil, model: nil, pooling: nil, pooling_dim: nil)
      model ||= sole_model
      merged!(model)
      atomically do
        written = @native.export_model(model.to_s, dir.to_s)
        carried = Export.publish(dir.to_s, from:, pooling:, pooling_dim:)
        # Last, so that a record of an export is a record of one that is
        # on disk whole.
        @journal&.note(step: @native.step, event: "exported", model: model.to_s,
                       paths: written.map(&:last), carried:)
        written.to_h
      end
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

    # Every name a tap could ask for, qualified by the model it is in
    # ("student.hidden"). Settled at open, like `parameter_paths`.
    def node_names = @native.node_names

    # Every output `forward` could ask for, qualified by the model that
    # declares it ("student.embedding"). Settled at open, like the rest of
    # the names.
    def output_names = @native.output_names

    # What the last step's taps saw, by tap name. A scalar tap (a loss,
    # say) comes back as a Float, since a caller compares it or writes it
    # down; anything with a shape comes back as a TensorData.
    #
    # Read-only, so it needs no journal entry of its own; `observe` is
    # how a decision made on one gets recorded.
    def tapped
      values_of(@native.tapped).transform_values do |data|
        data.shape.empty? ? data.to_a.first : data
      end
    end

    # What the run is made of: every parameter by qualified path, every
    # batch field it reads, and every name a tap may ask for.
    #
    # Settled when the session opened and answered from there, so a
    # watching thread gets them while a span is in flight, and they still
    # answer after `close`. What is being trained right now is not one of
    # these: freezing moves it, so `trainable` asks the engine.
    def parameter_paths = @native.parameter_paths
    def input_names = @native.input_names

    # A copy of one parameter, as a TensorData. Copies, never handles.
    #
    # The copy is bytes rather than numbers, and stays that way until a
    # caller asks for `to_a`: ruri-v3's embedding table is 200 MB of bytes
    # and was 600 MB of resident memory as a Ruby Array, for a value most
    # callers save or compare rather than read.
    def fetch(path)
      dtype, shape, bytes = @native.fetch(path.to_s)
      TensorData.new(shape, bytes, dtype: dtype.to_sym)
    end

    # The gradients of the loss with respect to named batch fields, by
    # field name. Does not update anything and does not differentiate the
    # parameters.
    #
    # What this is for is a loss over values computed elsewhere. A gradient
    # cache encodes its parts without gradients, works out the loss over
    # all of the representations at once, and needs that loss
    # differentiated **by the representations** so each part can be re-run
    # with the answer as its seed:
    #
    #   d = loss_session.field_gradients(batch, of: %i[queries documents])
    #
    # The fields must be in the batch, and the answer has their shapes.
    def field_gradients(batch, of:)
      needs_loss!("a gradient")
      names = Array(of).map(&:to_s)
      values_of(@native.field_gradients(Batch.pack(batch), names))
    end

    # The gradients for `batch`, by parameter path. Does not update anything.
    def gradients(batch)
      needs_loss!("a gradient")
      values_of(@native.gradients(Batch.pack(batch)))
    end

    private

    # What the engine answers with, as values by name.
    #
    # Four calls come back the same way, because the boundary carries
    # one shape of answer: the name, the dtype and shape readable, the
    # payload packed. How to read that is one fact and lives once.
    def values_of(answered)
      answered.to_h do |name, dtype, shape, bytes|
        [name, TensorData.new(shape, bytes, dtype: dtype.to_sym)]
      end
    end

    # What a run without a loss cannot do.
    #
    # Refused here rather than at the boundary, because what is missing is
    # a fact about the config: nothing is trained and no output is a
    # scalar, so there is nothing to differentiate or report. `forward` is
    # what such a run is for.
    def needs_loss!(what)
      return if @loss

      raise ConfigError,
            "#{what} needs a loss, and this run has none: nothing is trained and " \
            "no model output is an f32 scalar. That is what a config opened for " \
            "inference looks like, and `forward` is what it can do."
    end

    # That there is a model here to export, rather than a model and an
    # adapter beside it.
    #
    # LoRA leaves the weight alone and trains a pair of small matrices
    # next to it, so an adapted run holds the base weights it started
    # with and never moved. Exporting writes every parameter, which would
    # be those weights plus `lora_A` and `lora_B` tensors nothing else
    # knows: **a directory that looks trained and is not.** Merging them
    # (W + scale * BA, and the adapter dropped) is what makes one, and it
    # is not built yet. Until it is, this refuses rather than writes
    # something misleading; the checkpoint holds both halves, so nothing
    # is lost by waiting.
    def merged!(model)
      adapted = @native.parameter_paths.select do |path|
        path.start_with?("#{model}.") && LoRA.adapted?(path)
      end
      return if adapted.empty?

      raise ConfigError,
            "#{model.inspect} was trained through an adapter (#{adapted.size} " \
            "parameters like #{adapted.first.inspect}), and its base weights have " \
            "not moved. Writing them would be a model that looks trained and is " \
            "not. Merging the adapter into the weights is not implemented; the " \
            "checkpoint holds both halves."
    end

    # Which model to export when the run holds only one.
    #
    # From the parameter paths, which the engine settled at open and
    # answers without waiting for a step: a model name is the first
    # segment of a qualified path.
    def sole_model
      names = @native.parameter_paths.map { |path| path.split(".").first }.uniq
      return names.first if names.size == 1

      raise ArgumentError, "model: is required (this run has #{names.inspect})"
    end

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
