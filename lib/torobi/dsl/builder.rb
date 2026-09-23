# frozen_string_literal: true

module Torobi
  module DSL
    # The `g` inside Torobi.graph: everything that adds to the graph.
    #
    # The division of labour with Handle: a handle has the one-value ops
    # the manifest marks `handle: true`, whose attributes are all keywords
    # (`x.gelu`, `x.softmax(axis: -1)`, `x.dropout(p: 0.1)`), its
    # operators, and the shape helpers built from them (`split_heads`).
    # Everything else is here: what declares parameters (`param` and the
    # layers in `Layers`), what joins several values (`matmul`, `sdpa`),
    # and the ops whose call is more than keywords on one value: the
    # reductions, whose axes default to all of them, and `cast`, which is
    # no node at all when the dtype is already the one asked for.
    class Builder
      include Layers

      # What a graph is built with when nothing is adapted: every parameter
      # trains as declared, and no linear gains anything. The same two
      # questions a `LoRA` answers, so the builder asks them without first
      # asking whether there is an adapter at all.
      module NoAdapter
        def self.trains?(_path) = true
        def self.wraps?(_path) = false
      end

      # `models` is the set an objective may read outputs from; a model
      # graph is built with none.
      def initialize(models: {})
        @models = models.to_h { |name, graph| [name.to_s, graph] }
        @inputs = []
        @parameters = []
        @nodes = []
        @outputs = {}
        @scopes = []
        @sharing = 0
        @labels = []
        @adapter = NoAdapter
      end

      # --- graph boundary ---

      # An input fed from the batch, by field name.
      def input(name, shape, dtype: :f32) = from_batch(name, shape, dtype:)

      # An input fed from the batch under a different field name than the
      # one it is known by here.
      def from_batch(name, shape, dtype: :f32, field: name)
        declare_input(name, shape, dtype, IR::Source.batch(field))
      end

      # An input fed from a model's named output. The shape and dtype come
      # from that model's declaration, so the two halves cannot disagree.
      def from_model(model, output, as: nil)
        graph = @models.fetch(model.to_s) do
          known = @models.keys.map(&:inspect).join(", ")
          raise ConfigError,
                "no model named #{model.to_s.inspect} here; this objective was " \
                "given #{known.empty? ? "none" : known}"
        end
        shape, dtype = graph.output_signature(output)
        declare_input(as || "#{model}.#{output}", shape, dtype,
                      IR::Source.model_output(model, output))
      end

      # Names one of this graph's outputs. An objective's loss and a model's
      # logits are both named this way.
      def output(name, handle)
        own!(handle, where: "output #{name.to_s.inspect}")
        name = name.to_s
        raise ConfigError, "output #{name.inspect} is declared twice" if @outputs.key?(name)

        @outputs[name] = handle.ref
        handle
      end

      def to_graph
        IR::Graph.new(inputs: @inputs, parameters: @parameters, nodes: @nodes,
                      outputs: @outputs)
      end

      # --- naming ---

      # Prefixes the paths of parameters created inside the block, so a
      # component used in a loop yields "layers.3.wqkv.weight" and friends.
      def scope(prefix)
        @scopes.push(prefix.to_s)
        yield
      ensure
        @scopes.pop
      end

      # --- parameters ---

      # A parameter this graph already declares, read again.
      #
      # Weight tying: a decoder whose output projection is its embedding
      # table transposed is **one** parameter read twice, not two that are
      # kept equal. Declaring it twice would be two, and a checkpoint
      # would hold two copies of the same numbers.
      #
      #   table = g.embedding(ids, vocab:, dim:, name: "embed")
      #   ...
      #   logits = g.matmul(h, g.parameter("embed.weight").transpose(axes: [1, 0]))
      #
      # The name is scoped like any other, so this reads the parameter of
      # the scope it is called in.
      def parameter(name)
        path = scoped(name)
        spec = declared(path)
        unless spec
          raise ConfigError,
                "no parameter #{path.inspect} is declared yet (this graph has " \
                "#{@parameters.map(&:path).inspect}); a shared parameter is read " \
                "after whatever declares it"
        end

        emit("parameter", params: [spec.id])
      end

      # The same weights, applied again.
      #
      # Inside this block a parameter that is already declared, in every
      # respect, is **read** rather than declared a second time. So a
      # component can be applied more than once and the applications are
      # one set of weights: a query tower and a document tower that are
      # the same tower, differentiated once and checkpointed once
      # (docs/plan.md 15.63).
      #
      #   %i[queries documents].each do |side|
      #     build = Build.new(seq: nil, dtype:, fields: "#{side}.")
      #     ModernBERT::Describe.new(g, config, build, encoder_prefix:)
      #                         .tower(side, pooling:, normalize:)
      #   end
      #
      # `label` names the values rather than the weights: the parameters
      # are one set with one set of names, and the nodes are "queries.x"
      # and "documents.x", because a value computed twice from different
      # rows is two values and a tap has to be able to ask for either.
      # Without one, the second application collides on the first's node
      # names, which is the error it should be.
      #
      # It is a block rather than the default because two declarations of
      # one path are otherwise the mistake they have always been: a
      # component built twice by accident would silently become one and
      # nothing would say so. Asking for the same path with a *different*
      # shape, dtype, initializer or trainability is refused here too.
      # Two things that are not the same cannot be the same weights.
      def sharing(label = nil)
        @sharing += 1
        @labels.push(label.to_s) if label
        yield
      ensure
        @labels.pop if label
        @sharing -= 1
      end

      def param(name, shape, init:, dtype: :f32, trainable: true)
        path = scoped(name)
        # Whether it trains is the adapter's to say (`LoRA#trains?`). Asked
        # before the sharing lookup, so that a second application is
        # compared with what the first actually declared: an adapted base
        # weight is frozen, and asking whether it is the same parameter
        # has to ask about the same thing.
        trainable &&= @adapter.trains?(path)
        if @sharing.positive? && (already = declared(path))
          return shared(already, shape:, dtype:, init:, trainable:)
        end

        spec = IR::ParameterSpec.new(id: @parameters.size, path:, shape:,
                                     dtype:, initializer: init, trainable:)
        @parameters << spec
        emit("parameter", params: [spec.id])
      end

      # Builds a graph with an adapter in scope, so that every linear it
      # names is trained through a pair of small matrices instead of
      # being moved itself (`Torobi::LoRA`).
      #
      # A block rather than a keyword on `linear`, because what is being
      # adapted is decided once, by whoever is doing the fine-tune, and a
      # model description should not have to be rewritten to be adapted:
      #
      #   Torobi.graph do |g|
      #     g.adapting(adapter) { ... the model ... }
      #   end
      #
      # `nil` adapts nothing, so a builder that always writes this reads
      # the same either way.
      def adapting(adapter)
        return yield if adapter.nil?
        raise ConfigError, "an adapter is already in scope" unless @adapter.equal?(NoAdapter)

        @adapter = adapter
        begin
          yield
        ensure
          @adapter = NoAdapter
        end
      end

      # --- primitive joins ---

      def matmul(a, b) = emit("matmul", inputs: [a, b])

      # Attention. `mask` is an additive mask the caller builds (padding,
      # a sliding window); `causal:` is the triangle every decoder wants,
      # which the backend has a mode for, so it is asked for by name
      # rather than handed over as megabytes of the same number.
      def sdpa(q, k, v, mask: nil, scale: nil, causal: false)
        if causal && mask
          raise ConfigError,
                "sdpa: causal: is a mask, so it does not go with another one. " \
                "Add what the mask says to the causal triangle, or drop it."
        end

        emit("sdpa", inputs: [q, k, v, mask].compact, attrs: { scale:, causal: })
      end

      def mean(x, axes: nil, keepdims: false)
        emit("mean", inputs: [x], attrs: { axes:, keepdims: })
      end

      def sum(x, axes: nil, keepdims: false)
        emit("sum", inputs: [x], attrs: { axes:, keepdims: })
      end

      # The same numbers in another precision.
      #
      #   g.cast(logits, :f32)
      #
      # Where a model is held in bf16 and its loss is read as f32, this is
      # the seam. Written down rather than inserted: a precision change
      # nobody asked for is how a run quietly stops matching what it is
      # held to.
      # Casting to what something already is is not a node: a graph
      # should not carry a step that does nothing, and this is what lets
      # a model be written once and built in either precision.
      def cast(x, dtype)
        dtype = dtype.to_sym
        return x if x.dtype == dtype

        emit("cast", inputs: [x], attrs: { dtype: dtype.to_s })
      end

      def max(x, axes: nil, keepdims: false)
        emit("max", inputs: [x], attrs: { axes:, keepdims: })
      end

      # The loss at each position of a classification: what was scored,
      # and the class each position should have had.
      #
      #   loss = g.mean(g.cross_entropy(logits, g.input(:targets, [nil, seq], dtype: :i32)))
      #
      # It reduces nothing. Which positions count (a padded one does not,
      # nor does the last, which has nothing after it to predict) and how
      # they are weighed is the objective's to say, and saying it is a
      # multiply and a sum.
      def cross_entropy(logits, targets)
        emit("cross_entropy", inputs: [logits, targets])
      end

      # --- core ---

      # Adds one node: checks ownership, arity and attributes against the
      # manifest, infers shape and dtype, and returns the handle.
      #
      # Public because a handle adds its nodes through it. A model
      # description does not reach it: `Models::Description::VOCABULARY`
      # is what a description may say, and `emit` is left out of it on
      # purpose, so a node no layer means cannot be put in a graph that
      # way.
      def emit(op, inputs: [], params: [], attrs: {}, name: nil)
        where = "node #{@nodes.size} (#{op})"
        spec = Ops.fetch(op, where:)
        inputs.each { |handle| own!(handle, where:) }
        attrs = attrs.transform_keys(&:to_s)
        spec.check!(inputs: inputs.size, params: params.size, attrs:, where:)

        shape, dtype = Shape.infer(spec.shape_rule, inputs:,
                                   params: params.map { |id| @parameters.fetch(id) },
                                   attrs:, where:)
        node = IR::NodeSpec.new(id: @nodes.size, op:, name: name && unique_name(name),
                                inputs: inputs.map(&:ref), parameters: params,
                                attributes: attrs, shape:, dtype:)
        @nodes << node
        Handle.new(builder: self, ref: IR::Ref.node(node.id), shape:, dtype:)
      end

      # What `Handle#named` asks of the graph: the node behind `handle`,
      # given a name no other value here has.
      def named(handle, label)
        own!(handle, where: "named #{label.to_s.inspect}")
        ref = handle.ref
        if ref.input?
          raise ConfigError,
                "only a computed value can be named, and #{ref} is an input"
        end

        node = @nodes[ref.id]
        raise ConfigError, "#{handle.ref} is already named #{node.name.inspect}" if node.name

        @nodes[ref.id] = node.with(name: unique_name(label))
        handle
      end

      private

      # A parameter an earlier application already declared, read again.
      #
      # The candidate is built rather than compared field by field, so
      # the two go through the same normalizing: what is compared is what
      # would have been declared.
      def shared(already, shape:, dtype:, init:, trainable:)
        wanted = IR::ParameterSpec.new(id: already.id, path: already.path, shape:, dtype:,
                                       initializer: init, trainable:)
        unless wanted == already
          differs = %i[shape dtype initializer trainable]
                    .reject { |f| already.send(f) == wanted.send(f) }
          said = differs.map { |f| "#{f} #{already.send(f).inspect} vs #{wanted.send(f).inspect}" }
          raise ConfigError,
                "#{already.path.inspect} is declared already and this asks for a " \
                "different #{said.join(", ")}. Sharing is one parameter read twice, so " \
                "the applications have to be the same model."
        end

        emit("parameter", params: [already.id])
      end

      def declared(path) = @parameters.find { |p| p.path == path }

      def unique_name(label)
        candidate = (@labels + [scoped(label)]).join(".")
        if @nodes.any? { |n| n.name == candidate }
          raise ConfigError, "two values are named #{candidate.inspect} in one graph"
        end

        candidate
      end

      def declare_input(name, shape, dtype, source)
        spec = IR::InputSpec.new(id: @inputs.size, name: name.to_s, shape:, dtype:, source:)
        @inputs << spec
        Handle.new(builder: self, ref: IR::Ref.input(spec.id), shape: spec.shape,
                   dtype: spec.dtype)
      end

      def scoped(name) = (@scopes + [name.to_s]).join(".")

      # The adapter in scope (`NoAdapter` when there is none), for a layer
      # that may be adapted.
      attr_reader :adapter

      def own!(handle, where:)
        unless handle.is_a?(Handle)
          raise ConfigError, "#{where}: expected a graph value, got #{handle.inspect}"
        end
        return if handle.builder.equal?(self)

        raise ConfigError,
              "#{where}: #{handle.inspect} belongs to a different graph; " \
              "values cannot cross graphs"
      end

      def concrete_last_dim!(x, where)
        own!(x, where:)
        x.shape.last or
          raise ConfigError, "#{where}: the last dimension of #{x.inspect} must be concrete"
      end
    end
  end
end
