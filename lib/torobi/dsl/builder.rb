# frozen_string_literal: true

module Torobi
  module DSL
    # The `g` inside Torobi.graph: everything that adds to the graph.
    #
    # The division of labour with Handle is one rule: what creates
    # parameters or joins several values is a method here (linear, sdpa,
    # matmul); what transforms one value is a method or operator on the
    # handle. Layers apply immediately: `g.linear(x, 512, name: "wo")`
    # creates the parameters and the nodes in one call.
    class Builder
      # `models` is the set an objective may read outputs from; a model
      # graph is built with none.
      # The adapter in scope, if a caller put one there (`adapting`).
      attr_reader :adapter

      def initialize(models: {})
        @models = models.to_h { |name, graph| [name.to_s, graph] }
        @inputs = []
        @parameters = []
        @nodes = []
        @outputs = {}
        @scopes = []
        @sharing = 0
        @labels = []
      end

      # --- graph boundary ---

      # An input fed from the batch, by field name.
      def input(name, shape, dtype: :f32)
        declare_input(name, shape, dtype, IR::Source.batch(name))
      end

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
        spec = @parameters.find { |p| p.path == path }
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
        # Inside an adapting block, what is trained is the adapter and
        # nothing else. Said here rather than at each parameter, because
        # a base model left trainable by an oversight is not a LoRA
        # fine-tune, and no pattern anybody writes later can undo it: the
        # window's `unfreeze!` moves within what the graph declared
        # trainable, so this is the declaration that matters.
        #
        # Before the sharing lookup, so that a second application is
        # compared with what the first actually declared: an adapted base
        # weight is frozen, and asking whether it is the same parameter
        # has to ask about the same thing.
        trainable &&= @adapter.adapted?(path) if @adapter
        if @sharing.positive? && (already = @parameters.find { |p| p.path == path })
          return shared(already, shape:, dtype:, init:, trainable:)
        end

        spec = IR::ParameterSpec.new(id: @parameters.size, path:, shape:,
                                     dtype:, initializer: init, trainable:)
        @parameters << spec
        emit("parameter", params: [spec.id])
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

      # --- layers: parameters plus their application, in one call ---

      def linear(x, d_out, name:, bias: true)
        label = name
        d_in = concrete_last_dim!(x, "linear #{scoped(name).inspect}")
        # PyTorch layout [d_out, d_in], so pretrained checkpoints map 1:1.
        w = param("#{name}.weight", [d_out, d_in], dtype: x.dtype,
                  init: { "type" => "kaiming_uniform" })
        wt = emit("transpose", inputs: [w], attrs: { axes: [1, 0] })
        y = matmul(x, wt)
        y += param("#{name}.bias", [d_out], dtype: x.dtype, init: { "type" => "zeros" }) if bias
        y += low_rank(x, d_in, d_out, name:) if @adapter&.wraps?(scoped(name))
        self.name(label, y)
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
        raise ConfigError, "an adapter is already in scope" if @adapter

        @adapter = adapter
        begin
          yield
        ensure
          @adapter = nil
        end
      end

      # The adapter's own arithmetic: `x` through a narrow matrix and back
      # out to the width the linear has, scaled.
      #
      # `B` starts at zero, so this contributes nothing until something
      # has trained it. That is what makes an adapted model start as the
      # model it adapts.
      def low_rank(x, d_in, d_out, name:)
        rank = @adapter.rank
        a = param("#{name}.lora_A.weight", [rank, d_in], dtype: x.dtype,
                                                         init: { "type" => "kaiming_uniform" })
        b = param("#{name}.lora_B.weight", [d_out, rank], dtype: x.dtype,
                                                          init: { "type" => "zeros" })
        down = matmul(x, emit("transpose", inputs: [a], attrs: { axes: [1, 0] }))
        up = matmul(down, emit("transpose", inputs: [b], attrs: { axes: [1, 0] }))
        up * @adapter.scale
      end

      # The table, and the lookup into it.
      #
      # `dtype:` is where a model's precision is decided: everything
      # downstream takes its dtype from what it is given (`linear` and the
      # norms use `x.dtype`), so a bf16 table makes a bf16 model.
      def embedding(ids, vocab:, dim:, name:, dtype: :f32)
        table = param("#{name}.weight", [vocab, dim], dtype:,
                      init: { "type" => "normal", "std" => 0.02 })
        emit("take", inputs: [table, ids])
      end

      def layer_norm(x, name:, bias: false, eps: 1.0e-5)
        d = concrete_last_dim!(x, "layer_norm #{scoped(name).inspect}")
        w = param("#{name}.weight", [d], dtype: x.dtype, init: { "type" => "ones" })
        inputs = [x, w]
        inputs << param("#{name}.bias", [d], dtype: x.dtype, init: { "type" => "zeros" }) if bias
        emit("layer_norm", inputs:, attrs: { eps: })
      end

      # `offset:` is added to the learned weight before it scales.
      #
      # Gemma stores its norms as `w` and applies `(1 + w)`, so its
      # weights sit around zero where everyone else's sit around one.
      # The op is the same op; what differs is what is handed to it, and
      # that is a fact about the checkpoint rather than about norms.
      def rms_norm(x, name:, eps: 1.0e-5, offset: 0.0)
        d = concrete_last_dim!(x, "rms_norm #{scoped(name).inspect}")
        w = param("#{name}.weight", [d], dtype: x.dtype,
                                         init: { "type" => offset.zero? ? "ones" : "zeros" })
        w += offset unless offset.zero?
        emit("rms_norm", inputs: [x, w], attrs: { eps: })
      end

      # GeGLU as ModernBERT uses it: one projection producing act and gate,
      # gelu on the act half, a projection back down.
      def geglu(x, d_hidden, name:)
        d_in = concrete_last_dim!(x, "geglu #{scoped(name).inspect}")
        scope(name) do
          a, gate = linear(x, d_hidden * 2, name: "wi", bias: false).split(2, axis: -1)
          linear(a.gelu * gate, d_in, name: "wo", bias: false)
        end
      end

      # --- objective vocabulary ---

      def mse(a, b) = mean((a - b).square)

      # Values whose gradient does not flow back. A teacher's output goes
      # through this (docs/plan.md section 5A.3).
      def stop_gradient(x) = emit("stop_gradient", inputs: [x])

      # Inverted dropout, drawing from the session's RNG state. `p` is the
      # rate dropped; 0 is the identity.
      def dropout(x, p) = emit("dropout", inputs: [x], attrs: { p: })

      # --- core ---

      # Adds one node: checks ownership, arity and attributes against the
      # manifest, infers shape and dtype, and returns the handle.
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

      # Names a value, so a tap can ask for it later (docs/plan.md 6.4 and
      # 8.3). The name is taken under the scopes in force, so the same
      # component in a loop yields "layers.0.attn", "layers.1.attn".
      #
      #   h = g.name("attn", g.sdpa(q, k, v))
      def name(label, handle)
        own!(handle, where: "name #{label.to_s.inspect}")
        kind, id = IR::Ref.parse(handle.ref)
        if kind != :node
          raise ConfigError, "only a computed value can be named, and #{handle.ref} is an input"
        end

        node = @nodes[id]
        raise ConfigError, "#{handle.ref} is already named #{node.name.inspect}" if node.name

        @nodes[id] = IR::NodeSpec.new(
          id: node.id, op: node.op, name: unique_name(label), inputs: node.inputs,
          parameters: node.parameters, attributes: node.attributes,
          shape: node.shape, dtype: node.dtype
        )
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
