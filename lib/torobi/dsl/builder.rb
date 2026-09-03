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
      def initialize(models: {})
        @models = models.to_h { |name, graph| [name.to_s, graph] }
        @inputs = []
        @parameters = []
        @nodes = []
        @outputs = {}
        @scopes = []
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
        if @outputs.key?(name)
          raise ConfigError, "output #{name.inspect} is declared twice"
        end

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

      def param(name, shape, dtype: :f32, init:, trainable: true)
        spec = IR::ParameterSpec.new(id: @parameters.size, path: scoped(name), shape:,
                                     dtype:, initializer: init, trainable:)
        @parameters << spec
        emit("parameter", params: [spec.id])
      end

      # --- primitive joins ---

      def matmul(a, b) = emit("matmul", inputs: [a, b])

      def sdpa(q, k, v, mask: nil, scale: nil)
        emit("sdpa", inputs: [q, k, v, mask].compact, attrs: { scale: })
      end

      def mean(x, axes: nil, keepdims: false)
        emit("mean", inputs: [x], attrs: { axes:, keepdims: })
      end

      def sum(x, axes: nil, keepdims: false)
        emit("sum", inputs: [x], attrs: { axes:, keepdims: })
      end

      # --- layers: parameters plus their application, in one call ---

      def linear(x, d_out, name:, bias: true)
        d_in = concrete_last_dim!(x, "linear #{scoped(name).inspect}")
        # PyTorch layout [d_out, d_in], so pretrained checkpoints map 1:1.
        w = param("#{name}.weight", [d_out, d_in], dtype: x.dtype,
                  init: { "type" => "kaiming_uniform" })
        wt = emit("transpose", inputs: [w], attrs: { axes: [1, 0] })
        y = matmul(x, wt)
        return y unless bias

        y + param("#{name}.bias", [d_out], dtype: x.dtype, init: { "type" => "zeros" })
      end

      def embedding(ids, vocab:, dim:, name:)
        table = param("#{name}.weight", [vocab, dim],
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

      def rms_norm(x, name:, eps: 1.0e-5)
        d = concrete_last_dim!(x, "rms_norm #{scoped(name).inspect}")
        w = param("#{name}.weight", [d], dtype: x.dtype, init: { "type" => "ones" })
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
      def emit(op, inputs: [], params: [], attrs: {})
        where = "node #{@nodes.size} (#{op})"
        spec = Ops.fetch(op, where:)
        inputs.each { |handle| own!(handle, where:) }
        attrs = attrs.transform_keys(&:to_s)
        spec.check!(inputs: inputs.size, params: params.size, attrs:, where:)

        shape, dtype = Shape.infer(spec.shape_rule, inputs:,
                                   params: params.map { |id| @parameters.fetch(id) },
                                   attrs:, where:)
        node = IR::NodeSpec.new(id: @nodes.size, op:, inputs: inputs.map(&:ref),
                                parameters: params, attributes: attrs, shape:, dtype:)
        @nodes << node
        Handle.new(builder: self, ref: IR::Ref.node(node.id), shape:, dtype:)
      end

      private

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
