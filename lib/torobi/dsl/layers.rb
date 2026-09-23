# frozen_string_literal: true

module Torobi
  module DSL
    # The layers a model is written in: parameters plus their application,
    # in one call (`g.linear(x, 512, name: "wo")` declares the weights and
    # adds the nodes).
    #
    # Kept apart from `Builder` because they are the part of the DSL that
    # grows: each is written in the builder's own vocabulary (`param`,
    # `matmul`, `scope`, `emit`, and the handle's methods) and needs
    # nothing else from it, so adding a layer is adding a method here and
    # not reopening the class that owns the graph. Included into `Builder`,
    # so a description still says `g.linear`.
    module Layers
      def linear(x, d_out, name:, bias: true)
        d_in = concrete_last_dim!(x, "linear #{scoped(name).inspect}")
        # PyTorch layout [d_out, d_in], so pretrained checkpoints map 1:1.
        w = param("#{name}.weight", [d_out, d_in], dtype: x.dtype,
                  init: { "type" => "kaiming_uniform" })
        y = matmul(x, w.transpose(axes: [1, 0]))
        y += param("#{name}.bias", [d_out], dtype: x.dtype, init: { "type" => "zeros" }) if bias
        y += adapter.contribution(self, x, d_in:, d_out:, name:) if adapter.wraps?(scoped(name))
        y.named(name)
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
        scope name do
          a, gate = linear(x, d_hidden * 2, name: "wi", bias: false).split(2, axis: -1)
          linear(a.gelu * gate, d_in, name: "wo", bias: false)
        end
      end

      # The mean squared difference: the loss a regression or a distillation
      # reads.
      def mse(a, b) = mean((a - b).square)
    end
  end
end
