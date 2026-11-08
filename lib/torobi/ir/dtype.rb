# frozen_string_literal: true

module Torobi
  module IR
    # The dtypes the IR speaks. Deliberately few: what the target models
    # need, not what MLX offers.
    #
    # `u32` is not a number a graph computes with. It is the packed form
    # of a quantized weight (`Torobi::Quantization`, docs/qlora.md): 32 /
    # bits values in one word, read only by `quantized_matmul`, and never
    # the result of a `cast`.
    module Dtype
      ALL = %i[f32 bf16 i32 bool u32].freeze

      # The dtypes a value can be cast to: everything but the packed one.
      CASTABLE = (ALL - %i[u32]).freeze

      module_function

      def check!(dtype, where:)
        return dtype if ALL.include?(dtype)

        raise ConfigError,
              "#{where}: unknown dtype #{dtype.inspect}; expected one of #{ALL.join(", ")}"
      end
    end
  end
end
