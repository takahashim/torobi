# frozen_string_literal: true

module Torobi
  module IR
    # The dtypes the IR speaks. Deliberately few: what the target models
    # need, not what MLX offers.
    module Dtype
      ALL = %i[f32 bf16 i32 bool].freeze

      module_function

      def check!(dtype, where:)
        return dtype if ALL.include?(dtype)

        raise ConfigError, "#{where}: unknown dtype #{dtype.inspect}; expected one of #{ALL.join(", ")}"
      end
    end
  end
end
