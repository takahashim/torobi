# frozen_string_literal: true

module Torobi
  # Deep freezing for the IR values. GraphConfig is a serializable artifact
  # with a digest; nothing may mutate it after construction, or the digest
  # would lie.
  module Freeze
    module_function

    # Freezes +value+ and everything reachable from it. Returns +value+.
    def deep(value)
      case value
      when Hash
        value.each { |k, v| deep(k); deep(v) }
      when Array
        value.each { |v| deep(v) }
      end
      value.freeze
    end
  end
end
