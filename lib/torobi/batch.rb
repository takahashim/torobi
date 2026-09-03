# frozen_string_literal: true

module Torobi
  # How a batch reaches the engine.
  #
  # A batch is written as {input_name => {shape:, data:, dtype:}}, where
  # data is a flat Array of numbers or an already-packed String. It crosses
  # as {name => [dtype, shape, packed]}: the dtype and shape stay readable,
  # the payload goes as native-endian 4-byte values.
  #
  # The dtype travels because a graph may declare an i32 input, which is
  # what an embedding reads; a boundary that assumed f32 could not carry
  # one. The packing itself is measured rather than assumed: JSON cost two
  # thirds of a step at 512 rows while the call boundary was noise (see
  # docs/plan.md section 5A.2.1 and bench/boundary.rb). Callers that
  # already hold packed data can pass the String and skip even this.
  module Batch
    # dtype => the pack directive for a flat Array of that type.
    FORMATS = { "f32" => "f*", "i32" => "l<*" }.freeze
    DEFAULT_DTYPE = "f32"

    module_function

    def pack(batch)
      batch.to_h do |name, tensor|
        dtype = (tensor[:dtype] || tensor["dtype"] || DEFAULT_DTYPE).to_s
        format = FORMATS.fetch(dtype) do
          raise ArgumentError,
                "input #{name}: dtype #{dtype.inspect} does not cross the boundary " \
                "(#{FORMATS.keys.join(", ")})"
        end
        shape = tensor.fetch(:shape) { tensor.fetch("shape") }
        data = tensor.fetch(:data) { tensor.fetch("data") }
        [name.to_s, [dtype, shape, data.is_a?(String) ? data : data.pack(format)]]
      end
    end

    # The inverse, for what comes back out.
    def unpack(packed, dtype: DEFAULT_DTYPE)
      packed.unpack(FORMATS.fetch(dtype.to_s))
    end
  end
end
