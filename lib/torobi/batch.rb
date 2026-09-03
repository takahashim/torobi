# frozen_string_literal: true

module Torobi
  # How a batch reaches the engine.
  #
  # A batch is written as {input_name => {shape:, data:}}, where data is a
  # flat Array of floats or an already-packed String. It crosses as
  # {name => [shape, packed]}: the shape stays a small integer array (it is
  # short and worth reading), the payload goes as native-endian f32 bytes.
  #
  # This is measured, not assumed. JSON serialization cost two thirds of a
  # step at 512 rows while the call boundary itself was noise; see
  # docs/plan.md section 5A.2.1 and bench/boundary.rb. Callers that already
  # hold packed data (from a file, a database, another library) can pass the
  # String and skip even this.
  module Batch
    FORMAT = "f*"

    module_function

    def pack(batch)
      batch.to_h do |name, tensor|
        shape = tensor.fetch(:shape) { tensor.fetch("shape") }
        data = tensor.fetch(:data) { tensor.fetch("data") }
        [name.to_s, [shape, data.is_a?(String) ? data : data.pack(FORMAT)]]
      end
    end

    # The inverse, for what comes back out.
    def unpack(packed)
      packed.unpack(FORMAT)
    end
  end
end
