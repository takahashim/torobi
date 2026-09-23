# frozen_string_literal: true

module Torobi
  # One step's inputs: a TensorData per batch field.
  #
  # A batch is written as {name => tensor}, where a tensor is a
  # `Torobi::TensorData` or the plain {shape:, data:, dtype:} Hash that
  # predates it (`TensorData.from`). Either way it is a TensorData once it
  # is in here, so what crosses the boundary is one kind of value, checked
  # once: that its bytes are as many as its shape and dtype say.
  #
  # Prefer TensorData for anything large. A Hash of numbers means every
  # element exists in Ruby before any of it is packed, which for a mask is
  # millions of Floats built to be thrown away; TensorData can be built
  # from runs instead.
  #
  # The packing itself is measured rather than assumed: JSON cost two
  # thirds of a step at 512 rows while the call boundary was noise (see
  # docs/plan.md section 5A.2.1 and bench/boundary.rb).
  class Batch
    # `batch` may already be a Batch, which is returned as it is.
    def self.of(batch) = batch.is_a?(Batch) ? batch : new(batch)

    def initialize(fields)
      @fields = fields.to_h do |name, tensor|
        [name.to_s, TensorData.from(tensor, where: "input #{name}")]
      end.freeze
      freeze
    end

    # The fields, by name.
    attr_reader :fields

    def names = @fields.keys

    def fetch(name) = @fields.fetch(name.to_s)

    # What the engine is handed: {name => [dtype, shape, bytes]}.
    def to_native = @fields.transform_values(&:to_native)
  end
end
