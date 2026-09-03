# frozen_string_literal: true

module Torobi
  # Host-side data: bytes, a shape and a dtype. What crosses the boundary.
  #
  # It holds no operations, and that is deliberate (the same rule burn-rb's
  # TensorData follows). Reshaping, concatenating and slicing happen on the
  # device, where the graph says them; here there is only the value.
  #
  # Its reason for existing is the cost of *building* one. A batch used to
  # be written as a flat Ruby Array of Floats and packed on the way in,
  # which is fine for a spike and ruinous for a mask: an attention mask at
  # sequence 512 and batch 32 is 16.8 million Floats, built to be thrown
  # away a moment later. Most large tensors are not arbitrary numbers, they
  # are runs of the same one, and `runs` writes those without an Array at
  # all.
  #
  # Reading is the other direction and stays as it was: what comes back out
  # is numbers, because numbers are what a caller wants to look at.
  class TensorData
    # dtype => [pack directive, bytes per value].
    FORMATS = { f32: ["f", 4], i32: ["l<", 4] }.freeze

    attr_reader :dtype, :shape, :bytes

    # Already-packed bytes, native-endian. The form the boundary carries,
    # so this is the constructor the others end at.
    def initialize(shape, bytes, dtype: :f32)
      @dtype = dtype.to_sym
      @shape = shape.map { Integer(_1) }.freeze
      @bytes = bytes.b.freeze
      format!
      expected = size * FORMATS.fetch(@dtype).last
      unless @bytes.bytesize == expected
        raise ArgumentError,
              "#{@shape.inspect} of #{@dtype} wants #{expected} bytes, got #{@bytes.bytesize}"
      end
      freeze
    end

    class << self
      # From a flat Array of numbers. The plain way, and the expensive one:
      # every element exists in Ruby before any of it is packed.
      def from_a(shape, data, dtype: :f32)
        directive, = FORMATS.fetch(dtype.to_sym) { unknown!(dtype) }
        new(shape, data.pack("#{directive}*"), dtype:)
      end

      # From a nested Array, taking the shape from the nesting.
      #
      #   TensorData.nested([[1.0, 2.0], [3.0, 4.0]])   # shape [2, 2]
      def nested(array, dtype: :f32)
        shape = []
        node = array
        while node.is_a?(Array)
          shape << node.size
          node = node.first
        end
        from_a(shape, array.flatten, dtype:)
      end

      # One value, everywhere. No Array is built.
      def filled(shape, value, dtype: :f32)
        count = shape.inject(1, :*)
        new(shape, one(value, dtype) * count, dtype:)
      end

      # Runs of equal values, in order, covering the whole tensor.
      #
      # What a mask is: a row of a padding mask is "this many open, then
      # this many shut", and a sliding window row is three runs. Written as
      # String multiplication, so a 512x512 window costs three string
      # operations per row instead of 262,144 Floats.
      #
      #   TensorData.runs([2, 3], [[3, 0.0], [3, -1e9]])
      def runs(shape, runs, dtype: :f32)
        count = shape.inject(1, :*)
        given = runs.sum { |n, _| n }
        unless given == count
          raise ArgumentError, "#{shape.inspect} holds #{count} values, the runs give #{given}"
        end

        packed = +""
        runs.each { |n, value| packed << (one(value, dtype) * n) if n.positive? }
        new(shape, packed, dtype:)
      end

      private

      # One value as its bytes.
      def one(value, dtype)
        directive, = FORMATS.fetch(dtype.to_sym) { unknown!(dtype) }
        [value].pack(directive)
      end

      def unknown!(dtype)
        raise ArgumentError,
              "dtype #{dtype.inspect} does not cross the boundary " \
              "(#{FORMATS.keys.join(", ")})"
      end
    end

    # The {shape:, data:} pair the JSON paths speak (inline weights, and
    # `put` before it packs). Spells every number out, so it is a request,
    # not something the boundary does on its own.
    def to_h = { shape:, data: to_a }

    def rank = shape.size

    # How many values, not how many bytes.
    def size = shape.inject(1, :*)

    def bytesize = bytes.bytesize

    # The numbers, when a caller wants to look at them. Builds the Array
    # this class exists to avoid, so it is a question rather than a habit.
    def to_a
      directive, = FORMATS.fetch(dtype)
      bytes.unpack("#{directive}*")
    end

    # As a value: same shape, same dtype, same bytes.
    #
    # Ruby's == answers false for what it cannot compare rather than
    # raising, so `[t].include?(3)` and `t == nil` behave.
    def ==(other)
      other.is_a?(TensorData) && other.dtype == dtype &&
        other.shape == shape && other.bytes == bytes
    end
    alias eql? ==

    # Equal values must hash equally, and unequal ones may. Shape and dtype
    # sort them well enough; reading the whole buffer to key a Hash would
    # be paying for what eql? settles.
    def hash = [self.class, shape, dtype].hash

    def inspect = "#<Torobi::TensorData #{dtype}#{shape.inspect} #{bytesize} bytes>"
    alias to_s inspect

    private

    def format!
      FORMATS.fetch(dtype) do
        raise ArgumentError,
              "dtype #{dtype.inspect} does not cross the boundary " \
              "(#{FORMATS.keys.join(", ")})"
      end
    end
  end
end
