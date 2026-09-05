# frozen_string_literal: true

module Torobi
  module DSL
    # A symbolic value inside one graph under construction: it knows which
    # builder owns it, what it references, and its inferred shape and dtype.
    # Transformations live here (unary ops, operators, split); anything that
    # creates parameters or joins several values lives on the builder.
    class Handle
      attr_reader :builder, :ref, :shape, :dtype

      def initialize(builder:, ref:, shape:, dtype:)
        @builder = builder
        @ref = ref
        @shape = shape
        @dtype = dtype
      end

      { :+ => "add", :- => "sub", :* => "mul", :/ => "div" }.each do |operator, op|
        define_method(operator) do |other|
          case other
          when Handle then builder.emit(op, inputs: [self, other])
          when Numeric then builder.emit("#{op}_scalar", inputs: [self], attrs: { value: other })
          else raise ConfigError, "cannot #{operator} a #{other.class} to a graph value"
          end
        end
      end

      # Methods for every manifest op marked `handle: true`, e.g. `gelu`,
      # `softmax(axis: -1)`, `rope(theta: 10_000)`.
      Ops.handle_ops.each do |spec|
        define_method(spec.name) do |**attrs|
          builder.emit(spec.name, inputs: [self], attrs:)
        end
      end

      # Lets a number lead: `1.0 - x`, `2.0 * x`.
      #
      # Ruby asks the right-hand side to coerce when the left does not know
      # what to do with it, and the pair it returns is sent the operator
      # again. Returning `[Scalar.new(number), self]` puts the number in
      # something that knows how to fold itself into the graph, which is
      # what makes a loss read the way it is written on paper.
      def coerce(other)
        unless other.is_a?(Numeric)
          raise ConfigError, "cannot combine a #{other.class} with a graph value"
        end

        [Scalar.new(self, other), self]
      end

      # A number waiting to meet a graph value, and nothing else.
      #
      # It is what `coerce` returns, never something to build: it has no
      # meaning apart from the operator Ruby is about to send it. Kept
      # under Handle for that reason.
      class Scalar
        def initialize(handle, value)
          @handle = handle
          @value = value
          freeze
        end

        # Addition and multiplication do not care which side they are on.
        def +(other) = other + @value
        def *(other) = other * @value

        # Subtraction and division do, so these are the reversed forms:
        # `1.0 - x` is `-(x - 1.0)`, and `1.0 / x` needs the graph to
        # divide the other way round.
        def -(other) = (other - @value) * -1.0
        def /(other) = @handle.builder.emit("div", inputs: [filled(other), other])

        def inspect = "#<Torobi::DSL::Handle::Scalar #{@value}>"
        alias to_s inspect

        private

        # The number as a graph value of the same shape, so `div` has two
        # sides. Made from the handle rather than beside it, because a
        # scalar has no shape of its own.
        def filled(other)
          ones = @handle.builder.emit("mul_scalar", inputs: [other], attrs: { "value" => 0.0 })
          @handle.builder.emit("add_scalar", inputs: [ones], attrs: { "value" => @value })
        end
      end

      # The same numbers, read as `count` heads instead of one wide row:
      # [batch, seq, width] -> [batch, count, seq, width / count].
      #
      #   q = g.linear(x, heads * dim, name: "q_proj").split_heads(heads)
      #
      # Two nodes, and both of them are already here: the width is
      # divided up (`reshape`) and the heads are moved in front of the
      # sequence, which is the layout attention runs in (`transpose`).
      # Named because `[0, 2, 1, 3]` says where the axes went and not what
      # was meant, and every description that wrote it by hand wrote a
      # comment underneath saying this sentence.
      #
      # The leading two dimensions are kept as they are, so a graph built
      # for no particular batch or sequence splits heads the same way one
      # built for both does (docs/plan.md 15.63).
      def split_heads(count)
        count = Integer(count)
        raise ConfigError, "split_heads: #{count} heads is not a count" unless count.positive?
        unless shape.size == 3
          raise ConfigError,
                "split_heads: expected [batch, seq, width], got #{dtype}#{shape.inspect}"
        end

        width = shape.last or
          raise ConfigError, "split_heads: the width being divided must be concrete"
        unless (width % count).zero?
          raise ConfigError, "split_heads: #{width} does not divide into #{count} heads"
        end

        builder.emit("reshape", inputs: [self], attrs: { shape: [0, 0, count, width / count] })
               .transpose(axes: [0, 2, 1, 3])
      end

      # The inverse: [batch, count, seq, dim] -> [batch, seq, count * dim].
      #
      #   linear(attended.merge_heads, hidden, name: "o_proj")
      #
      # The width it lands on is the heads times what each of them holds,
      # which this reads off its own shape. A model whose attention is
      # wider or narrower than its hidden state (Gemma 3 is: four heads of
      # 256 over a hidden state of 640) therefore needs to say nothing
      # here, and the projection that follows says the rest.
      def merge_heads
        unless shape.size == 4
          raise ConfigError,
                "merge_heads: expected [batch, heads, seq, dim], got #{dtype}#{shape.inspect}"
        end

        heads, dim = shape.values_at(1, 3)
        unless heads && dim
          raise ConfigError,
                "merge_heads: the heads and their width must both be concrete, " \
                "and this is #{shape.inspect}"
        end

        transpose(axes: [0, 2, 1, 3])
          .then { |h| builder.emit("reshape", inputs: [h], attrs: { shape: [0, 0, heads * dim] }) }
      end

      # Splits into `count` equal parts along `axis`, as slice nodes.
      def split(count, axis: -1)
        normalized = Shape.axis!(axis, shape.size, where: "split")
        dim = shape[normalized]
        raise ConfigError, "split: cannot split symbolic dimension #{axis}" if dim.nil?
        unless (dim % count).zero?
          raise ConfigError, "split: dimension #{dim} does not divide into #{count} parts"
        end

        length = dim / count
        Array.new(count) do |i|
          builder.emit("slice", inputs: [self],
                                attrs: { axis: normalized, start: i * length, length: })
        end
      end

      def inspect
        "#<Torobi::DSL::Handle #{ref} #{dtype}#{shape.inspect}>"
      end
    end
  end
end
