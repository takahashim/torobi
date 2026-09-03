# frozen_string_literal: true

module Torobi
  # Shape and dtype inference, one rule per manifest entry. Rules work on
  # shapes whose nil dimensions are symbolic: nil unifies with anything,
  # concrete dimensions must match exactly. Everything raises ConfigError
  # naming the node and op, so a shape mistake is reported at build time.
  module Shape
    # The rules this module implements, which must be exactly the set
    # `config/ops.yml` names. Declared rather than left implicit in the
    # dispatch below, so the two can be compared instead of remembered.
    RULES = %i[
      parameter same_as_input broadcast transpose reshape slice reduce
      matmul take sdpa
    ].freeze

    module_function

    # Returns [shape, dtype] for one emission.
    def infer(rule, inputs:, params:, attrs:, where:)
      case rule
      when :parameter then [params.fetch(0).shape, params.fetch(0).dtype]
      when :same_as_input then same_as_input(inputs, where:)
      when :broadcast then binary(inputs, where:)
      when :transpose then transpose(inputs, attrs, where:)
      when :reshape then reshape(inputs, attrs, where:)
      when :slice then slice(inputs, attrs, where:)
      when :reduce then reduce(inputs, attrs, where:)
      when :matmul then matmul(inputs, where:)
      when :take then take(inputs, where:)
      when :sdpa then sdpa(inputs, where:)
      else
        raise ConfigError, "#{where}: no shape rule #{rule.inspect} (ops.yml and Shape disagree)"
      end
    end

    def unify_dim(a, b, where:, axis:)
      return b if a.nil?
      return a if b.nil? || a == b

      raise ConfigError, "#{where}: dimension #{axis} mismatch (#{a} vs #{b})"
    end

    def unify(a, b, where:)
      unless a.size == b.size
        raise ConfigError, "#{where}: rank mismatch (#{a.inspect} vs #{b.inspect})"
      end

      a.zip(b).each_with_index.map { |(x, y), i| unify_dim(x, y, where:, axis: i) }
    end

    def same_dtype(inputs, where:)
      dtypes = inputs.map(&:dtype).uniq
      return dtypes.first if dtypes.size == 1

      raise ConfigError, "#{where}: mixed dtypes #{dtypes.join(", ")}"
    end

    def axis!(axis, rank, where:)
      normalized = axis.negative? ? axis + rank : axis
      unless (0...rank).cover?(normalized)
        raise ConfigError, "#{where}: axis #{axis} is out of range for rank #{rank}"
      end
      normalized
    end

    # --- rules ---

    def same_as_input(inputs, where:)
      [inputs.first.shape, same_dtype(inputs, where:)]
    end

    # One dimension may be -1 and takes whatever is left over.
    #
    # A symbolic dimension can only go there: nothing else can stand for
    # "however many rows this batch has". So a reshape of a symbolic input
    # must name a -1, and the concrete part has to be preserved exactly,
    # which is a check worth having rather than a formality.
    def reshape(inputs, attrs, where:)
      from = inputs.first.shape
      target = attrs.fetch("shape")
      inferred = target.count(-1)
      if inferred > 1
        raise ConfigError, "#{where}: only one dimension may be -1, got #{target.inspect}"
      end
      if target.any? { |d| !d.is_a?(Integer) || (d < 1 && d != -1) }
        raise ConfigError, "#{where}: a shape is positive integers and at most one -1, " \
                           "got #{target.inspect}"
      end

      known = target.reject { |d| d == -1 }.reduce(1, :*)
      if from.include?(nil)
        unless inferred == 1
          raise ConfigError,
                "#{where}: #{from.inspect} has a symbolic dimension, so the target " \
                "must name a -1 for it to go into (got #{target.inspect})"
        end
        concrete = from.compact.reduce(1, :*)
        unless concrete == known
          raise ConfigError,
                "#{where}: #{from.inspect} holds #{concrete} per symbolic step and " \
                "#{target.inspect} wants #{known}"
        end
        return [target.map { |d| d == -1 ? nil : d }, inputs.first.dtype]
      end

      total = from.reduce(1, :*)
      if inferred.zero?
        unless total == known
          raise ConfigError, "#{where}: #{from.inspect} holds #{total}, " \
                             "#{target.inspect} wants #{known}"
        end
        return [target, inputs.first.dtype]
      end
      unless known.positive? && (total % known).zero?
        raise ConfigError, "#{where}: #{from.inspect} holds #{total}, which does not " \
                           "divide into #{target.inspect}"
      end
      [target.map { |d| d == -1 ? total / known : d }, inputs.first.dtype]
    end

    def binary(inputs, where:)
      a, b = inputs
      [broadcast(a.shape, b.shape, where:), same_dtype(inputs, where:)]
    end

    # NumPy-style broadcasting, right-aligned: 1 stretches, nil is symbolic
    # and assumed to match, anything else must be equal.
    def broadcast(a, b, where:)
      rank = [a.size, b.size].max
      pad_a = [1] * (rank - a.size) + a
      pad_b = [1] * (rank - b.size) + b
      pad_a.zip(pad_b).each_with_index.map do |(x, y), i|
        next y if x == 1
        next x if y == 1

        unify_dim(x, y, where:, axis: i - rank)
      end
    end

    def transpose(inputs, attrs, where:)
      shape = inputs.first.shape
      axes = attrs.fetch("axes")
      unless axes.sort == (0...shape.size).to_a
        raise ConfigError,
              "#{where}: axes #{axes.inspect} is not a permutation of 0...#{shape.size}"
      end
      [axes.map { |i| shape[i] }, inputs.first.dtype]
    end

    def slice(inputs, attrs, where:)
      shape = inputs.first.shape.dup
      axis = axis!(attrs.fetch("axis"), shape.size, where:)
      start, length = attrs.fetch("start"), attrs.fetch("length")
      dim = shape[axis]
      if dim.nil?
        raise ConfigError, "#{where}: cannot slice symbolic dimension #{axis}"
      end
      unless length.positive? && start >= 0 && start + length <= dim
        raise ConfigError,
              "#{where}: slice #{start}...#{start + length} is out of 0...#{dim} on axis #{axis}"
      end
      shape[axis] = length
      [shape, inputs.first.dtype]
    end

    def reduce(inputs, attrs, where:)
      shape = inputs.first.shape
      keepdims = attrs.fetch("keepdims")
      axes = attrs.fetch("axes")&.map { |a| axis!(a, shape.size, where:) } || (0...shape.size).to_a
      if axes.uniq.size != axes.size
        raise ConfigError, "#{where}: duplicate reduction axes #{attrs["axes"].inspect}"
      end
      out = shape.each_with_index.filter_map do |dim, i|
        next dim unless axes.include?(i)

        keepdims ? 1 : nil
      end
      [out, inputs.first.dtype]
    end

    def matmul(inputs, where:)
      a, b = inputs.map(&:shape)
      if a.size < 2 || b.size < 2
        raise ConfigError, "#{where}: matmul needs rank >= 2 (#{a.inspect} @ #{b.inspect})"
      end

      unify_dim(a[-1], b[-2], where:, axis: -1)
      batch =
        if b.size == 2
          a[0...-2]
        else
          unify(a[0...-2], b[0...-2], where: "#{where} (batch dims)")
        end
      [batch + [a[-2], b[-1]], same_dtype(inputs, where:)]
    end

    def take(inputs, where:)
      table, indices = inputs
      unless indices.dtype == :i32
        raise ConfigError, "#{where}: take indices must be i32, got #{indices.dtype}"
      end
      if table.shape.size < 2
        raise ConfigError, "#{where}: take table needs rank >= 2, got #{table.shape.inspect}"
      end
      [indices.shape + table.shape[1..], table.dtype]
    end

    def sdpa(inputs, where:)
      q, k, v = inputs.first(3)
      unify(q.shape, k.shape, where: "#{where} (q vs k)")
      unify(q.shape, v.shape, where: "#{where} (q vs v)")
      [q.shape, same_dtype(inputs.first(3), where:)]
    end
  end
end
