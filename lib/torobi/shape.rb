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
      matmul take sdpa cross_entropy cast
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
      when :cross_entropy then cross_entropy(inputs, where:)
      when :cast then [inputs.first.shape, IR::Dtype.check!(attrs.fetch("dtype").to_sym, where:)]
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

    # One dimension may be -1 and takes whatever is left over, and any
    # number of leading 0s keep the dimensions the input already has.
    #
    # A symbolic dimension can only go into one of those two: nothing
    # else can stand for "however many rows this batch has". A -1 takes
    # one of them, which is enough while the batch is the only thing that
    # varies. A 0 is for when it is not: splitting the last dimension
    # into heads under a batch *and* a sequence that are both unknown
    # needs two dimensions carried through untouched, and "keep what is
    # there" is the only way to say that (docs/plan.md 15.63).
    def reshape(inputs, attrs, where:)
      from = inputs.first.shape
      target = attrs.fetch("shape")
      if target.any? { |d| !d.is_a?(Integer) || d < -1 }
        raise ConfigError,
              "#{where}: a shape is positive integers, at most one -1, and leading 0s " \
              "for the dimensions kept as they are, got #{target.inspect}"
      end

      keep = target.take_while(&:zero?).size
      if target.drop(keep).include?(0)
        raise ConfigError,
              "#{where}: a 0 keeps the dimension the input has there, so only the " \
              "leading ones can be kept (got #{target.inspect})"
      end
      if keep > from.size
        raise ConfigError,
              "#{where}: #{target.inspect} keeps #{keep} dimensions and " \
              "#{from.inspect} has #{from.size}"
      end

      shape, dtype = divide(inputs.first, from.drop(keep), target.drop(keep), where:, kept: keep)
      [from.take(keep) + shape, dtype]
    end

    # What is left of a reshape once the kept dimensions are set aside:
    # no 0s, so this is the whole of the old rule. The concrete part has
    # to be preserved exactly, which is a check worth having rather than
    # a formality.
    def divide(input, from, target, where:, kept: 0)
      after = kept.zero? ? "" : " (past the #{kept} it keeps)"
      inferred = target.count(-1)
      if inferred > 1
        raise ConfigError, "#{where}: only one dimension may be -1, got #{target.inspect}"
      end

      known = target.reject { |d| d == -1 }.reduce(1, :*)
      if from.include?(nil)
        unless inferred == 1
          raise ConfigError,
                "#{where}: #{from.inspect}#{after} has a symbolic dimension, so the " \
                "target must name a -1 or a 0 for it to go into (got #{target.inspect})"
        end
        concrete = from.compact.reduce(1, :*)
        unless concrete == known
          raise ConfigError,
                "#{where}: #{from.inspect}#{after} holds #{concrete} per symbolic step " \
                "and #{target.inspect} wants #{known}"
        end
        return [target.map { |d| d == -1 ? nil : d }, input.dtype]
      end

      total = from.reduce(1, :*)
      if inferred.zero?
        unless total == known
          raise ConfigError, "#{where}: #{from.inspect}#{after} holds #{total}, " \
                             "#{target.inspect} wants #{known}"
        end
        return [target, input.dtype]
      end
      unless known.positive? && (total % known).zero?
        raise ConfigError, "#{where}: #{from.inspect}#{after} holds #{total}, which does " \
                           "not divide into #{target.inspect}"
      end
      [target.map { |d| d == -1 ? total / known : d }, input.dtype]
    end

    def binary(inputs, where:)
      a, b = inputs
      [broadcast(a.shape, b.shape, where:), same_dtype(inputs, where:)]
    end

    # NumPy-style broadcasting, right-aligned: 1 stretches, nil is symbolic
    # and assumed to match, anything else must be equal.
    def broadcast(a, b, where:)
      rank = [a.size, b.size].max
      pad_a = ([1] * (rank - a.size)) + a
      pad_b = ([1] * (rank - b.size)) + b
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
      start = attrs.fetch("start")
      length = attrs.fetch("length")
      dim = shape[axis]
      raise ConfigError, "#{where}: cannot slice symbolic dimension #{axis}" if dim.nil?
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

      # flat_map rather than filter_map: a dimension that is kept can
      # itself be nil (the batch is symbolic), and dropping those was how
      # a reduction over [nil, seq, hidden] came back as [hidden].
      out = shape.each_with_index.flat_map do |dim, i|
        next [dim] unless axes.include?(i)

        keepdims ? [1] : []
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

    # Attention over [batch, heads, positions, head_dim].
    #
    # The head counts need not agree. Grouped-query attention gives
    # several query heads one key head, and the backend takes k and v
    # untiled, so what is required of them is that they divide rather than
    # match. Everything else does have to: one batch, keys and values in
    # step with each other, and a query as wide as a key.
    def sdpa(inputs, where:)
      given = inputs.first(3).map(&:shape)
      rank = given.first.size
      unless [3, 4].include?(rank) && given.all? { |shape| shape.size == rank }
        raise ConfigError,
              "#{where}: attention takes [batch, heads, positions, head_dim], or " \
              "the same without the heads; q is #{given[0].inspect}, k is " \
              "#{given[1].inspect}, v is #{given[2].inspect}"
      end

      # One head, written without saying so, is still one head.
      q, k, v = given.map { |shape| rank == 3 ? [shape[0], 1, shape[1], shape[2]] : shape }
      batch = unify_dim(q[0], unify_dim(k[0], v[0], where:, axis: 0), where:, axis: 0)
      unify_dim(k[1], v[1], where: "#{where} (k vs v heads)", axis: 1)
      unify_dim(k[2], v[2], where: "#{where} (k vs v positions)", axis: 2)
      unify_dim(q[3], k[3], where: "#{where} (q vs k head_dim)", axis: 3)
      grouped!(q[1], k[1], where:)
      out = rank == 3 ? [batch, q[2], v[3]] : [batch, q[1], q[2], v[3]]
      [out, same_dtype(inputs.first(3), where:)]
    end

    # Each key head serves the same number of query heads, so the one
    # count divides the other. Unknown either way is left alone, as
    # everywhere else a dimension is symbolic.
    def grouped!(heads, kv_heads, where:)
      return if heads.nil? || kv_heads.nil?
      return if kv_heads.positive? && (heads % kv_heads).zero?

      raise ConfigError,
            "#{where}: #{heads} query heads do not divide into #{kv_heads} key heads"
    end

    # Logits and the class each position wants, to the loss at each
    # position: [..., classes] and [...] to [...].
    #
    # The class axis is the last one and disappears; what is left is the
    # positions, which is what the two sides have to agree about.
    def cross_entropy(inputs, where:)
      logits, targets = inputs
      unless targets.dtype == :i32
        raise ConfigError,
              "#{where}: targets are class indices, so i32, got #{targets.dtype}"
      end
      if logits.shape.size < 2
        raise ConfigError,
              "#{where}: logits need a class axis after the positions, " \
              "got #{logits.shape.inspect}"
      end

      [unify(logits.shape[0..-2], targets.shape, where:), logits.dtype]
    end
  end
end
