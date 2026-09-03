# frozen_string_literal: true

module Torobi
  # A contrastive batch larger than the machine can hold.
  #
  # A loss whose negatives are the other rows wants the largest batch it
  # can get, and the memory a batch needs is set by the activations kept
  # for the backward pass. The gradient cache (Gao et al., 2021) separates
  # the two: encode the parts one at a time **without** keeping activations,
  # work out the loss over all of the representations at once, then encode
  # each part again and back-propagate it with the loss's gradient by its
  # representations as the seed. Memory is then set by one part, and the
  # contrastive signal by the whole.
  #
  # Nothing here is new machinery. It is four things Torobi already has,
  # in the order the algorithm puts them:
  #
  #   1. `evaluate` with a tap, which is a forward that keeps nothing
  #   2. `field_gradients`, which is dL/d(representations)
  #   3. an objective of `sum(representations * seed)`, whose gradient by
  #      the parameters is exactly the product of the seed with the
  #      Jacobian, which is what back-propagating a seed means
  #   4. `accumulate` / `apply!`, which sums the parts and steps once
  #
  # What it asks of the graphs:
  #
  #   encoder   a named node holding the representations (`tap:`), and a
  #             batch field for the seed (`seed:`) whose objective is
  #             `sum(node * seed)`. Declare the seed with a symbolic
  #             shape (`[nil, nil]`): on the pass that only wants the tap
  #             a single zero stands in for it and broadcasts, so this
  #             needs to know neither how wide a representation is nor how
  #             many rows a part holds.
  #   loss      a batch field for the representations (`into:`), and the
  #             real loss as its objective. It is opened once and never
  #             steps; nothing in it moves.
  #
  # **The graph must draw no randomness.** Each part is encoded twice and
  # the two must agree; dropout would make them different passes, and the
  # seed from the first would be the wrong answer for the second. ModernBERT
  # as `Torobi::Models` builds it has no random ops.
  #
  #   cache = Torobi::GradCache.new(encoder, loss: scores,
  #                                 tap: "m.embeddings", into: :vectors,
  #                                 seed: :cotangent)
  #   cache.step(parts)   # => the loss over the whole batch
  #
  # `parts` is an Enumerable of batches. Their rows keep their order: the
  # representations are stacked in the order the parts arrive, and the
  # loss graph sees them that way.
  class GradCache
    # The seed for a pass that is not using it. One zero, which broadcasts
    # over whatever the representations turn out to be, so the first pass
    # needs no notion of their shape.
    NOTHING = TensorData.filled([1, 1], 0.0)

    # How many parts the last step was made of, and how many rows they
    # held. For a caller watching what its splitting actually does.
    attr_reader :parts, :rows

    def initialize(session, loss:, tap:, into:, seed:)
      @session = session
      @loss = loss
      @tap = tap.to_s
      @into = into
      @seed = seed
      @parts = 0
      @rows = 0
    end

    # One optimizer step over every part, and the loss it was taken on.
    #
    # The parts are walked three times: once to encode them, once for the
    # loss and its gradient, once to back-propagate. They are therefore
    # kept, so an Enumerable that can only be read once is read into an
    # Array here rather than surprising the caller half way through.
    def step(parts)
      parts = parts.to_a
      raise ArgumentError, "a step needs at least one part" if parts.empty?

      cached = parts.map { |part| encode(part) }
      @parts = parts.size
      @rows = cached.sum { |data| data.shape.first }

      whole = stack(cached)
      loss = @loss.evaluate(@into => whole)
      seeds = @loss.field_gradients({ @into => whole }, of: [@into]).fetch(@into.to_s)

      at = 0
      parts.zip(cached) do |part, data|
        height = data.shape.first
        @session.accumulate(part.merge(@seed => slice(seeds, at, height)))
        at += height
      end
      @session.apply!
      loss
    end

    private

    # One part's representations, from a forward that keeps nothing.
    #
    # The seed is zeros here: the objective is a product with it, so the
    # loss is zero and nothing is being asked of it. What is wanted is the
    # tap, and a tap reports whatever pass it watches.
    def encode(part)
      @session.tap(@tap, stat: :full)
      @session.evaluate(part.merge(@seed => NOTHING))
      @session.tapped.fetch(@tap)
    ensure
      @session.untap(@tap)
    end

    # The parts' representations as one, in the order they arrived.
    #
    # Bytes, so this is a concatenation rather than a conversion: rows are
    # contiguous and equal width, which makes stacking them along the first
    # axis the same as joining what they are written as.
    def stack(parts)
      width = parts.first.shape.last
      unless parts.all? { |p| p.shape.last == width }
        raise Error, "the parts hold representations of different widths"
      end

      TensorData.new([parts.sum { |p| p.shape.first }, width],
                     parts.map(&:bytes).join, dtype: parts.first.dtype)
    end

    # `height` rows of `data`, starting at `at`. The same fact the other
    # way round.
    def slice(data, at, height)
      stride = data.shape.last * 4
      TensorData.new([height, data.shape.last],
                     data.bytes.byteslice(at * stride, height * stride), dtype: data.dtype)
    end
  end
end
