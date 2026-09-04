# frozen_string_literal: true

module Torobi
  module Models
    # The masks that depend on a shape and on nothing else.
    #
    # A sliding window is the same bytes at every step of a run: what a
    # position may see is decided by the sequence length and the
    # architecture's width, and a batch has no say in it. So it is built
    # once per shape and handed out again, which is worth doing (0.21 ms
    # at sequence 192 and 0.62 ms at 512, against a step measured in
    # seconds) and worth bounding.
    #
    # **Bounded, because the shapes are no longer few.** A graph used to
    # be built for one length and a run saw one window. Now a batch is
    # padded to its own longest row (docs/plan.md 15.63), so a run sees a
    # different length most steps, and a window is seq*seq*4 bytes: a
    # megabyte each at 512. Holding every one a run ever meets is a leak
    # with a shape. What falls out is rebuilt, which costs what building
    # costs.
    #
    # Least recently used, because a run's lengths cluster: the sorted
    # batches of a mining pass walk through lengths in order and come back
    # to the long ones.
    class Windows
      LIMIT = 8

      def initialize(limit: LIMIT)
        raise ArgumentError, "a cache holds at least one" unless limit.positive?

        @limit = Integer(limit)
        @held = {}
      end

      def size = @held.size

      # What is held for `key`, or what the block builds.
      #
      # Ruby's Hash keeps insertion order, so re-inserting on a hit is
      # what makes the first key the least recently used one.
      def fetch(key)
        held = @held.delete(key)
        @held[key] = held || yield
        @held.delete(@held.keys.first) while @held.size > @limit
        @held.fetch(key)
      end
    end
  end
end
