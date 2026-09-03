# frozen_string_literal: true

module Torobi
  # What the device is holding.
  #
  # Ruby's GC does not see device memory: a session is one small object
  # here and a hundred megabytes there, so a long run needs numbers it can
  # watch and a limit it can set (docs/plan.md section 11.3).
  #
  # These are process-wide, because MLX's allocator is. Two sessions share
  # one pool, which is a reason to run one training session per process
  # rather than a reason to hide the sharing.
  module Memory
    module_function

    # Bytes: what is in use, what the allocator is keeping, the high-water
    # mark, and the cap (0 meaning none).
    def report = Native.memory.transform_keys(&:to_sym)

    def active = report.fetch(:active)
    def cache = report.fetch(:cache)
    def peak = report.fetch(:peak)
    def limit = report.fetch(:limit)

    # Frees what the allocator holds but is not using. Returns how many
    # bytes went.
    def clear_cache!
      before, after = Native.clear_cache
      before - after
    end

    # Caps what this process may allocate on the device; 0 lifts the cap.
    # Read `limit` first if you mean to restore it: MLX reports the cap now
    # in force, not the one it replaced.
    def limit=(bytes)
      Native.memory_limit = bytes
    end

    # Forgets the high-water mark, so a later reading is about what
    # follows rather than about everything so far.
    def reset_peak! = Native.reset_peak_memory
  end
end
