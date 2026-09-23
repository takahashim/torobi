# frozen_string_literal: true

module Torobi
  # A checkpoint directory, and what it says about itself, read without a
  # session.
  #
  # A checkpoint is a run's record, not only its numbers: it carries the
  # GraphConfig it belongs to, the parameter inventory with shapes and
  # dtypes, the optimizer and its step count, the build and the machine,
  # and whatever the caller recorded about where in the data it was
  # (docs/plan.md section 11.2). Reading it is how a caller decides which
  # one to resume from, before committing a session to it.
  #
  #   checkpoint = s.checkpoint!("run/000200", at: { epoch: 2 })
  #   checkpoint.step       # => 200
  #   checkpoint.position   # => {"epoch" => 2}
  #
  # Read from disk each time it is asked, never remembered: the directory
  # is the checkpoint, and a run may write it again.
  class Checkpoint
    def initialize(dir)
      @dir = dir.to_s
      freeze
    end

    attr_reader :dir

    # A checkpoint stands where a path is asked for: `File.exist?(c)`,
    # `s.restore(c)`.
    def to_path = @dir
    alias to_s to_path

    # Whether the directory holds a checkpoint at all.
    def exist? = File.exist?(File.join(@dir, "manifest.json"))

    # The manifest, as a Hash. Raises if the directory holds no checkpoint
    # or holds one of another schema.
    def manifest = Native.checkpoint_manifest(@dir)

    # The step the run had reached.
    def step = manifest.fetch("step")

    # Where in the data the run was, as the caller recorded it. Torobi does
    # not own datasets, so this is whatever `checkpoint!` was given.
    def position = manifest.dig("run", "position")

    # The GraphConfig the checkpoint belongs to, as JSON text. Present so
    # that a checkpoint can be read by someone who does not have the
    # description that produced it: a digest names one, it does not
    # reconstruct one.
    def graph_json = File.read(File.join(@dir, "graph.json"))

    def ==(other) = other.is_a?(Checkpoint) && other.dir == dir
    alias eql? ==

    def hash = [Checkpoint, dir].hash

    def inspect = "#<Torobi::Checkpoint #{dir}>"
  end
end
