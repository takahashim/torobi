# frozen_string_literal: true

module Torobi
  # What a checkpoint says about itself, read without a session.
  #
  # A checkpoint is a run's record, not only its numbers: it carries the
  # GraphConfig it belongs to, the parameter inventory with shapes and
  # dtypes, the optimizer and its step count, the build and the machine,
  # and whatever the caller recorded about where in the data it was
  # (docs/plan.md section 11.2). Reading it is how a caller decides which
  # one to resume from, before committing a session to it.
  module Checkpoint
    module_function

    # The manifest, as a Hash. Raises if the directory holds no checkpoint
    # or holds one of another schema.
    def manifest(dir)
      Native.checkpoint_manifest(dir.to_s)
    end

    # The GraphConfig the checkpoint belongs to, as JSON text. Present so
    # that a checkpoint can be read by someone who does not have the
    # description that produced it: a digest names one, it does not
    # reconstruct one.
    def graph_json(dir)
      File.read(File.join(dir.to_s, "graph.json"))
    end

    # Where in the data the run was, as the caller recorded it. Torobi does
    # not own datasets, so this is whatever `checkpoint!` was given.
    def position(dir)
      manifest(dir).dig("run", "position")
    end

    # Whether a directory holds a checkpoint at all.
    def exist?(dir)
      File.exist?(File.join(dir.to_s, "manifest.json"))
    end
  end
end
