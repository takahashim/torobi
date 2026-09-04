# frozen_string_literal: true

module Torobi
  # Whether the compiled extension is older than the engine it was built
  # from.
  #
  # Torobi is one description held in two halves: the Ruby that builds a
  # graph and the Rust that runs it. They are compiled at different
  # times, and when the Rust half is behind, what it looks like is the
  # engine refusing a graph Ruby built happily. An op it does not know, a
  # shape it will not take: all of them true, none of them the reason.
  #
  #   m: node 11 (reshape): a shape is positive integers and at most
  #   one -1, got [0, 0, 8, 64]
  #
  # That is a confusing way to be told to run `rake compile`, so this
  # says it plainly instead, once, when the sources are there to compare
  # against. An installed gem carries no `engine/src` and this does
  # nothing at all.
  module Freshness
    module_function

    # The extension this process actually loaded, whatever it is called
    # on this platform, or nil if none was.
    def extension
      $LOADED_FEATURES.grep(%r{/torobi\.(bundle|so|dylib)\z}).first
    end

    # The newest thing the engine is built from, or nil where there is
    # nothing to compare against. `Cargo.toml` counts: a dependency
    # changing is a rebuild too.
    def engine(root)
      Dir[File.join(root, "engine", "src", "**", "*.rs"),
          File.join(root, "engine", "Cargo.toml")]
        .filter_map { |path| File.mtime(path) }.max
    end

    def stale?(built = extension, root: File.expand_path("../..", __dir__))
      return false unless built && File.exist?(built)

      newest = engine(root)
      !newest.nil? && newest > File.mtime(built)
    end

    # Said on stderr rather than raised. What is loaded may still be
    # perfectly able to run what this process asks of it, and a warning
    # that turns out to be unnecessary is cheaper than a refusal that
    # turns out to be wrong.
    def warn!(built = extension, root: File.expand_path("../..", __dir__), io: $stderr)
      return unless stale?(built, root:)

      io.puts "torobi: engine/src is newer than the extension that is loaded " \
              "(#{built}). Run `rake compile`; until then the engine may refuse " \
              "graphs this Ruby builds, and say something else."
    end
  end
end
