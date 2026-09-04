# frozen_string_literal: true

require_relative "test_helper"
require "tmpdir"
require "fileutils"

# Saying that the extension is behind the engine, before the engine says
# something else (docs/plan.md 15.64).
#
# The failure this is about is a real one from this session: the Ruby
# half had learned to write `[0, 0, 8, 64]` and the compiled half had
# not, so the engine refused a shape that was correct. Everything it
# said was true and none of it was the reason.
class FreshnessTest < Minitest::Test
  def with_tree
    Dir.mktmpdir do |root|
      FileUtils.mkdir_p(File.join(root, "engine", "src"))
      source = File.join(root, "engine", "src", "op.rs")
      built = File.join(root, "torobi.bundle")
      File.write(source, "//")
      File.write(built, "")
      yield root, source, built
    end
  end

  def test_an_extension_older_than_the_engine_is_stale
    with_tree do |root, source, built|
      File.utime(Time.now + 60, Time.now + 60, source)

      assert Torobi::Freshness.stale?(built, root:)
    end
  end

  def test_an_extension_built_after_the_engine_is_not
    with_tree do |root, _source, built|
      File.utime(Time.now + 60, Time.now + 60, built)

      refute Torobi::Freshness.stale?(built, root:)
    end
  end

  # An installed gem has no engine to compare against, and a warning
  # about a rebuild nobody can do is worse than no warning.
  def test_nothing_is_claimed_where_there_is_no_engine_to_compare_against
    Dir.mktmpdir do |root|
      built = File.join(root, "torobi.bundle")
      File.write(built, "")

      refute Torobi::Freshness.stale?(built, root:)
    end
  end

  def test_the_warning_says_what_to_run
    with_tree do |root, source, built|
      File.utime(Time.now + 60, Time.now + 60, source)
      said = StringIO.new

      Torobi::Freshness.warn!(built, root:, io: said)

      assert_match(/rake compile/, said.string)
      assert_match(/refuse graphs this Ruby builds/, said.string)
    end
  end

  # The one this process is running, so the check is about what is
  # loaded rather than what is lying around.
  def test_it_finds_the_extension_this_process_loaded
    skip "extension not compiled" unless defined?(Torobi::Session)

    assert_match(/torobi\.(bundle|so|dylib)\z/, Torobi::Freshness.extension.to_s)
  end
end
