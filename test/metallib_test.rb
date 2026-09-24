# frozen_string_literal: true

require_relative "test_helper"
require "minitest/mock"
require "stringio"
require "tmpdir"

# What a run is told about the kernels, and where they come from: the copy
# beside the bundle in a checkout, or one fetched into a cache the gem can
# write. `Torobi::Native.set_metallib_path` is the point - MLX is named the
# file rather than left to find it beside a directory that may be
# read-only.
class MetallibTest < Minitest::Test
  def setup
    skip "not macOS" unless MlxPrebuilt.metal?
    skip "extension not compiled" unless defined?(Torobi::Native)
  end

  def test_a_checkout_copy_is_what_mlx_is_told_to_use
    beside = Torobi::Preflight::METALLIB
    skip "no metallib beside the bundle (run `rake metallib`)" unless File.file?(beside)

    named = []

    Torobi::Native.stub(:set_metallib_path, ->(path) { named << path }) do
      assert_equal beside, Torobi::Metallib.ensure!(io: StringIO.new)
    end

    assert_equal [beside], named, "MLX should be told where the kernels are"
  end

  def test_the_copy_beside_the_bundle_wins_over_the_cache
    in_dir do |dir|
      beside = touch(dir, "beside.metallib")
      cached = touch(dir, "cached.metallib")

      assert_equal beside, Torobi::Metallib.resolved(io: StringIO.new, beside:, cached:)
    end
  end

  def test_a_cached_copy_is_used_rather_than_fetched_again
    in_dir do |dir|
      cached = touch(dir, "cached.metallib")
      absent = File.join(dir, "absent.metallib")

      MlxPrebuilt.stub(:fetch_metallib, lambda { |**|
        flunk "a copy that is already there is not fetched again"
      }) do
        assert_equal cached, Torobi::Metallib.resolved(io: StringIO.new, beside: absent, cached:)
      end
    end
  end

  def touch(dir, name)
    path = File.join(dir, name)
    File.write(path, "kernels")
    path
  end

  def in_dir(&)
    Dir.mktmpdir("torobi-metallib", &)
  end
end
