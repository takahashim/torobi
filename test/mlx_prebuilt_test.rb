# frozen_string_literal: true

require_relative "test_helper"
require_relative "../lib/torobi/mlx_prebuilt"
require "minitest/mock"
require "digest"
require "fileutils"
require "stringio"

# The fetch a platform gem does on first use: MLX's Metal kernels come from
# an asset of their own, are checked against the pin, and land as one file
# beside the bundle. Only the download is faked here; the check, the
# unpack and the placement are the real ones, and they are where a wrong
# or replaced asset is caught. A fresh process reaches no network.
class MlxPrebuiltTest < Minitest::Test
  KERNELS = "pretend kernels"

  def test_the_metallib_asset_is_checked_unpacked_and_placed
    with_asset do |dir, archive, digest|
      into = File.join(dir, "into")
      placed = stub_fetch(archive, digest) { MlxPrebuilt.fetch_metallib(into:, io: StringIO.new) }

      assert_equal File.join(into, "mlx.metallib"), placed
      assert_equal KERNELS, File.read(placed)
      refute_path_exists File.join(into, "mlx.metallib.part"),
                         "the part file should be renamed away"
    end
  end

  def test_an_asset_that_is_not_the_pinned_one_is_refused
    with_asset do |dir, archive, _digest|
      into = File.join(dir, "into")

      error = assert_raises(MlxPrebuilt::Refused) do
        stub_fetch(archive, "0" * 64) { MlxPrebuilt.fetch_metallib(into:, io: StringIO.new) }
      end

      assert_match(/not the file this was built against/, error.message)
      refute_path_exists File.join(into, "mlx.metallib"), "nothing should be placed"
    end
  end

  # A tar.gz holding `mlx.metallib` at its root, as the release's asset is,
  # with the digest the pin would record for it.
  def with_asset
    Dir.mktmpdir("torobi-metallib-test") do |dir|
      File.write(File.join(dir, "mlx.metallib"), KERNELS)
      archive = File.join(dir, "kernels.tar.gz")
      unless system("tar", "-czf", archive, "-C", dir, "mlx.metallib")
        flunk "tar is needed to build the fixture"
      end

      yield dir, archive, Digest::SHA256.file(archive).hexdigest
    end
  end

  # The asset and its digest as the pin would answer, and the download as
  # the bytes already on disk.
  def stub_fetch(archive, digest, &)
    MlxPrebuilt.stub(:metallib_asset, "kernels.tar.gz") do
      MlxPrebuilt.stub(:metallib_digest, digest) do
        MlxPrebuilt.stub(:stream, ->(_uri, file, _io) { file.write(File.binread(archive)) }, &)
      end
    end
  end
end
