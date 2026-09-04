# frozen_string_literal: true

require "minitest/autorun"
require "json"
require "torobi/parquet"

# Reading parquet without arrow (docs/plan.md section 15.59).
#
# What is claimed is agreement with the implementation that wrote the
# files: `parquet/test/fixtures.py` makes them with pyarrow and records
# what pyarrow says they hold, and this reads them and gets the same.
# The fixtures are generated rather than borrowed, so what is tested is
# the format rather than somebody's dataset.
#
# And the other half: what this does not implement, it refuses. A reader
# that guessed at a codec it does not have would answer with numbers
# that are wrong, which is worse than not answering.
class ParquetTest < Minitest::Test
  FIXTURES = File.expand_path("fixtures", __dir__)

  def expected = @expected ||= JSON.parse(File.read(File.join(FIXTURES, "expected.json")))

  def fixture(name) = File.join(FIXTURES, "#{name}.parquet")

  def test_it_reads_what_pyarrow_wrote
    got = Torobi::Parquet.read(fixture("plain"))

    assert_equal expected.fetch("rows"), got
    assert_equal 20, got.size
    assert_nil got.first.fetch("score"), "a row with no value has none"
    assert_equal 4000, got[4].fetch("score"), "and an integer is an integer"
  end

  # Three row groups of eight, so the reader has to walk them and join
  # what it finds.
  def test_it_walks_every_row_group
    File.open(fixture("plain"), "rb") do |io|
      _, groups = Torobi::Parquet::Metadata.read(io)

      assert_equal 3, groups.size
      assert_equal [8, 8, 4], groups.map(&:rows)
    end
  end

  def test_it_reads_a_file_that_was_not_compressed
    assert_equal expected.fetch("rows"), Torobi::Parquet.read(fixture("uncompressed"))
  end

  # A column whose values are all different is written without a
  # dictionary, which is the other encoding a data page comes in.
  def test_it_reads_pages_that_have_no_dictionary
    assert_equal expected.fetch("wide"), Torobi::Parquet.read(fixture("no_dictionary"))
  end

  # --- what it asks for ---

  def test_a_column_nobody_asked_for_is_not_read
    got = Torobi::Parquet.read(fixture("plain"), columns: ["id"])

    assert_equal ["id"], got.first.keys
    assert_equal(expected.fetch("rows").map { |r| r.fetch("id") }, got.map { |r| r.fetch("id") })
  end

  def test_a_column_the_file_does_not_have_is_refused
    e = assert_raises(Torobi::Parquet::Error) do
      Torobi::Parquet.read(fixture("plain"), columns: %w[id nowhere])
    end

    assert_match(/no column "nowhere"/, e.message)
    assert_match(/"text"/, e.message, "and says what there is")
  end

  # Reading stops at the end of the group that reaches the ceiling,
  # because a group is the unit the file is written in.
  def test_it_stops_asking_after_enough_rows
    seen = []
    read = Torobi::Parquet.each_row(fixture("plain"), columns: ["id"], rows: 10) do |row|
      seen << row.fetch("id")
    end

    assert_equal 16, read, "two groups of eight, not ten rows"
    assert_equal 16, seen.size
    assert_equal "r0", seen.first
  end

  def test_it_can_be_asked_for_an_enumerator
    rows = Torobi::Parquet.each_row(fixture("plain"), columns: ["id"]).first(3)

    assert_equal(%w[r0 r1 r2], rows.map { |row| row.fetch("id") })
  end

  # --- what it refuses ---

  def test_a_codec_it_does_not_have_is_named_and_refused
    e = assert_raises(Torobi::Parquet::Error) { Torobi::Parquet.read(fixture("gzip")) }

    assert_match(/gzip is not read here/, e.message)
  end

  def test_the_second_page_version_is_named_and_refused
    e = assert_raises(Torobi::Parquet::Error) { Torobi::Parquet.read(fixture("page_v2")) }

    assert_match(/second version/, e.message)
  end

  def test_a_nested_column_is_named_and_refused
    e = assert_raises(Torobi::Parquet::Error) { Torobi::Parquet.read(fixture("nested")) }

    assert_match(/nested/, e.message)
  end

  def test_something_that_is_not_parquet_is_refused
    e = assert_raises(Torobi::Parquet::Error) { Torobi::Parquet.read(__FILE__) }

    assert_match(/PAR1/, e.message)
  end

  # --- the pieces, where they are worth holding on their own ---

  def test_snappy_repeats_a_pattern_a_copy_reaches_back_into
    # A literal "ab" and then a copy of six bytes from two back, which is
    # snappy's way of saying "repeat that three more times".
    block = [8, (1 << 2), "ab", 0b0000_1001, 2].pack("CCa2CC")

    assert_equal "abababab", Torobi::Parquet::Snappy.inflate(block)
  end

  def test_a_thrift_struct_comes_back_by_field_number
    # 15 00 = field 1, i32, zigzag 0; 15 04 = field 2, i32, zigzag 2; stop.
    fields, at = Torobi::Parquet::Thrift.struct([0x15, 0x00, 0x15, 0x04, 0x00].pack("C*"), 0)

    assert_equal({ 1 => 0, 2 => 2 }, fields)
    assert_equal 5, at
  end
end
