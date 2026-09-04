# frozen_string_literal: true

require_relative "test_helper"

# What a model graph is being built for, as one value.
#
# Small enough that the interesting claims are about its defaults: a
# build that says nothing has to describe the ordinary graph, because
# every model's public constructor makes one and most callers name
# nothing at all.
class BuildTest < Minitest::Test
  def test_a_build_that_says_nothing_is_the_ordinary_graph
    build = Torobi::Models::Build.new

    assert_nil build.seq, "declared for no particular length"
    assert_nil build.rows, "and no particular number of rows"
    assert_equal :f32, build.dtype
    assert_equal :input_ids, build.field(:input_ids), "reading the plain batch fields"
  end

  # The prefix is what makes one description read two sets of rows
  # (docs/plan.md 15.63), so it is the one thing this value is really for.
  def test_fields_prefix_the_batch_names
    build = Torobi::Models::Build.new(fields: "queries.")

    assert_equal :"queries.input_ids", build.field(:input_ids)
    assert_equal :"queries.tokens", build.field("tokens")
  end

  # A second tower is the first with somewhere else to read from, which
  # is what `Data#with` says and why nothing here defines it.
  def test_a_build_can_be_pointed_somewhere_else
    build = Torobi::Models::Build.new(seq: 192, rows: 8, dtype: :bf16)
    other = build.with(fields: "documents.")

    assert_equal :"documents.input_ids", other.field(:input_ids)
    assert_equal [192, 8, :bf16], [other.seq, other.rows, other.dtype]
    assert_equal :input_ids, build.field(:input_ids), "and the first is untouched"
  end
end
