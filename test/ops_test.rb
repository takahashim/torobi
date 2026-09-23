# frozen_string_literal: true

require_relative "test_helper"

# `config/ops.yml` is the single source of truth for the operation
# vocabulary, and `Ops::Spec` is what holds an emission to it. Its refusals
# are the contract the manifest makes, so each is asserted here rather than
# found out when a graph reaches the engine.
class OpsTest < Minitest::Test
  def spec(name) = Torobi::Ops.fetch(name, where: "t")

  def test_an_unknown_op_is_refused_by_name
    e = assert_raises(Torobi::ConfigError) { Torobi::Ops.fetch("improvise", where: "node 3") }

    assert_match(/node 3: unknown op "improvise"/, e.message)
  end

  def test_an_unknown_attribute_type_in_the_manifest_is_refused
    e = assert_raises(Torobi::ConfigError) do
      Torobi::Ops::Spec.from_manifest(
        { "name" => "x", "inputs" => 1, "shape_rule" => "same_as_input",
          "attributes" => { "a" => "bogus" } }
      )
    end

    assert_match(/unknown type "bogus"/, e.message)
  end

  def test_arity_is_checked_against_the_manifest
    e = assert_raises(Torobi::ConfigError) { spec("add").check!(inputs: 3, params: 0, attrs: {}, where: "t") }

    assert_match(/add takes 2\.\.2 input\(s\), got 3/, e.message)
  end

  def test_the_number_of_parameters_is_checked_against_the_manifest
    e = assert_raises(Torobi::ConfigError) { spec("add").check!(inputs: 2, params: 1, attrs: {}, where: "t") }

    assert_match(/add takes 0 parameter\(s\), got 1/, e.message)
  end

  def test_an_attribute_the_op_does_not_have_is_refused
    e = assert_raises(Torobi::ConfigError) do
      spec("abs").check!(inputs: 1, params: 0, attrs: { "axis" => 0 }, where: "t")
    end

    assert_match(/abs has no attribute "axis"/, e.message)
  end
end
