# frozen_string_literal: true

require "minitest/autorun"
require_relative "../tools/jev_format"

# Pure Ruby, so it needs no extension: `tools/jev_format.rb` renders text
# and nothing else (docs/plan.md section 12, layer 1).
#
# Every expected string here is what the reference produced, byte for
# byte. The spaces after commas are the whole point of having this file:
# `json.dumps` writes them and Ruby's `JSON.generate` does not, and a
# state rendered with the wrong ones is text the model never trained on.
class JevFormatTest < Minitest::Test
  def test_content_matches_python_json_dumps
    assert_equal '{"前提": "二重請求", "仮説": "返金"}',
                 Jev.content("前提" => "二重請求", "仮説" => "返金")
    assert_equal '{"a": 1, "b": [1, 2]}', Jev.content("a" => 1, "b" => [1, 2])
    # Booleans and numbers are labels in the data (`choices` of a Noul
    # question are [true, false], of a Score question 0..5).
    assert_equal "true", Jev.content(true)
    assert_equal "false", Jev.content(false)
    assert_equal "0", Jev.content(0)
    # A string is itself, never quoted.
    assert_equal "x", Jev.content("x")
  end

  # A string is returned verbatim, so escaping happens only where a
  # string sits inside a state.
  def test_a_string_inside_a_state_is_escaped_as_json_does
    assert_equal '{"a": "x\ny"}', Jev.content("a" => "x\ny")
    assert_equal '{"a": "say \"hi\""}', Jev.content("a" => 'say "hi"')
  end

  def test_the_question_comes_before_the_state
    assert_equal "質問: 担当部署は？\n状況: {\"a\": 1}",
                 Jev.render_context({ "a" => 1 }, "担当部署は？")
  end

  def test_a_candidate_is_the_label_and_its_description
    assert_equal "billing — 請求・返金", Jev.render_candidate("billing", "請求・返金")
    # No description is the label alone; an integer label is serialized.
    assert_equal "billing", Jev.render_candidate("billing", nil)
    assert_equal "0 — 全く関係がない", Jev.render_candidate(0, "全く関係がない")
  end

  def test_candidates_render_in_order_and_check_their_length
    assert_equal ["a — true"],
                 Jev.render_candidates(%w[a], [true])
    assert_equal %w[entailment contradiction neutral],
                 Jev.render_candidates(%w[entailment contradiction neutral], nil)
    e = assert_raises(ArgumentError) { Jev.render_candidates(%w[a b], [nil]) }
    assert_match(/align with choices/, e.message)
  end

  def test_an_unsupported_value_is_refused_rather_than_guessed
    assert_raises(ArgumentError) { Jev.content(Object.new) }
  end
end
