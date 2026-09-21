# frozen_string_literal: true

require "json"

# The rendering Jev Local shares between training and inference, ported so
# a Torobi run can produce a checkpoint its loader will accept
# (jev_local `modernbert/prompting.py`).
#
# **The bytes are the model's input, so they are not a style choice.**
# `content` reproduces Python's `json.dumps(value, ensure_ascii=False)`,
# which Ruby's `JSON.generate` is not: Python puts a space after a comma
# (`, `) as well as after a colon (`: `), and Ruby's compact form puts one
# after neither. A `state` rendered with the wrong separators is a
# different string from the one the reference trained on, and the model
# would be served text it has never seen.
module Jev
  # Stored with a checkpoint and checked at load time
  # (`jev_modernbert.json`); a model is never served under a different one.
  FORMAT_VERSION = "modernbert-jev/1"

  module_function

  # Strings verbatim, other JSON values serialized, as the reference does.
  def content(value)
    value.is_a?(String) ? value : dump(value)
  end

  # The question is first so that right-side truncation during training
  # removes state text and never the question.
  def render_context(state, question)
    "質問: #{question}\n状況: #{content(state)}"
  end

  def render_candidate(label, description = nil)
    text = content(label)
    description.nil? ? text : "#{text} — #{content(description)}"
  end

  def render_candidates(choices, descriptions)
    descriptions ||= Array.new(choices.size)
    unless descriptions.size == choices.size
      raise ArgumentError, "descriptions must align with choices"
    end

    choices.zip(descriptions).map { |label, description| render_candidate(label, description) }
  end

  # Python's `json.dumps` with its default separators and `ensure_ascii:
  # false`. Only the shapes a `state`, a label or a description can hold
  # are supported, and anything else is refused rather than guessed.
  def dump(value)
    case value
    when Hash
      pairs = value.map { |key, held| "#{quote(key.to_s)}: #{dump(held)}" }
      "{#{pairs.join(", ")}}"
    when Array
      "[#{value.map { |held| dump(held) }.join(", ")}]"
    when String then quote(value)
    when Integer then value.to_s
    when Float then JSON.generate(value)
    when true then "true"
    when false then "false"
    when nil then "null"
    else
      raise ArgumentError, "#{value.class} is not something a state holds"
    end
  end

  def quote(string) = JSON.generate(string)
end
