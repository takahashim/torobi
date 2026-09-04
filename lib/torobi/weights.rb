# frozen_string_literal: true

require "json"

module Torobi
  # Where a session's parameters come from.
  #
  # Three answers, and exactly one of them per session: values written
  # out, one file holding every model's parameters under their qualified
  # paths, or a published file per model. Which it is decides how the
  # engine is opened, so the choosing happens once, here, rather than as
  # a branch inside `Session.open` next to everything else that opening a
  # session involves.
  #
  # The engine has the same three (`plan.rs`), which is why they are three
  # rather than one with options: they are read differently, not
  # configured differently.
  class Weights
    # Says which way, or refuses. `fresh` names the parameters meant to
    # be in no file, which only a published model can be missing.
    def self.of(weights: nil, weights_file: nil, pretrained: nil, fresh: [])
      fresh = Array(fresh).map(&:to_s)
      if !fresh.empty? && pretrained.nil?
        raise ArgumentError, "fresh: names what no file holds, so it goes with pretrained:"
      end

      given = { weights:, weights_file:, pretrained: }.compact
      unless given.size == 1
        raise ArgumentError,
              "a session needs its parameters, from exactly one place: " \
              "weights: {params: {path => {shape:, data:}}}, " \
              "weights_file: a safetensors path, or " \
              "pretrained: {model => path}" \
              "#{" (given #{given.keys.join(", ")})" unless given.empty?}"
      end

      kind, source = given.first
      new(kind:, source:, fresh:)
    end

    def initialize(kind:, source:, fresh: [])
      @kind = kind
      @source = source
      @fresh = fresh
      freeze
    end

    # Which of the three this is: :weights, :weights_file or :pretrained.
    attr_reader :kind

    # Opens the engine the way this source is read.
    def open(config, optimizer, seed)
      rule = JSON.generate(optimizer)
      case @kind
      when :weights_file
        Native::Session.open_from_file(config.canonical_json, @source.to_s, rule, seed)
      when :pretrained
        Native::Session.open_pretrained(config.canonical_json, JSON.generate(files),
                                        JSON.generate(@fresh), rule, seed)
      else
        Native::Session.open(config.canonical_json, JSON.generate(inline), rule, seed)
      end
    end

    private

    def files = @source.to_h { |model, path| [model.to_s, path.to_s] }

    # Inline weights as JSON: a TensorData spells its numbers out here.
    #
    # That is what this path is, and part of why it is the small one. A
    # parameter worth keeping as bytes arrives as a file, which never
    # becomes numbers in Ruby at all.
    def inline
      params = @source.fetch(:params) { @source.fetch("params") }
      { params: params.to_h { |path, t| [path, t.is_a?(TensorData) ? t.to_h : t] } }
    end
  end
end
