# frozen_string_literal: true

require "json"

module Torobi
  # A base model held in a few bits per weight, as something a graph is
  # built with (docs/qlora.md).
  #
  # `Torobi::LoRA` leaves the base where it is and trains a pair of small
  # matrices beside each linear. The base still has to be in memory, and
  # in bf16 a 32B model is 61 GiB, which is on no machine this is for.
  # QLoRA (Dettmers et al., 2023) holds the base quantized instead: each
  # weight becomes a 4-bit integer, and every group of 64 of them shares
  # one scale and one bias, which is 4.5 bits a weight. The forward reads
  # it packed (`quantized_matmul`); the backward reaches the adapter and
  # the activations, never the base, so the base is frozen by nature
  # rather than by choice.
  #
  # What this holds is the decision, not the arithmetic:
  #
  #   quant   = Torobi::Quantization.new(bits: 4, group_size: 64)
  #   adapter = Torobi::LoRA.new(rank: 8, alpha: 16, on: %w[q_proj v_proj])
  #   model = Torobi::Models::Llama.causal_lm(config, seq: 512, adapter:, quantization: quant)
  #
  # A quantized linear is **three parameters**, named as mlx-lm's
  # checkpoints name them (`{name}.weight` packed u32, `{name}.scales`,
  # `{name}.biases`), so a published `*-4bit` file imports with no
  # renaming and a checkpoint of a quantized run is an ordinary
  # checkpoint. Only linears are quantized here. An embedding is a `take`
  # rather than a matmul, and a quantized one is a different operation;
  # a file that holds one quantized is dequantized when it is read.
  Quantization = Data.define(:bits, :group_size, :mode, :on) do
    # `on:` names the linears to quantize by their last segment, as a
    # `LoRA`'s `on:` does, or `:linears` for all of them, which is what a
    # published 4-bit checkpoint holds and the default.
    def initialize(bits: 4, group_size: 64, mode: "affine", on: :linears)
      bits = Integer(bits)
      group_size = Integer(group_size)
      mode = mode.to_s
      unless Shape::BITS.include?(bits)
        raise ConfigError, "bits must be one of #{Shape::BITS.join(", ")}, got #{bits}"
      end
      unless Shape::GROUP_SIZES.include?(group_size)
        raise ConfigError,
              "group_size must be one of #{Shape::GROUP_SIZES.join(", ")}, got #{group_size}"
      end
      unless Shape::QUANTIZATION_MODES.include?(mode)
        raise ConfigError,
              "quantization mode must be #{Shape::QUANTIZATION_MODES.join(", ")}, " \
              "got #{mode.inspect}"
      end

      on = on == :linears ? :linears : Array(on).map(&:to_s).freeze
      raise ConfigError, "a quantization quantizes something: on: names which linears" if on == []

      super
    end

    # What a checkpoint's `config.json` says, as mlx-lm writes it:
    #
    #   "quantization": {"group_size": 64, "bits": 4}
    #
    # `nil` when the config says nothing, which is a bf16 checkpoint.
    # `on:` is left at every linear; `from_header` reads which ones the
    # file actually holds quantized.
    def self.from_config(raw, on: :linears)
      given = raw["quantization"] || raw["quantization_config"]
      return nil unless given

      new(bits: given.fetch("bits"), group_size: given.fetch("group_size"),
          mode: given.fetch("mode", "affine"), on:)
    end

    # The same, with `on:` read from a safetensors header: the modules
    # that hold a `.scales` are the ones that are quantized. A published
    # file may leave some linears out, and the file is the only thing
    # that says which.
    def self.from_header(raw, header)
      quantized = header.keys.filter_map do |name|
        name.delete_suffix(".scales") if name.end_with?(".scales")
      end
      from_config(raw, on: quantized.map { |path| path.split(".").last }.uniq.sort)
    end

    # Both of the above from a checkpoint directory: its `config.json`
    # and the header at the front of its `model.safetensors`.
    def self.from_directory(dir)
      raw = JSON.parse(File.read(File.join(dir, "config.json")))
      header = File.open(File.join(dir, "model.safetensors"), "rb") do |f|
        JSON.parse(f.read(f.read(8).unpack1("Q<")))
      end
      from_header(raw, header)
    end

    # Whether the linear at `path` is held quantized. Matched on the last
    # segment, as a `LoRA` matches.
    def wraps?(path)
      return true if on == :linears

      path = path.to_s
      on.any? { |target| path == target || path.end_with?(".#{target}") }
    end

    # How many values one u32 of the packed weight holds.
    def values_per_word = 32 / bits

    # The shapes a `[d_out, d_in]` weight becomes.
    def packed_shape(d_out, d_in) = [d_out, d_in * bits / 32]
    def group_shape(d_out, d_in) = [d_out, d_in / group_size]

    # What a quantized linear adds to the graph, in place of its weight:
    # the three parameters, frozen, and the matmul that reads them
    # packed. Built through the graph's public vocabulary, with the names
    # a checkpoint uses.
    #
    # Frozen by declaration (`trainable: false`), which no adapter can
    # undo: a packed weight has no gradient, and the scales and biases
    # are the base model's, not something a fine-tune moves.
    def apply(g, x, d_in:, d_out:, name:)
      where = "linear #{name.inspect}"
      unless (d_in % group_size).zero?
        fits = Shape::GROUP_SIZES.select { |size| (d_in % size).zero? }
        raise ConfigError,
              "#{where}: d_in #{d_in} is not a multiple of group_size #{group_size}" \
              "#{fits.empty? ? ", and no group size divides it" : "; #{fits.join(" or ")} would"}"
      end
      unless ((d_in * bits) % 32).zero?
        raise ConfigError, "#{where}: d_in #{d_in} at #{bits} bits does not pack into whole words"
      end

      frozen = { "type" => "zeros" }
      weight = g.param("#{name}.weight", packed_shape(d_out, d_in), dtype: :u32,
                                                                     init: frozen, trainable: false)
      scales = g.param("#{name}.scales", group_shape(d_out, d_in), dtype: x.dtype,
                                                                   init: frozen, trainable: false)
      biases = g.param("#{name}.biases", group_shape(d_out, d_in), dtype: x.dtype,
                                                                   init: frozen, trainable: false)
      g.emit("quantized_matmul", inputs: [x, weight, scales, biases],
                                 attrs: { group_size:, bits:, mode: })
    end

    # Whether a parameter path is one of a quantized weight's own: the
    # packed weight cannot be told from a plain one by its name, so this
    # names the two that only a quantized linear has.
    def self.quantized?(path)
      %w[scales biases].any? { |suffix| path.to_s.end_with?(".#{suffix}") }
    end
  end
end
