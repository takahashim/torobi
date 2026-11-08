# frozen_string_literal: true

require_relative "test_helper"
require_relative "support/parity"
require "json"

# A published 4-bit model, read as it is and held to the implementation
# it was made for (docs/qlora.md, Phase 1).
#
# The structural half (the description declares what the file holds, by
# name and shape) is in `quantization_test.rb` and needs no weights. This
# is the numeric half: the hidden states and the next tokens, from mlx-lm
# on the same file (`rake oracle:qwen2_4bit_forward`). It needs the
# weights and the recorded reference, and skips saying which is missing.
class QLoRATest < Minitest::Test
  include Parity

  def described = Torobi::Models::Llama

  SOURCE = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
  NAME = "qwen2.5-0.5b-instruct-4bit"

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  def inventory(name)
    path = File.join(ORACLE, "#{name}.json")
    skip "no inventory at #{path} (rake oracle:qwen2_4bit)" unless File.exist?(path)

    JSON.parse(File.read(path))
  end

  def test_the_forward_through_the_published_4bit_file_agrees_with_mlx_lm
    why = why_not(SOURCE, NAME)
    skip "nothing to compare. #{SOURCE}: #{why}" if why

    raw = inventory(NAME).fetch("config")
    config = described.from_hash(raw)
    quantization = Torobi::Quantization.from_header(raw, inventory(NAME).fetch("parameters"))
    reference = JSON.parse(File.read(File.join(ORACLE, "#{NAME}.forward.json")))
    dir = weights_for(SOURCE)

    assert_equal config.hidden_size, reference.fetch("hidden_size")
    check_weights(inventory(NAME), reference, dir, SOURCE)
    reference.fetch("cases").each_with_index do |c, i|
      hidden, logits = run_case(config, dir, c.fetch("input_ids"), dtype: :bf16, quantization:)
      compare_hidden(hidden, c.fetch("hidden"), "#{SOURCE} case #{i}", tolerance: PARITY_BF16)
      compare_logits(logits, c.fetch("top_logits"), "#{SOURCE} case #{i}", tolerance: PARITY_BF16)
    end
  end

  # The same file opened twice: once as it is, once as a bf16 model would
  # be opened from it (the packed weights read as numbers, docs/qlora.md
  # 3.4 (A')). The two are the same numbers in two layouts, so their
  # forward agrees to bf16's rounding, and the second holds about twice
  # the bytes: what quantizing is for, measured on the model it is for.
  def test_the_same_file_read_packed_and_read_as_numbers_agree
    why = why_not(SOURCE, NAME)
    skip "nothing to compare. #{SOURCE}: #{why}" if why

    raw = inventory(NAME).fetch("config")
    config = described.from_hash(raw)
    quantization = Torobi::Quantization.from_header(raw, inventory(NAME).fetch("parameters"))
    dir = weights_for(SOURCE)
    ids = JSON.parse(File.read(File.join(ORACLE, "#{NAME}.forward.json")))
              .fetch("cases").first.fetch("input_ids")

    packed, = run_case(config, dir, ids, dtype: :bf16, quantization:)
    numbers, = run_case(config, dir, ids, dtype: :bf16)

    compare_hidden(packed, numbers, "#{SOURCE} packed against unpacked", tolerance: PARITY_BF16)
  end
end
