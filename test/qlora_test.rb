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
      compare_logits(logits, c.fetch("top_logits"), "#{SOURCE} case #{i}",
                     tolerance: PARITY_BF16, ranking: :first)
    end
  end

  # The wiring, without bf16 in the way. The same file read packed and
  # read as numbers, both in f32: the packed matmul unpacks each weight
  # as `scale · q + bias` in f32 and the other side dequantizes to f32
  # first, so the two differ only in the order the arithmetic happened
  # in, which is what `PARITY` is for (1.4e-5 between two f32
  # implementations of the bf16 model). A `quantized_matmul` that read
  # the wrong group, the wrong bits or the wrong transpose would be off
  # by the scale here, where bf16 would let it hide.
  def test_in_f32_the_packed_and_unpacked_readings_agree_to_rounding
    why = why_not(SOURCE, NAME)
    skip "nothing to compare. #{SOURCE}: #{why}" if why

    raw = inventory(NAME).fetch("config")
    config = described.from_hash(raw)
    quantization = Torobi::Quantization.from_header(raw, inventory(NAME).fetch("parameters"))
    dir = weights_for(SOURCE)
    ids = JSON.parse(File.read(File.join(ORACLE, "#{NAME}.forward.json")))
              .fetch("cases").first.fetch("input_ids")

    packed, packed_logits = run_case(config, dir, ids, dtype: :f32, quantization:)
    numbers, numbers_logits = run_case(config, dir, ids, dtype: :f32)

    compare_hidden(packed, numbers, "#{SOURCE} packed against unpacked, f32")
    top = numbers_logits.each_with_index.max_by(10) { |value, _| value }
                        .map { |value, id| { "id" => id, "value" => value } }
    compare_logits(packed_logits, top, "#{SOURCE} packed against unpacked, f32")
  end

  # The same file opened twice in bf16: once as it is, once as a bf16
  # model would be opened from it (the packed weights read as numbers,
  # docs/qlora.md 3.4 (A')). The same numbers in two layouts, so the
  # forward agrees to what bf16 lets two readings of one model agree to
  # (`PARITY_BF16`, and the measurement recorded there is this test's).
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
