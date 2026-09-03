# frozen_string_literal: true

require_relative "test_helper"
require_relative "support/parity"
require "json"

# Gemma 3 as Torobi describes it (docs/plan.md section 15.54).
#
# The same three kinds of claim the Llama-shaped family is held to, on a
# model that is deliberately not one of them: four norms a layer, norms
# that scale by `1 + w`, q and k normalized per head, a gated MLP through
# the tanh approximation of GELU, embeddings scaled by the square root of
# the width, and most layers seeing only the recent past.
#
# Every one of those is the kind of difference that leaves a model which
# runs and trains and is not the published one, which is why the third
# kind of claim (the numbers) is the one that settles it.
class Gemma3Test < Minitest::Test
  include Parity

  def described = Torobi::Models::Gemma3

  PUBLISHED = { "google/gemma-3-270m" => "gemma-3-270m" }.freeze

  def inventory(name)
    (@inventories ||= {})[name] ||=
      JSON.parse(File.read(File.expand_path("oracle/#{name}.json", __dir__)))
  end

  def published = described.from_hash(inventory("gemma-3-270m").fetch("config"))

  def test_the_numbers_agree_with_the_reference_implementation
    compare_against_references(PUBLISHED)
  end

  def test_the_recorded_config_is_the_one_this_builder_understands
    c = published

    assert_equal 640, c.hidden_size
    assert_equal 18, c.num_hidden_layers
    # Four heads of 256 over a hidden state of 640: what the heads take
    # up is not the width of the model, which the Llama family assumes.
    assert_equal 256, c.head_dim
    assert_equal 1024, c.attention_size
    assert_equal 1, c.num_key_value_heads, "one key head for all four queries"
    assert c.tie_word_embeddings, "Gemma ties, and holds no lm_head at all"
    # Five local layers to one global, and the local ones rotate at a
    # lower base because they are addressing a window rather than 32k.
    assert_equal([5, 11, 17], (0...18).reject { |i| c.sliding?(i) })
    assert_in_delta 10_000.0, c.theta(0)
    assert_in_delta 1_000_000.0, c.theta(5)
    assert_in_delta 0.0625, c.scale, 1e-9
  end

  # The claim that makes `pretrained:` work with no renaming: 13 tensors
  # a layer, which is the four norms and the two head norms over what a
  # Llama layer has.
  def test_the_graph_declares_exactly_what_the_checkpoint_holds
    held = inventory("gemma-3-270m").fetch("parameters").transform_values { |t| t.fetch("shape") }
    declared = described.causal_lm(published, seq: 8)
                        .parameters.to_h { |spec| [spec.path, spec.shape] }

    assert_empty held.keys - declared.keys, "the checkpoint holds parameters this graph does not"
    assert_empty declared.keys - held.keys, "this graph declares parameters the checkpoint lacks"
    mismatched = declared.filter_map do |path, shape|
      "#{path}: declares #{shape.inspect}, holds #{held[path].inspect}" if held[path] != shape
    end

    assert_empty mismatched
    assert_equal 236, declared.size
    refute_includes declared.keys, "lm_head.weight"
    assert_includes declared.keys, "model.layers.0.self_attn.q_norm.weight"
    assert_includes declared.keys, "model.layers.0.post_feedforward_layernorm.weight"
  end

  # Gemma's norms are stored around zero and applied as `1 + w`, so a
  # fresh one is the identity where everyone else's is.
  def test_the_norms_are_stored_around_zero
    fresh = described.causal_lm(published, seq: 4).parameters
                     .find { |spec| spec.path.end_with?("layers.0.input_layernorm.weight") }

    assert_equal({ "type" => "zeros" }, fresh.initializer)
  end

  # --- what a local layer may see ---

  # The window is causal and windowed in one: a position sees itself and
  # at most `sliding_window - 1` positions before it, and nothing after.
  def test_the_window_is_the_recent_past_and_nothing_ahead
    config = published.with(sliding_window: 3)
    mask = described.window(config, seq: 5).to_a.each_slice(5).to_a
    allowed = mask.map { |row| row.each_index.select { |j| row[j].zero? } }

    assert_equal [[0], [0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]], allowed
    assert(mask.flatten.all? { |v| v.zero? || v < -1e8 }, "what is blocked is blocked hard")
  end

  # --- a model small enough to answer for itself ---

  SEQ = 6

  # Two local layers and one global, so the alternation is exercised, and
  # a window that cuts something off at this length.
  def small
    @small ||= described.from_hash(
      "vocab_size" => 11, "hidden_size" => 8, "intermediate_size" => 16,
      "num_hidden_layers" => 3, "num_attention_heads" => 2,
      "num_key_value_heads" => 1, "head_dim" => 4, "rms_norm_eps" => 1e-6,
      "rope_theta" => 1_000_000.0, "rope_local_base_freq" => 10_000.0,
      "sliding_window" => 3, "sliding_window_pattern" => 3,
      "query_pre_attn_scalar" => 4, "eos_token_id" => 10
    )
  end

  def model = @model ||= described.causal_lm(small, seq: SEQ)

  def graph_config
    @graph_config ||= Torobi::GraphConfig.new(
      models: { m: model },
      objective: Torobi.objective(m: model) do |g|
        at = g.cross_entropy(g.from_model(:m, :logits),
                             g.from_batch(:targets, [nil, SEQ], dtype: :i32))
        g.output :loss, g.mean(at)
      end
    )
  end

  def weights
    @weights ||= begin
      rng = Random.new(31)
      params = graph_config.parameters.to_h do |parameter|
        shape = parameter.spec.shape
        [parameter.qualified_path,
         { shape:, data: Array.new(shape.reduce(1, :*)) { rng.rand(-0.4..0.4) } }]
      end
      { params: }
    end
  end

  ROWS = [[3, 8, 5, 9, 2, 7], [4, 6, 1, 2, 5, 3]].freeze

  def batch(rows = ROWS)
    described.batch(small, rows, seq: SEQ)
             .merge(targets: Torobi::TensorData.from_a(
               [rows.size, SEQ], rows.flat_map { |r| r[1..] + [small.pad_token_id] }, dtype: :i32
             ))
  end

  def test_the_model_scores_every_position_over_the_whole_vocabulary
    logits = Torobi::Session.open(graph_config, weights:) { |s| s.forward(batch)["m.logits"] }

    assert_equal [ROWS.size, SEQ, small.vocab_size], logits.shape
  end

  def test_a_position_is_not_changed_by_what_comes_after_it
    later = ROWS.map(&:dup)
    later.first[-1] = (later.first[-1] + 1) % small.vocab_size

    before, after = Torobi::Session.open(graph_config, weights:) do |s|
      [s.forward(batch)["m.logits"].to_a, s.forward(batch(later))["m.logits"].to_a]
    end
    rows = ->(all) { all.each_slice(small.vocab_size).each_slice(SEQ).to_a }

    assert_equal rows.call(before).first.first(SEQ - 1), rows.call(after).first.first(SEQ - 1),
                 "an earlier position read a later token"
  end

  # For every parameter of a whole Gemma, autodiff agrees with the
  # forward it is differentiating. What this covers that the Llama tests
  # do not: the wrapped norms, the offset on their weights, the head
  # norms, the tanh gate, and a mask that is a window rather than a flag.
  def test_every_gradient_agrees_with_central_differences
    Torobi::Session.open(graph_config, weights:) do |s|
      analytic = s.gradients(batch)
      biggest = analytic.values.flat_map(&:to_a).map(&:abs).max

      assert_operator biggest, :>, 1e-3, "the model should have gradients worth checking"

      worst = { path: nil, at: nil, delta: 0.0 }
      s.parameter_paths.each do |path|
        held = s.fetch(path)
        sample(held.size).each do |i|
          numeric = central_difference(s, path, held, i)
          delta = (analytic.fetch(path).to_a[i] - numeric).abs
          worst = { path:, at: i, delta: } if delta > worst[:delta]
        end
        s.put(path, held)
      end

      assert_operator worst[:delta], :<, TOLERANCE,
                      "#{worst[:path]}[#{worst[:at]}] disagrees with the forward"
    end
  end

  def test_a_batch_carries_the_window_the_local_layers_read
    fields = described.batch(small, ROWS, seq: SEQ)

    assert_equal %i[input_ids window], fields.keys.sort
    assert_equal [1, 1, SEQ, SEQ], fields.fetch(:window).shape

    # A configuration with no local layer asks for no window, for the
    # reason ModernBERT gives: a batch field with nothing to read it.
    global = small.with(layer_types: ["full_attention"] * small.num_hidden_layers)

    assert_equal [:input_ids], described.batch(global, ROWS, seq: SEQ).keys
  end

  # Swept rather than guessed, as `modern_bert_gradient_test` was, and
  # the bottom of the curve is elsewhere: 4.7e-3 at a step of 3e-3,
  # 6.5e-4 at 1e-3, 6.8e-4 at 3e-4, then 2.5e-3 at 1e-4 as f32 rounding
  # in the two forwards takes over.
  #
  # Coarser than the encoder's because Gemma is: the embeddings are
  # multiplied by the square root of the width and every norm scales by
  # `1 + w`, so the loss curves harder and a difference of two forwards
  # says less about the slope. Against a largest gradient of 0.47 this
  # still refuses anything wrong by half a percent, and a backward that
  # missed a norm or took the exact GELU for the tanh one is wrong by
  # tens of percent.
  TOLERANCE = 3e-3
  STEP = 1e-3

  private

  def central_difference(session, path, held, index, step: STEP)
    moved = lambda do |delta|
      data = held.to_a
      data[index] += delta
      session.put(path, Torobi::TensorData.from_a(held.shape, data))
      session.evaluate(batch)
    end
    (moved.call(step) - moved.call(-step)) / (2 * step)
  end

  def sample(size)
    return (0...size).to_a if size <= 3

    Array.new(3) { |i| (i * (size - 1)) / 2 }
  end
end
