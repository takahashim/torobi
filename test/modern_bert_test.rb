# frozen_string_literal: true

require_relative "test_helper"
require "json"

# ModernBERT as Torobi describes it, held to a checkpoint that exists
# (docs/plan.md M3b).
#
# The comparison is against a versioned artifact rather than the model
# itself: `test/oracle/ruri-v3-130m.json` records what cl-nagoya/ruri-v3-130m
# actually holds, so this runs on a machine that has never downloaded it.
# Regenerating it is `rake oracle` and fails closed when the checkpoint is
# not there (section 12).
#
# What it proves is structural rather than numerical: that the graph
# declares exactly the parameters the published model has, by name and by
# shape. That is most of what an architecture mistake is, and it is worth
# having before any forward pass.
class ModernBertTest < Minitest::Test
  ORACLE = File.expand_path("oracle/ruri-v3-130m.json", __dir__)

  def oracle = @oracle ||= JSON.parse(File.read(ORACLE))

  def config = Torobi::Models::ModernBERT.from_hash(oracle.fetch("config"))

  def graph(seq: 16) = Torobi::Models::ModernBERT.graph(config, seq:)

  def test_the_recorded_config_is_the_one_this_builder_understands
    c = config

    assert_equal 512, c.hidden_size
    assert_equal 19, c.num_hidden_layers
    assert_equal 8, c.num_attention_heads
    assert_equal 64, c.head_dim
    # Layer 0 attends globally, then every third; the rest are local, and
    # they use a different rotary base.
    assert c.global?(0)
    refute c.global?(1)
    assert c.global?(3)
    assert_in_delta(160_000.0, c.theta(0))
    assert_in_delta(10_000.0, c.theta(1))
  end

  # The claim: every parameter the published model holds is one this graph
  # declares, with the same shape, and nothing else.
  def test_the_graph_declares_exactly_what_the_checkpoint_holds
    declared = graph.parameters.to_h { |spec| [spec.path, spec.shape] }
    held = oracle.fetch("parameters").transform_values { |t| t.fetch("shape") }

    assert_empty held.keys - declared.keys, "the checkpoint holds parameters this graph does not"
    assert_empty declared.keys - held.keys, "this graph declares parameters the checkpoint lacks"
    mismatched = declared.filter_map do |path, shape|
      "#{path}: declares #{shape.inspect}, holds #{held[path].inspect}" if held[path] != shape
    end

    assert_empty mismatched
    assert_equal 116, declared.size
  end

  # Which means it imports with no renaming: the paths are the
  # checkpoint's own names.
  def test_the_paths_are_the_checkpoints_own_names
    paths = graph.parameters.map(&:path)

    assert_includes paths, "embeddings.tok_embeddings.weight"
    assert_includes paths, "layers.0.attn.Wqkv.weight"
    assert_includes paths, "layers.18.mlp.Wo.weight"
    assert_includes paths, "final_norm.weight"
    # Layer 0 has no attention norm, which is the reference's own
    # asymmetry rather than an omission here.
    refute_includes paths, "layers.0.attn_norm.weight"
    assert_includes paths, "layers.1.attn_norm.weight"
  end

  def test_the_graph_is_built_and_shaped_before_anything_runs
    g = graph(seq: 32)

    assert_equal [nil, 32, 512], g.nodes.last.shape
    assert_equal %w[hidden], g.outputs.keys
    # 19 layers of it, so a mistake in one block is 19 mistakes.
    assert_operator g.nodes.size, :>, 600
  end

  # The rest of this file needs the published checkpoint, which is half a
  # gigabyte and not something to require of a checkout. Everything above
  # runs without it, which is why the inventory is committed.
  def checkpoint
    dir = ENV["RURI_V3_130M"] || File.expand_path(
      "~/.cache/huggingface/hub/models--cl-nagoya--ruri-v3-130m/snapshots/" \
      "#{oracle.fetch("revision")}"
    )
    File.directory?(dir) ? dir : nil
  end

  def batch(seq)
    vocab = config.vocab_size
    ids = Array.new(seq) { |i| ((i * 137) + 11) % vocab }
    Torobi::Models::ModernBERT.batch(config, [ids], seq:)
  end

  # With an objective, so the run has something to differentiate.
  def trainable_config(seq)
    encoder = graph(seq:)
    Torobi::GraphConfig.new(
      models: { m: encoder },
      objective: Torobi.objective(m: encoder) do |g|
        g.output :loss, g.mean(g.from_model(:m, :hidden).square)
      end
    )
  end

  # 19 layers, 130M parameters, the published weights, one forward and one
  # step. What M3b asks for as a memory budget, measured rather than
  # estimated.
  def test_the_published_model_loads_and_takes_a_step
    dir = checkpoint
    skip "cl-nagoya/ruri-v3-130m is not in the cache (set RURI_V3_130M)" unless dir

    seq = 16
    Torobi::Session.open(trainable_config(seq),
                         pretrained: { m: File.join(dir, "model.safetensors") }) do |s|
      assert_equal 116, s.parameter_paths.size

      after_load = Torobi::Memory.report
      # 130M f32 parameters is about 520 MB, and the run should be holding
      # roughly that and not several times it.
      assert_operator after_load[:active], :>, 400_000_000
      assert_operator after_load[:active], :<, 900_000_000

      loss = s.evaluate(batch(seq))

      assert_predicate loss, :finite?
      assert_operator loss, :>, 0, "the mean of squared hidden states is positive"

      before = s.fetch("m.layers.0.attn.Wqkv.weight").to_a.first(4)
      s.step!(batch(seq))

      assert_equal 1, s.step
      refute_equal before, s.fetch("m.layers.0.attn.Wqkv.weight").to_a.first(4),
                   "a step should have moved the weights"
    end
  end

  # The numbers, against a second implementation of the same architecture.
  #
  # kohagi runs ModernBERT on candle, on the CPU, from the same published
  # weights, and its output is what people already rely on. Comparing
  # against it answers the question central differences cannot: whether the
  # forward is the right function, not merely one the backward agrees with
  # (docs/plan.md 9.2, and section 12's third layer).
  #
  # The artifact records the token ids as well as the vectors, so no
  # tokenizer is needed here; `rake oracle:forward` regenerates it and
  # fails closed when kohagi or the checkpoint is missing.
  FORWARD = File.expand_path("oracle/ruri-v3-130m.forward.json", __dir__)

  def forward_oracle = @forward_oracle ||= JSON.parse(File.read(FORWARD))

  def test_the_forward_matches_kohagi_on_the_published_weights
    dir = checkpoint
    skip "cl-nagoya/ruri-v3-130m is not in the cache (set RURI_V3_130M)" unless dir

    reference = forward_oracle

    assert_equal "mean", reference.dig("settings", "pooling")
    assert reference.dig("settings", "normalized")

    reference.fetch("cases").each do |example|
      ids = example.fetch("input_ids")
      mine = embed(dir, ids, reference.fetch("dim"))
      theirs = example.fetch("embedding")
      cosine = mine.zip(theirs).sum { |a, b| a * b }

      # Two f32 implementations this close saturate f32 arithmetic at
      # about 1e-7 (kohagi's own parity_check.py says so); 1e-5 leaves two
      # orders of magnitude and still catches a wrong op.
      assert_in_delta 1.0, cosine, 1e-5,
                      "#{example.fetch("text")[0, 16]}: #{ids.size} tokens"
    end
  end

  # The long case is the one that matters for the sliding window: at 302
  # tokens the local layers see less than the global ones, so a mask built
  # wrongly shows up here and nowhere else.
  def test_the_oracle_reaches_past_the_local_attention_window
    lengths = forward_oracle.fetch("cases").map { |c| c.fetch("input_ids").size }

    assert_operator lengths.max, :>, config.local_attention,
                    "a case longer than the local window is what tests the local mask"
  end

  # Mean pooling over the sequence, then L2, which is what ruri-v3's
  # 1_Pooling config asks for and what kohagi did.
  def embed(dir, ids, dim)
    seq = ids.size
    encoder = graph(seq:)
    session_config = Torobi::GraphConfig.new(
      models: { m: encoder },
      objective: Torobi.objective(m: encoder) do |g|
        g.output :loss, g.mean(g.from_model(:m, :hidden))
      end
    )
    Torobi::Session.open(session_config,
                         pretrained: { m: File.join(dir, "model.safetensors") }) do |s|
      s.tap("m.hidden", stat: :full)
      s.evaluate(Torobi::Models::ModernBERT.batch(config, [ids], seq:))
      rows = s.tapped.fetch("m.hidden").to_a.each_slice(dim).to_a
      pooled = Array.new(dim) { |j| rows.sum { |row| row[j] } / rows.size }
      length = Math.sqrt(pooled.sum { |v| v * v })
      pooled.map { |v| v / length }
    end
  end

  # The line: what config.json determines is Torobi's, what is upstream of
  # it is not. The pad token and the window come from the config this
  # already reads; tokenizing does not.
  def test_a_batch_is_built_from_ids_with_what_the_config_knows
    b = Torobi::Models::ModernBERT.batch(config, [[1, 843, 2], [1, 90, 77, 55, 2]], seq: 6)
    ids = b.fetch(:input_ids)

    assert_equal [2, 6], ids.shape
    assert_equal :i32, ids.dtype
    pad = config.pad_token_id

    assert_equal 3, pad, "ruri-v3 pads with 3, and this should read it rather than guess"
    assert_equal [1, 843, 2, pad, pad, pad, 1, 90, 77, 55, 2, pad], ids.to_a

    # Padding is masked per row, and only along the key axis: whether a
    # position may be attended to is a property of the key, so the query
    # axis is 1 and MLX broadcasts it.
    mask = b.fetch(:mask)

    assert_equal [2, 1, 1, 6], mask.shape
    rows = mask.to_a.each_slice(6).to_a

    assert_equal [0.0, 0.0, 0.0], rows[0].first(3)
    assert(rows[0].last(3).all?(&:negative?), "the first row's padding is masked")
    assert_equal [0.0] * 5, rows[1].first(5), "the second row keeps five positions"
  end

  # Nothing large is built as an Array of Floats. At the sizes a real run
  # uses, the mask would be the biggest thing Ruby touches.
  def test_the_masks_are_bytes_rather_than_millions_of_floats
    seq = 256
    b = Torobi::Models::ModernBERT.batch(config, Array.new(8) { Array.new(seq, 1) }, seq:)

    assert_kind_of Torobi::TensorData, b.fetch(:mask)
    assert_kind_of Torobi::TensorData, b.fetch(:window)
    # The padding is per row along one axis; the window is one for the
    # whole batch. Together that is far less than batch * seq^2.
    assert_equal 8 * seq * 4, b.fetch(:mask).bytesize
    assert_equal seq * seq * 4, b.fetch(:window).bytesize
    assert_operator b.fetch(:mask).bytesize + b.fetch(:window).bytesize,
                    :<, 8 * seq * seq * 4 / 4,
                    "the two together should be a fraction of one dense mask"
  end

  # The window depends on nothing in the batch, so it is the same object
  # from one step to the next.
  def test_the_window_is_built_once_for_a_shape
    first = Torobi::Models::ModernBERT.window(config, seq: 64)
    second = Torobi::Models::ModernBERT.window(config, seq: 64)

    assert_same first, second
  end

  # The padding width is the batch's now rather than the graph's, since a
  # graph can be built for no particular length at all (`towers`). What
  # the row is too long for is the padding it was handed.
  def test_a_row_longer_than_the_padding_is_refused_rather_than_cut
    e = assert_raises(Torobi::ConfigError) do
      Torobi::Models::ModernBERT.batch(config, [[1, 2, 3, 4]], seq: 2)
    end

    assert_match(/4 tokens and this batch pads to 2/, e.message)
    assert_match(/caller's to decide/, e.message)
  end

  # The window is the architecture's number, and it says what a position
  # may see rather than what this batch happens to hold.
  def test_the_window_is_the_one_the_config_names
    seq = 300
    half = config.local_attention / 2
    w = Torobi::Models::ModernBERT.window(config, seq:).to_a.each_slice(seq).to_a

    assert_in_delta(0.0, w[0][half], 0.001, "the far edge of the window is open")
    assert_operator w[0][half + 1], :<, 0, "one past it is not"
    assert_in_delta(0.0, w[200][200 - half])
    assert_operator w[200][200 - half - 1], :<, 0
    assert_operator w[200][200 + half], :>=, 0.0
  end

  def test_a_hidden_size_that_does_not_divide_into_heads_is_refused
    raw = oracle.fetch("config").merge("num_attention_heads" => 7)
    e = assert_raises(Torobi::ConfigError) { Torobi::Models::ModernBERT.from_hash(raw) }

    assert_match(/does not divide/, e.message)
  end
end
