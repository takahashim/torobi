# frozen_string_literal: true

require_relative "test_helper"

class DslTest < Minitest::Test
  def linear_model
    Torobi.graph do |g|
      x = g.input :x, [nil, 4]
      g.output :out, g.linear(x, 2, name: "linear")
    end
  end

  def test_a_linear_model_lowers_to_named_parameters_and_primitives
    graph = linear_model

    assert_equal %w[linear.weight linear.bias], graph.parameters.map(&:path)
    assert_equal [2, 4], graph.parameters[0].shape # PyTorch layout [out, in]
    assert_equal %w[parameter transpose matmul parameter add], graph.nodes.map(&:op)
    assert_equal [nil, 2], graph.nodes.last.shape
    assert_equal :f32, graph.nodes.last.dtype
  end

  # A config without an objective takes the model's one output as the loss,
  # so the model here reduces to a scalar.
  def scalar_model
    Torobi.graph do |g|
      x = g.input :x, [nil, 4]
      g.output :loss, g.mean(g.linear(x, 2, name: "linear"))
    end
  end

  def test_the_same_definition_yields_the_same_digest
    a = Torobi::GraphConfig.new(models: { "m" => scalar_model })
    b = Torobi::GraphConfig.new(models: { "m" => scalar_model })

    assert_equal a.digest, b.digest
  end

  def test_scopes_name_the_parameters_of_repeated_components
    graph = Torobi.graph do |g|
      h = g.input :h, [nil, nil, 8]
      2.times do |i|
        g.scope("layers.#{i}") { h += g.linear(h, 8, name: "ff", bias: false) }
      end
      g.output :out, h
    end

    assert_equal %w[layers.0.ff.weight layers.1.ff.weight], graph.parameters.map(&:path)
  end

  def test_shape_errors_are_reported_at_build_time_with_the_node_named
    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph do |g|
        a = g.input :a, [nil, 3]
        b = g.input :b, [4, 5]
        g.output :out, g.matmul(a, b)
      end
    end
    assert_match(/node 0 \(matmul\)/, e.message)
    assert_match(/dimension .* mismatch \(3 vs 4\)/, e.message)
  end

  # A reshape may keep dimensions it does not know, which is the only way
  # to split the last axis of something whose batch *and* sequence are
  # both symbolic (docs/plan.md 15.63).
  def test_a_reshape_can_keep_the_dimensions_it_does_not_know
    shape = nil
    Torobi.graph do |g|
      x = g.input :x, [nil, nil, 8]
      split = x.reshape(shape: [0, 0, 2, 4])
      shape = split.shape
      g.output :out, split
    end

    assert_equal [nil, nil, 2, 4], shape
  end

  # A component names its parameters under a scope, and a caller may
  # write the same path with a dot instead. The two are the same graph,
  # down to the node names a tap reaches for and the paths an adapter
  # matches against, so a description may say which of them it means
  # (`Models::Llama` says it with scopes) without saying anything else.
  def test_a_scope_and_a_dotted_name_are_the_same_path
    dotted = Torobi.graph do |g|
      x = g.input :x, [nil, 8]
      g.scope("layers.0") { g.output :out, g.linear(x, 8, name: "mlp.gate_proj", bias: false) }
    end
    scoped = Torobi.graph do |g|
      x = g.input :x, [nil, 8]
      g.scope("layers.0") do
        g.scope("mlp") { g.output :out, g.linear(x, 8, name: "gate_proj", bias: false) }
      end
    end

    assert_equal %w[layers.0.mlp.gate_proj.weight], scoped.parameters.map(&:path)
    assert_equal dotted.parameters.map(&:path), scoped.parameters.map(&:path)
    assert_equal dotted.nodes.filter_map(&:name), scoped.nodes.filter_map(&:name)
    assert_equal Torobi::GraphConfig.new(models: { m: dotted }, train: []).digest,
                 Torobi::GraphConfig.new(models: { m: scoped }, train: []).digest
  end

  # --- reading a wide row as heads ---

  # The pair is a `reshape` and a `transpose` and nothing else, so a
  # description that says what it means gets the graph it would have got
  # by writing `[0, 2, 1, 3]` out.
  def test_splitting_and_merging_heads_is_the_reshape_and_transpose_by_hand
    def_by_hand = Torobi.graph do |g|
      x = g.input :x, [nil, nil, 896]
      h = x.reshape(shape: [0, 0, 14, 64]).transpose(axes: [0, 2, 1, 3])
      g.output :out, h.transpose(axes: [0, 2, 1, 3]).reshape(shape: [0, 0, 896])
    end
    by_name = Torobi.graph do |g|
      x = g.input :x, [nil, nil, 896]
      g.output :out, x.split_heads(14).merge_heads
    end

    assert_equal Torobi::GraphConfig.new(models: { m: def_by_hand }, train: []).digest,
                 Torobi::GraphConfig.new(models: { m: by_name }, train: []).digest
  end

  # Both leading dimensions stay unknown, which is what lets a graph be
  # built for no particular sequence (docs/plan.md 15.63). And the query
  # heads may outnumber the key heads, which is what grouped-query
  # attention is; each side is told its own count.
  def test_heads_split_over_a_symbolic_batch_and_sequence
    shapes = []
    Torobi.graph do |g|
      q = g.input(:q, [nil, nil, 896]).split_heads(14)
      k = g.input(:k, [nil, nil, 128]).split_heads(2)
      back = q.merge_heads
      shapes = [q.shape, k.shape, back.shape]
      g.output :q, back
      g.output :k, k
    end

    assert_equal [[nil, 14, nil, 64], [nil, 2, nil, 64], [nil, nil, 896]], shapes
  end

  # The width it lands on is the heads times what each holds, which is
  # not the model's hidden size: Gemma 3 270m projects a hidden state of
  # 640 into four heads of 256, and comes back through `o_proj`.
  def test_merging_heads_lands_on_what_the_heads_hold
    shape = nil
    Torobi.graph do |g|
      attended = g.input :attended, [nil, nil, 1024]
      merged = attended.split_heads(4).merge_heads
      shape = merged.shape
      g.output :out, merged
    end

    assert_equal [nil, nil, 1024], shape
  end

  def test_a_width_that_does_not_divide_into_heads_is_refused
    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph do |g|
        g.output :out, g.input(:x, [nil, nil, 896]).split_heads(5)
      end
    end

    assert_match(/896 does not divide into 5 heads/, e.message)
  end

  def test_heads_are_split_out_of_a_row_and_merged_out_of_a_head_layout
    split = assert_raises(Torobi::ConfigError) do
      Torobi.graph { |g| g.output :out, g.input(:x, [nil, 896]).split_heads(14) }
    end
    merge = assert_raises(Torobi::ConfigError) do
      Torobi.graph { |g| g.output :out, g.input(:x, [nil, nil, 8]).merge_heads }
    end

    assert_match(/expected \[batch, seq, width\]/, split.message)
    assert_match(/expected \[batch, heads, seq, dim\]/, merge.message)
  end

  def test_a_width_nothing_knows_cannot_be_divided_into_heads
    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph { |g| g.output :out, g.input(:x, [nil, nil, nil]).split_heads(14) }
    end

    assert_match(/must be concrete/, e.message)
  end

  def test_a_kept_dimension_the_input_does_not_have_is_refused
    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph do |g|
        x = g.input :x, [nil, 8]
        g.output :out, x.reshape(shape: [0, 0, 0, 8])
      end
    end

    assert_match(/keeps 3 dimensions/, e.message)
    assert_match(/has 2/, e.message)
  end

  # Only the leading ones: what follows a kept dimension is being
  # re-divided, and has no dimension of its own to stand for.
  def test_a_kept_dimension_after_a_divided_one_is_refused
    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph do |g|
        x = g.input :x, [nil, 4, 8]
        g.output :out, x.reshape(shape: [0, 2, 0])
      end
    end

    assert_match(/only the leading ones can be kept/, e.message)
  end

  # The old rule still holds past what is kept: the concrete part has to
  # be preserved exactly, and the message says where it was looking.
  def test_what_is_left_after_the_kept_dimensions_still_has_to_fit
    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph do |g|
        x = g.input :x, [nil, nil, 8]
        g.output :out, x.reshape(shape: [0, 0, 3, 4])
      end
    end

    assert_match(/past the 2 it keeps/, e.message)
    assert_match(/holds 8/, e.message)
  end

  def test_values_cannot_cross_graphs
    stray = nil
    Torobi.graph do |g|
      stray = g.input :x, [2]
      g.output :out, stray.neg
    end
    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph { |g| g.output :out, stray.gelu }
    end
    assert_match(/belongs to a different graph/, e.message)
  end

  def test_split_produces_slices_and_checks_divisibility
    graph = Torobi.graph do |g|
      x = g.input :x, [nil, 6]
      a, b, c = x.split(3, axis: -1)
      g.output :out, a + b + c
    end
    slices = graph.nodes.select { |n| n.op == "slice" }

    assert_equal([0, 2, 4], slices.map { |n| n.attributes["start"] })
    assert_equal [nil, 2], slices.first.shape

    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph { |g| g.input(:x, [nil, 5]).split(2) }
    end
    assert_match(/does not divide/, e.message)
  end

  def test_embedding_gathers_by_i32_ids
    graph = Torobi.graph do |g|
      ids = g.input :ids, [nil, nil], dtype: :i32
      g.output :out, g.embedding(ids, vocab: 100, dim: 16, name: "emb")
    end

    assert_equal [nil, nil, 16], graph.nodes.last.shape

    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph do |g|
        ids = g.input :ids, [nil], dtype: :f32
        g.output :out, g.embedding(ids, vocab: 10, dim: 4, name: "emb")
      end
    end
    assert_match(/indices must be i32/, e.message)
  end

  def test_the_objective_vocabulary_reaches_a_scalar_loss
    graph = Torobi.graph do |g|
      s = g.input :student_logits, [nil]
      t = g.input :teacher_logits, [nil]
      labels = g.input :label_loss, [nil]
      g.output :out, g.mean((g.mse(s, t) * 0.7) + (g.mean(labels) * 0.3))
    end

    assert_equal [], graph.nodes.last.shape
  end

  def test_a_modernbert_shaped_block_builds_and_keeps_its_shape
    dim = 8
    heads = 2
    graph = Torobi.graph do |g|
      h = g.input :h, [nil, nil, dim]
      g.scope "layers.0" do
        qkv = g.linear(h, dim * 3, name: "wqkv", bias: false)
        q, k, v = qkv.split(3, axis: -1)
        a = g.sdpa(q.rope(theta: 10_000), k.rope(theta: 10_000), v, scale: 1.0 / heads)
        h += g.linear(a, dim, name: "wo", bias: false)
        h += g.geglu(g.layer_norm(h, name: "mlp_norm"), dim * 2, name: "mlp")
      end
      g.output :out, h
    end

    assert_equal [nil, nil, dim], graph.nodes.last.shape
    assert_includes graph.parameters.map(&:path), "layers.0.mlp.wi.weight"
    assert_includes graph.parameters.map(&:path), "layers.0.mlp_norm.weight"
  end

  # A number may lead: `1.0 - x` is how a loss is written on paper, and
  # Ruby's coerce is what lets it be written that way here.
  def test_a_number_may_lead_an_expression
    graph = Torobi.graph do |g|
      x = g.input :x, [nil, 2]
      g.output :loss, g.mean(2.0 * (1.0 - x))
    end

    assert_operator graph.nodes.size, :>, 2
    # And nothing but a number may.
    assert_raises(TypeError) do
      Torobi.graph { |g| g.output :loss, g.mean("two" * g.input(:x, [nil, 2])) }
    end
  end

  def test_the_manifest_and_the_ruby_side_agree
    Torobi::Ops::REGISTRY.each_value do |spec|
      assert_includes Torobi::Shape::RULES, spec.shape_rule,
                      "op #{spec.name} has an unknown shape rule"
    end
    # And nothing in Shape that no op asks for: a rule with no op is a rule
    # nobody has checked.
    asked_for = Torobi::Ops::REGISTRY.each_value.map(&:shape_rule).uniq

    assert_empty Torobi::Shape::RULES - asked_for,
                 "Shape implements rules no op names"
    Torobi::Ops.handle_ops.each do |spec|
      assert Torobi::DSL::Handle.method_defined?(spec.name),
             "manifest marks #{spec.name} as a handle op, but Handle does not define it"
    end
  end
end
