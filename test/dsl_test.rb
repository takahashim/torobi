# frozen_string_literal: true

require_relative "test_helper"

class DslTest < Minitest::Test
  def linear_model
    Torobi.graph do |g|
      x = g.input :x, [nil, 4]
      g.output g.linear(x, 2, name: "linear")
    end
  end

  def test_a_linear_model_lowers_to_named_parameters_and_primitives
    graph = linear_model
    assert_equal %w[linear.weight linear.bias], graph.parameters.map(&:path)
    assert_equal [2, 4], graph.parameters[0].shape   # PyTorch layout [out, in]
    assert_equal %w[parameter transpose matmul parameter add], graph.nodes.map(&:op)
    assert_equal [nil, 2], graph.nodes.last.shape
    assert_equal :f32, graph.nodes.last.dtype
  end

  def test_the_same_definition_yields_the_same_digest
    a = Torobi::GraphConfig.new(models: { "m" => linear_model })
    b = Torobi::GraphConfig.new(models: { "m" => linear_model })
    assert_equal a.digest, b.digest
  end

  def test_scopes_name_the_parameters_of_repeated_components
    graph = Torobi.graph do |g|
      h = g.input :h, [nil, nil, 8]
      2.times do |i|
        g.scope("layers.#{i}") { h = h + g.linear(h, 8, name: "ff", bias: false) }
      end
      g.output h
    end
    assert_equal %w[layers.0.ff.weight layers.1.ff.weight], graph.parameters.map(&:path)
  end

  def test_shape_errors_are_reported_at_build_time_with_the_node_named
    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph do |g|
        a = g.input :a, [nil, 3]
        b = g.input :b, [4, 5]
        g.output g.matmul(a, b)
      end
    end
    assert_match(/node 0 \(matmul\)/, e.message)
    assert_match(/dimension .* mismatch \(3 vs 4\)/, e.message)
  end

  def test_values_cannot_cross_graphs
    stray = nil
    Torobi.graph do |g|
      stray = g.input :x, [2]
      g.output stray.neg
    end
    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph { |g| g.output stray.gelu }
    end
    assert_match(/belongs to a different graph/, e.message)
  end

  def test_split_produces_slices_and_checks_divisibility
    graph = Torobi.graph do |g|
      x = g.input :x, [nil, 6]
      a, b, c = x.split(3, axis: -1)
      g.output a + b + c
    end
    slices = graph.nodes.select { |n| n.op == "slice" }
    assert_equal [0, 2, 4], slices.map { |n| n.attributes["start"] }
    assert_equal [nil, 2], slices.first.shape

    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph { |g| g.input(:x, [nil, 5]).split(2) }
    end
    assert_match(/does not divide/, e.message)
  end

  def test_embedding_gathers_by_i32_ids
    graph = Torobi.graph do |g|
      ids = g.input :ids, [nil, nil], dtype: :i32
      g.output g.embedding(ids, vocab: 100, dim: 16, name: "emb")
    end
    assert_equal [nil, nil, 16], graph.nodes.last.shape

    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph do |g|
        ids = g.input :ids, [nil], dtype: :f32
        g.output g.embedding(ids, vocab: 10, dim: 4, name: "emb")
      end
    end
    assert_match(/indices must be i32/, e.message)
  end

  def test_the_objective_vocabulary_reaches_a_scalar_loss
    graph = Torobi.graph do |g|
      s = g.input :student_logits, [nil]
      t = g.input :teacher_logits, [nil]
      labels = g.input :label_loss, [nil]
      g.output g.mean(g.mse(s, t) * 0.7 + g.mean(labels) * 0.3)
    end
    assert_equal [], graph.nodes.last.shape
  end

  def test_a_modernbert_shaped_block_builds_and_keeps_its_shape
    dim, heads = 8, 2
    graph = Torobi.graph do |g|
      h = g.input :h, [nil, nil, dim]
      g.scope "layers.0" do
        qkv = g.linear(h, dim * 3, name: "wqkv", bias: false)
        q, k, v = qkv.split(3, axis: -1)
        a = g.sdpa(q.rope(theta: 10_000), k.rope(theta: 10_000), v, scale: 1.0 / heads)
        h = h + g.linear(a, dim, name: "wo", bias: false)
        h = h + g.geglu(g.layer_norm(h, name: "mlp_norm"), dim * 2, name: "mlp")
      end
      g.output h
    end
    assert_equal [nil, nil, dim], graph.nodes.last.shape
    assert_includes graph.parameters.map(&:path), "layers.0.mlp.wi.weight"
    assert_includes graph.parameters.map(&:path), "layers.0.mlp_norm.weight"
  end

  def test_the_manifest_and_the_ruby_side_agree
    known_rules = %i[parameter same_as_input broadcast transpose slice reduce matmul take sdpa]
    Torobi::Ops::REGISTRY.each_value do |spec|
      assert_includes known_rules, spec.shape_rule, "op #{spec.name} has an unknown shape rule"
    end
    Torobi::Ops.handle_ops.each do |spec|
      assert Torobi::DSL::Handle.method_defined?(spec.name),
             "manifest marks #{spec.name} as a handle op, but Handle does not define it"
    end
  end
end
