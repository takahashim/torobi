# frozen_string_literal: true

require_relative "test_helper"
require "json"

# A base held quantized (docs/qlora.md), as far as a graph can say it.
#
# Everything here runs without the engine: what a quantized linear
# declares, what the shape rule refuses, what the graph refuses to train,
# and how many bytes the real model becomes. What the engine does with
# the declaration (read a published 4-bit file, run the packed matmul,
# leave the base untouched through a step) is the engine's tests and
# `qlora_test.rb`.
class QuantizationTest < Minitest::Test
  DIM = 64

  def quant(**overrides) = Torobi::Quantization.new(bits: 4, group_size: 32, **overrides)

  # x -> linear -> linear, the second one quantized, as a graph a test
  # can look at.
  def graph(quantization = quant, adapter: nil, d_in: DIM, bias: true)
    Torobi.graph do |g|
      x = g.input(:x, [nil, d_in], dtype: :bf16)
      g.quantized(quantization) do
        g.adapting(adapter) do
          g.output :y, g.linear(x, 8, name: "proj", bias:)
        end
      end
    end
  end

  # --- the declaration ---

  def test_a_quantized_linear_is_three_frozen_parameters_and_a_packed_matmul
    g = graph
    by_path = g.parameters.to_h { |p| [p.path, p] }

    assert_equal %w[proj.weight proj.scales proj.biases proj.bias], by_path.keys
    assert_equal([:u32, [8, DIM * 4 / 32], false], by_path["proj.weight"].then do |p|
      [p.dtype, p.shape, p.trainable]
    end)
    assert_equal([:bf16, [8, DIM / 32], false], by_path["proj.scales"].then do |p|
      [p.dtype, p.shape, p.trainable]
    end)
    assert_equal([:bf16, [8, DIM / 32], false], by_path["proj.biases"].then do |p|
      [p.dtype, p.shape, p.trainable]
    end)
    assert by_path["proj.bias"].trainable, "the bias is not quantized and trains as before"

    matmul = g.nodes.find { |n| n.op == "quantized_matmul" }

    refute_nil matmul
    assert_equal({ "group_size" => 32, "bits" => 4, "mode" => "affine" }, matmul.attributes)
    assert_equal [[nil, 8], :bf16], [matmul.shape, matmul.dtype]
    assert_empty g.nodes.select { |n|
      n.op == "matmul"
    }, "the plain matmul is replaced, not added to"
  end

  def test_a_linear_the_quantization_does_not_name_is_left_as_it_was
    g = graph(quant(on: %w[other]))

    assert_equal %w[proj.weight proj.bias], g.parameters.map(&:path)
    assert_equal :bf16, g.parameters.first.dtype
    assert_equal(1, g.nodes.count { |n| n.op == "matmul" })
  end

  def test_an_adapter_inside_a_quantized_scope_trains_only_itself
    adapter = Torobi::LoRA.new(rank: 2, on: %w[proj])
    g = graph(adapter:)
    trainable = g.parameters.select(&:trainable).map(&:path)

    assert_equal %w[proj.lora_A.weight proj.lora_B.weight], trainable
    assert_equal :f32, g.parameters.find { |p| p.path == "proj.lora_A.weight" }.dtype,
                 "the adapter trains in f32, whatever the base is held in"
  end

  def test_the_scopes_nest_either_way_and_not_twice
    a = graph
    b = Torobi.graph do |g|
      x = g.input(:x, [nil, DIM], dtype: :bf16)
      g.adapting(nil) { g.quantized(quant) { g.output :y, g.linear(x, 8, name: "proj") } }
    end

    assert_equal a.to_h, b.to_h

    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph { |g| g.quantized(quant) { g.quantized(quant) { nil } } }
    end
    assert_match(/already in scope/, e.message)
    none = Torobi.graph do |g|
      x = g.input(:x, [nil, DIM], dtype: :bf16)
      g.quantized(nil) { g.output :y, g.linear(x, 8, name: "proj") }
    end

    assert_equal :bf16, none.parameters.first.dtype, "nil quantizes nothing"
  end

  # --- what is refused, and where ---

  def test_a_width_no_group_divides_is_refused_with_the_linears_name
    e = assert_raises(Torobi::ConfigError) { graph(quant(group_size: 64), d_in: 48) }

    assert_match(/linear "proj": d_in 48 is not a multiple of group_size 64/, e.message)
    assert_match(/no group size divides it/, e.message)

    e = assert_raises(Torobi::ConfigError) { graph(quant(group_size: 128), d_in: 96) }

    assert_match(/32 would/, e.message)
  end

  def test_the_quantization_itself_is_checked_where_it_is_written
    assert_match(/bits must be one of 2, 3, 4, 5, 6, 8/, assert_raises(Torobi::ConfigError) do
      quant(bits: 7)
    end.message)
    assert_match(/group_size must be one of 32, 64, 128/,
                 assert_raises(Torobi::ConfigError) { quant(group_size: 48) }.message)
    assert_match(/mode must be affine/, assert_raises(Torobi::ConfigError) do
      quant(mode: "mxfp4")
    end.message)
    assert_match(/names which linears/, assert_raises(Torobi::ConfigError) do
      quant(on: [])
    end.message)
  end

  def test_a_packed_weight_declared_trainable_is_refused_when_the_config_is_built
    g = Torobi.graph do |g|
      x = g.input(:x, [nil, DIM], dtype: :bf16)
      w = g.param("proj.weight", [8, DIM / 8], dtype: :u32, init: { "type" => "zeros" })
      s = g.param("proj.scales", [8, DIM / 32], dtype: :bf16, init: { "type" => "zeros" },
trainable: false)
      g.output :y,
               g.emit("quantized_matmul", inputs: [x, w, s],
attrs: { group_size: 32, bits: 4, mode: "affine" })
    end
    e = assert_raises(Torobi::ConfigError) { Torobi::GraphConfig.new(models: { m: g }, train: []) }

    assert_match(/"proj.weight" trainable, and it is u32/, e.message)
    assert_match(/Torobi::LoRA/, e.message)
  end

  def test_nothing_is_cast_to_u32
    e = assert_raises(Torobi::ConfigError) do
      Torobi.graph { |g| g.output :y, g.cast(g.input(:x, [2]), :u32) }
    end

    assert_match(/cannot cast to :u32/, e.message)
    assert_includes Torobi::IR::Dtype::ALL, :u32
    refute_includes Torobi::IR::Dtype::CASTABLE, :u32
  end

  # The shape rule, held to each of its refusals on an emission written by
  # hand, which is the only way to reach them: `linear` declares shapes
  # that fit.
  REFUSALS = {
    "a quantized weight is packed u32" => [[8, 8], :bf16, [8, 2], [8, 2], {}],
    "a packed weight for d_in 64 at 4 bits is [8, 8], got [8, 7]" => [[8, 7], :u32, [8, 2], [8, 2],
                                                                      {}],
    "scales must be [8, 2], got [8, 3]" => [[8, 8], :u32, [8, 3], [8, 2], {}],
    "biases must be [8, 2], got [7, 2]" => [[8, 8], :u32, [8, 2], [7, 2], {}],
    "d_in 64 is not a multiple of group_size 128" => [[8, 8], :u32, [8, 2], [8, 2],
                                                      { group_size: 128 }],
    "bits must be one of 2, 3, 4, 5, 6, 8, got 7" => [[8, 8], :u32, [8, 2], [8, 2], { bits: 7 }],
    "quantization mode must be affine, got \"mxfp4\"" => [[8, 8], :u32, [8, 2], [8, 2],
                                                          { mode: "mxfp4" }],
    "mixed dtypes bf16, f32" => [[8, 8], :u32, [8, 2], [8, 2], { scales_dtype: :f32 }]
  }.freeze

  def test_every_shape_refusal_says_what_it_refused
    REFUSALS.each do |fragment, (w_shape, w_dtype, s_shape, b_shape, attrs)|
      e = assert_raises(Torobi::ConfigError, "expected a refusal for #{fragment}") do
        Torobi.graph do |g|
          x = g.input(:x, [nil, DIM], dtype: :bf16)
          w = g.input(:w, w_shape, dtype: w_dtype)
          s = g.input(:s, s_shape, dtype: attrs.fetch(:scales_dtype, :bf16))
          b = g.input(:b, b_shape, dtype: :bf16)
          g.output :y, g.emit("quantized_matmul", inputs: [x, w, s, b],
                                                  attrs: { group_size: 32, bits: 4, mode: "affine" }
                                                    .merge(attrs.except(:scales_dtype)))
        end
      end
      assert_includes e.message, fragment
    end
  end

  def test_a_fresh_pattern_naming_a_packed_weight_is_refused
    skip "extension not compiled" unless defined?(Torobi::Session)

    config = Torobi::GraphConfig.new(models: { m: graph }, train: [])
    e = assert_raises(Torobi::ConfigError) do
      Torobi::Weights.of(pretrained: { m: "nowhere.safetensors" },
                         fresh: ["m.proj.*"]).check_fresh!(config)
    end

    assert_match(/"m.proj.weight", which is a packed quantized weight/, e.message)
    assert Torobi::Weights.of(pretrained: { m: "x" }, fresh: ["m.proj.bias"]).check_fresh!(config)
  end

  # --- what a checkpoint says ---

  def test_a_quantization_is_read_from_a_checkpoints_config_and_header
    raw = { "quantization" => { "group_size" => 64, "bits" => 4 } }

    assert_equal quant(group_size: 64, on: :linears), Torobi::Quantization.from_config(raw)
    assert_nil Torobi::Quantization.from_config({}), "a bf16 checkpoint says nothing"

    header = %w[model.embed_tokens.weight model.embed_tokens.scales
                model.layers.0.self_attn.q_proj.scales model.layers.1.self_attn.q_proj.scales
                model.layers.0.mlp.gate_proj.weight __metadata__].to_h { |name| [name, {}] }
    from_header = Torobi::Quantization.from_header(raw, header)

    assert_equal %w[embed_tokens q_proj], from_header.on
    assert from_header.wraps?("model.layers.3.self_attn.q_proj")
    refute from_header.wraps?("model.layers.0.mlp.gate_proj"), "the file holds this one as numbers"
  end

  def test_from_directory_reads_the_header_at_the_front_of_the_file
    Dir.mktmpdir do |dir|
      File.write(File.join(dir, "config.json"),
                 JSON.generate("quantization" => { "group_size" => 32, "bits" => 8 }))
      header = JSON.generate("a.scales" => { "dtype" => "BF16", "shape" => [1, 1],
                                             "data_offsets" => [0, 2] })
      length = [header.bytesize].pack("Q<")
      File.binwrite(File.join(dir, "model.safetensors"), "#{length}#{header}\0\0")

      q = Torobi::Quantization.from_directory(dir)

      assert_equal [8, 32, ["a"]], [q.bits, q.group_size, q.on]
    end
  end

  # --- the claim, on the model this is for ---

  def published
    Torobi::Models::Llama.from_hash(
      JSON.parse(File.read(File.expand_path("oracle/qwen2.5-0.5b.json", __dir__))).fetch("config")
    )
  end

  BYTES = { f32: 4, bf16: 2, u32: 4 }.freeze

  # The same description, under half the bytes: linears at 4.5 bits a
  # weight, the embedding (and so the tied head) still in bf16, which on
  # a 0.5B model is a quarter of what is left (docs/qlora.md section 1.2).
  def test_a_quantized_model_is_under_half_of_the_same_model_in_bf16
    held = lambda do |quantization|
      graph = Torobi::Models::Llama.causal_lm(published, seq: 8, dtype: :bf16, quantization:)
      graph.parameters.sum { |spec| spec.shape.reduce(1, :*) * BYTES.fetch(spec.dtype) }
    end
    bf16 = held.call(nil)
    quantized = held.call(Torobi::Quantization.new(bits: 4, group_size: 64))

    assert_in_delta 0.92, bf16 / (1024.0**3), 0.01
    assert_in_delta 0.44, quantized / (1024.0**3), 0.01,
                    "Qwen2.5-0.5B, 4 bit g64, embedding in bf16"
    assert_in_delta 2.1, bf16.to_f / quantized, 0.05
  end

  def test_the_real_model_declares_three_parameters_per_linear_and_one_embedding
    quantization = Torobi::Quantization.new(bits: 4, group_size: 64)
    graph = Torobi::Models::Llama.causal_lm(published, seq: 8, dtype: :bf16, quantization:)
    paths = graph.parameters.map(&:path)

    assert_equal 24 * 7, paths.count { |p|
      p.end_with?(".scales")
    }, "seven linears a layer, all quantized"
    assert_equal(paths.count { |p| p.end_with?(".scales") }, paths.count do |p|
      p.end_with?(".biases")
    end)
    assert_equal :u32, graph.parameters.find { |p|
      p.path == "model.layers.0.mlp.down_proj.weight"
    }.dtype
    assert_equal [896, 4864 * 4 / 32], graph.parameters.find { |p|
      p.path == "model.layers.0.mlp.down_proj.weight"
    }.shape
    assert_equal :bf16, graph.parameters.find { |p| p.path == "model.embed_tokens.weight" }.dtype
    refute_includes paths, "lm_head.weight", "tied: the head reads the (bf16) embedding"
    assert_equal 290 + (24 * 7 * 2), paths.size
  end

  # What mlx-community/Qwen2.5-0.5B-Instruct-4bit holds, by name, shape
  # and dtype, against what this description declares for it. The oracle
  # is the file's own header (`tools/inventory.rb mlx-community/...`),
  # so this needs no weights; it needs the inventory to have been
  # recorded, which needs the Hub once.
  def test_the_description_declares_what_the_published_4bit_file_holds
    path = File.expand_path("oracle/qwen2.5-0.5b-instruct-4bit.json", __dir__)
    unless File.exist?(path)
      skip "no inventory of the 4-bit file yet: " \
           "ruby tools/inventory.rb mlx-community/Qwen2.5-0.5B-Instruct-4bit #{path}"
    end

    inventory = JSON.parse(File.read(path))
    raw = inventory.fetch("config")
    quantization = Torobi::Quantization.from_header(raw, inventory.fetch("parameters"))
    graph = Torobi::Models::Llama.causal_lm(Torobi::Models::Llama.from_hash(raw), seq: 8,
dtype: :bf16, quantization:)
    declared = graph.parameters.to_h { |p| [p.path, [p.shape, p.dtype]] }
    held = inventory.fetch("parameters").transform_values { |t| [t["shape"], t["dtype"]] }

    # The embedding is the one the file holds packed and the graph holds
    # in bf16 (read as numbers, docs/qlora.md 3.4 (A')); everything else
    # matches name for name and shape for shape.
    packed_embedding = %w[model.embed_tokens.weight model.embed_tokens.scales
                          model.embed_tokens.biases]

    assert_equal (held.keys - packed_embedding).sort,
                 (declared.keys - ["model.embed_tokens.weight"]).sort
    (held.keys - packed_embedding).each do |name|
      assert_equal held[name][0], declared[name][0], name
      assert_equal held[name][1] == "U32" ? :u32 : :bf16, declared[name][1], name
    end
  end

  # The adapter is the one thing that trains, so on a bf16 model it is
  # held in f32, where an Adam step on one of its entries would otherwise
  # round away; the linear it adapts still hands on bf16, so the rest of
  # the model is as it was. On an f32 model there is nothing to cast.
  def test_on_a_bf16_model_the_adapter_is_f32_and_the_linear_is_still_bf16
    config = Torobi::Models::Llama.from_hash(SMALL)
    adapter = Torobi::LoRA.new(rank: 2, on: %w[q_proj v_proj])
    graph = Torobi::Models::Llama.causal_lm(config, seq: 4, adapter:, dtype: :bf16)
    by_path = graph.parameters.to_h { |p| [p.path, p] }

    assert_equal [:f32], by_path.values.select { |p| adapter.adapted?(p.path) }.map(&:dtype).uniq
    assert_equal :bf16, by_path.fetch("model.layers.0.self_attn.q_proj.weight").dtype
    q = graph.nodes.find { |n| n.name == "model.layers.0.self_attn.q_proj" }

    assert_equal :bf16, q.dtype, "what the adapted linear hands on"
    f32 = Torobi::Models::Llama.causal_lm(config, seq: 4, adapter:)

    assert_empty f32.nodes.select { |n| n.op == "cast" }, "an f32 model casts nothing"
  end

  # Gemma3 is the other decoder this library describes, and the one that
  # is least like Llama: four norms a layer, norms on q and k per head, a
  # head_dim of 256 wider than its hidden size over its heads, norms that
  # scale by `1 + w`. None of that is a linear, and every linear it has
  # divides by every group size, so the same quantization applies.
  def gemma
    Torobi::Models::Gemma3.from_hash(
      JSON.parse(File.read(File.expand_path("oracle/gemma-3-270m.json", __dir__))).fetch("config")
    )
  end

  def test_gemma3_holds_its_linears_quantized_and_its_adapter_in_f32
    config = gemma
    adapter = Torobi::LoRA.new(rank: 8,
                               on: %w[q_proj k_proj v_proj o_proj gate_proj up_proj down_proj])
    quantization = Torobi::Quantization.new(bits: 4, group_size: 64)
    graph = Torobi::Models::Gemma3.causal_lm(config, seq: 8, dtype: :bf16, adapter:, quantization:)
    by_path = graph.parameters.to_h { |p| [p.path, p] }

    assert_equal 18 * 7, by_path.keys.count { |p|
      p.end_with?(".scales")
    }, "seven linears a layer, all held packed"
    assert_equal [:u32], by_path.values.select { |p|
      p.path.end_with?(".weight") && p.dtype == :u32
    }
                                .map(&:dtype).uniq
    q = by_path.fetch("model.layers.0.self_attn.q_proj.weight")

    assert_equal [config.attention_size, config.hidden_size * 4 / 32], q.shape,
                 "q is [heads × head_dim, hidden] = [1024, 640], packed"
    assert_equal :bf16, by_path.fetch("model.embed_tokens.weight").dtype,
                 "the tied table stays numbers"
    assert_equal adapter.fresh(Torobi::GraphConfig.new(models: { m: graph }, train: [])).map { |p|
      p.delete_prefix("m.")
    }.sort,
                 graph.parameters.select(&:trainable).map(&:path).sort, "only the adapter trains"
    assert_equal [:f32], graph.parameters.select { |p| adapter.adapted?(p.path) }.map(&:dtype).uniq
    assert_equal [:bf16], graph.nodes.select { |n| n.op == "quantized_matmul" }.map(&:dtype).uniq,
                 "every packed matmul hands on bf16"
  end

  # The bytes, which on this model say something else than on Qwen: its
  # vocabulary of 262k is 63% of its parameters, and the embedding is not
  # quantized, so quantizing the linears saves a quarter rather than half.
  def test_gemma3_s_embedding_is_most_of_what_it_holds
    held = lambda do |quantization|
      graph = Torobi::Models::Gemma3.causal_lm(gemma, seq: 8, dtype: :bf16, quantization:)
      graph.parameters.sum { |spec| spec.shape.reduce(1, :*) * BYTES.fetch(spec.dtype) }
    end
    bf16 = held.call(nil)
    quantized = held.call(Torobi::Quantization.new(bits: 4, group_size: 64))

    assert_in_delta 0.50, bf16 / (1024.0**3), 0.02, "gemma-3-270m in bf16"
    assert_operator quantized.to_f / bf16, :>, 0.70,
                    "the embedding is not quantized, and it is most of it"
    assert_operator quantized, :<, bf16
  end

  # --- the whole of it, through the engine ---

  SMALL = { "vocab_size" => 11, "hidden_size" => 64, "intermediate_size" => 128,
            "num_hidden_layers" => 2, "num_attention_heads" => 4,
            "num_key_value_heads" => 2, "rms_norm_eps" => 1e-6, "rope_theta" => 10_000.0,
            "tie_word_embeddings" => true, "eos_token_id" => 10 }.freeze

  def safetensors(tensors)
    offset = 0
    header = tensors.to_h do |name, t|
      bytes = t[:data].size * 4
      entry = { "dtype" => "F32", "shape" => t[:shape], "data_offsets" => [offset, offset + bytes] }
      offset += bytes
      [name, entry]
    end
    json = JSON.generate(header)
    json += " " * ((8 - (json.bytesize % 8)) % 8)
    [json.bytesize].pack("Q<") + json + tensors.values.map { |t| t[:data].pack("e*") }.join
  end

  # What QLoRA is, end to end, on a model small enough to run anywhere:
  # a bf16 description with its linears quantized as an f32 file is read
  # (docs/qlora.md 3.4 (B)), an adapter drawn fresh beside them, and steps
  # taken. Every piece of this had been tested apart; together they found
  # that a fresh adapter was drawn in f32 and left so in a bf16 model,
  # which the engine's dtype check refused on the first step (2026-11-09).
  def test_a_bf16_quantized_model_with_a_fresh_adapter_takes_steps
    skip "extension not compiled" unless defined?(Torobi::Session)

    config = Torobi::Models::Llama.from_hash(SMALL)
    seq = 4
    rng = Random.new(5)
    base = Torobi::Models::Llama.causal_lm(config, seq:).parameters.to_h do |spec|
      [spec.path, { shape: spec.shape, data: Array.new(spec.shape.reduce(1, :*)) do
        rng.rand(-0.4..0.4)
      end }]
    end
    adapter = Torobi::LoRA.new(rank: 2, alpha: 4, on: %w[q_proj v_proj])
    graph = Torobi::Models::Llama.causal_lm(config, seq:, dtype: :bf16, adapter:,
quantization: quant)
    run = Torobi::GraphConfig.new(
      models: { m: graph },
      objective: Torobi.objective(m: graph) do |g|
        logits = g.cast(g.from_model(:m, :logits), :f32)
        g.output :loss,
                 g.mean(g.cross_entropy(logits, g.from_batch(:targets, [nil, seq], dtype: :i32)))
      end
    )
    rows = [[3, 8, 5, 9], [4, 6, 1, 2]]
    batch = Torobi::Models::Llama.batch(config, rows, seq:).merge(
      targets: Torobi::TensorData.from_a([2, seq], rows.flat_map { |r| r[1..] + [10] }, dtype: :i32)
    )

    Dir.mktmpdir do |dir|
      file = File.join(dir, "model.safetensors")
      File.binwrite(file, safetensors(base))
      Torobi::Session.open(run, pretrained: { m: file }, fresh: adapter.fresh(run),
                                optimizer: { kind: :adamw, lr: 0.05 }) do |s|
        assert_equal adapter.fresh(run).sort, s.trainable.sort, "only the adapter trains"
        packed = "m.model.layers.0.self_attn.q_proj.weight"
        before = s.fetch(packed).to_a
        losses = Array.new(3) { s.step!(batch) }

        losses.each { |loss| assert_predicate loss, :finite?, losses.inspect }
        assert_operator losses.last, :<, losses.first, "the adapter learns"
        assert_equal before, s.fetch(packed).to_a, "the packed base does not move"
      end
    end
  end
end
