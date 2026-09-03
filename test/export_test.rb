# frozen_string_literal: true

require_relative "test_helper"
require "json"

# Exporting a trained model in HF / sentence-transformers format.
class ExportTest < Minitest::Test
  DIM = 4

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
    @dir = Dir.mktmpdir("torobi-export")
  end

  def teardown
    FileUtils.remove_entry(@dir) if @dir && File.exist?(@dir)
  end

  def config
    model = Torobi.graph do |g|
      x = g.input :x, [nil, DIM]
      y = g.input :y, [nil, 1]
      g.output :loss, g.mse(g.linear(x, 1, name: "l"), y)
    end
    Torobi::GraphConfig.new(models: { "student" => model })
  end

  def weights
    { params: { "student.l.weight" => { shape: [1, DIM], data: [0.3, -0.4, 0.1, 0.2] },
                "student.l.bias" => { shape: [1], data: [0.1] } } }
  end

  def batch
    xs = Array.new(6) { |i| Array.new(DIM) { |j| ((i + j) % 5) * 0.1 } }
    { x: { shape: [6, DIM], data: xs.flatten },
      y: { shape: [6, 1], data: xs.map { |row| [row.sum] }.flatten } }
  end

  def test_export_writes_hf_layout_files
    export_dir = File.join(@dir, "out")
    trained_weight = nil

    Torobi::Session.open(config, weights:) do |s|
      s.adjust(lr: 0.2)
      s.repeat(batch, steps: 10)
      trained_weight = s.fetch("student.l.weight").to_a
      s.export_model!(export_dir, pooling: :cls, pooling_dim: DIM,
                      config_json: '{"vocab_size":100}')
    end

    assert File.exist?(File.join(export_dir, "model.safetensors")), "model.safetensors missing"
    assert File.exist?(File.join(export_dir, "modules.json")), "modules.json missing"
    assert File.exist?(File.join(export_dir, "1_Pooling", "config.json")), "1_Pooling/config.json missing"
    assert File.exist?(File.join(export_dir, "config_sentence_transformers.json")), "config_sentence_transformers.json missing"
    assert File.exist?(File.join(export_dir, "config.json")), "config.json missing"

    tensors = read_safetensors(File.join(export_dir, "model.safetensors"))
    assert_equal %w[l.bias l.weight].sort, tensors.keys.sort
    assert_equal 1, tensors["l.bias"][:data].size
    assert_predicate tensors["l.bias"][:data].first, :finite?
    assert_equal trained_weight, tensors["l.weight"][:data]
    assert_equal "F32", tensors["l.weight"][:dtype]

    modules = JSON.parse(File.read(File.join(export_dir, "modules.json")))
    assert_equal 2, modules.size
    assert_equal "sentence_transformers.models.Transformer", modules[0]["type"]
    assert_equal "sentence_transformers.models.Pooling", modules[1]["type"]

    pooling = JSON.parse(File.read(File.join(export_dir, "1_Pooling", "config.json")))
    assert_equal DIM, pooling["word_embedding_dimension"]
    assert_equal true, pooling["pooling_mode_cls_token"]

    st = JSON.parse(File.read(File.join(export_dir, "config_sentence_transformers.json")))
    assert_equal({}, st["prompts"])

    assert_equal({ "vocab_size" => 100 }, JSON.parse(File.read(File.join(export_dir, "config.json"))))
  end

  def test_export_mean_pooling
    export_dir = File.join(@dir, "out")
    Torobi::Session.open(config, weights:) do |s|
      s.export_model!(export_dir, pooling: :mean, pooling_dim: DIM)
    end
    pooling = JSON.parse(File.read(File.join(export_dir, "1_Pooling", "config.json")))
    assert_equal true, pooling["pooling_mode_mean_tokens"]
    assert_equal false, pooling["pooling_mode_cls_token"]
  end

  def test_export_requires_pooling_dim
    e = assert_raises(ArgumentError) do
      Torobi::Session.open(config, weights:) do |s|
        s.export_model!(@dir, pooling: :cls)
      end
    end
    assert_match(/pooling_dim/, e.message)
  end

  def test_export_requires_model_name_when_ambiguous
    model = Torobi.graph do |g|
      x = g.input :x, [nil, DIM]
      g.output :logits, g.linear(x, 1, name: "l", bias: false)
    end
    objective = Torobi.graph do |g|
      g.output :loss, g.mean(g.input(:logits, [nil, 1], dtype: :f32))
    end
    config = Torobi::GraphConfig.new(
      models: { "a" => model, "b" => model },
      objective:,
      train: ["a"]
    )
    w = { params: { "a.l.weight" => { shape: [1, DIM], data: Array.new(DIM, 0.1) },
                    "b.l.weight" => { shape: [1, DIM], data: Array.new(DIM, 0.2) } } }

    e = assert_raises(ArgumentError) do
      Torobi::Session.open(config, weights: w) do |s|
        s.export_model!(@dir, pooling: :cls, pooling_dim: DIM)
      end
    end
    assert_match(/model: is required/, e.message)
  end

  def test_export_refuses_unknown_model
    e = assert_raises(Torobi::StepError) do
      Torobi::Session.open(config, weights:) do |s|
        s.export_model!(@dir, model: "teacher", pooling: :cls, pooling_dim: DIM)
      end
    end
    assert_match(/no model named.*teacher/, e.message)
  end

  def test_exported_weights_file_loads_as_pretrained
    export_dir = File.join(@dir, "out")
    config_with_prefix = Torobi.graph do |g|
      x = g.input :x, [nil, DIM]
      y = g.input :y, [nil, 1]
      g.scope("model") do
        g.output :loss, g.mse(g.linear(x, 1, name: "l"), y)
      end
    end
    config_with_prefix = Torobi::GraphConfig.new(models: { "student" => config_with_prefix })
    prefixed_weights = {
      params: {
        "student.model.l.weight" => { shape: [1, DIM], data: [0.3, -0.4, 0.1, 0.2] },
        "student.model.l.bias" => { shape: [1], data: [0.1] }
      }
    }

    trained_weight = nil
    Torobi::Session.open(config_with_prefix, weights: prefixed_weights) do |s|
      s.adjust(lr: 0.2)
      s.repeat(batch, steps: 10)
      trained_weight = s.fetch("student.model.l.weight").to_a
      s.export_model!(export_dir, pooling: :cls, pooling_dim: DIM)
    end

    # A fresh config whose model name is gone and whose paths are the
    # exported keys can load the file directly.
    bare = Torobi.graph do |g|
      x = g.input :x, [nil, DIM]
      y = g.input :y, [nil, 1]
      g.output :loss, g.mse(g.linear(x, 1, name: "model.l"), y)
    end
    bare_config = Torobi::GraphConfig.new(models: { "m" => bare })

    Torobi::Session.open(bare_config, pretrained: { m: File.join(export_dir, "model.safetensors") }) do |s|
      assert_equal trained_weight, s.fetch("m.model.l.weight").to_a
    end
  end

  private

  # A minimal safetensors reader, just enough to check the export layout.
  def read_safetensors(path)
    bytes = IO.binread(path)
    header_len = bytes[0, 8].unpack1("Q<")
    header = JSON.parse(bytes[8, header_len])
    base = 8 + header_len
    header.reject { |name, _| name == "__metadata__" }
           .transform_values do |entry|
      start, fin = entry["data_offsets"]
      {
        dtype: entry["dtype"],
        shape: entry["shape"],
        data: bytes[base + start, fin - start].unpack("e*")
      }
    end
  end
end
