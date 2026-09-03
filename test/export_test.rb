# frozen_string_literal: true

require_relative "test_helper"
require "json"

# Exporting a trained model as one somebody else can load.
#
# The claim being tested is not "files were written" but "the result is
# the source model with new weights": the tokenizer that turns text into
# ids, the transformer's own config, and the pooling as the source had it,
# including the parts of it this code knows nothing about.
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

  # A published model, in miniature: what a fine-tune starts from, down to
  # the pooling key that decides whether the task prefix is inside the
  # average (`include_prompt`), and the weights that must not travel.
  def source
    dir = File.join(@dir, "source")
    FileUtils.mkdir_p(File.join(dir, "1_Pooling"))
    File.write(File.join(dir, "config.json"), '{"hidden_size":4,"model_type":"modernbert"}')
    File.write(File.join(dir, "sentence_bert_config.json"),
               '{"max_seq_length":8192,"do_lower_case":false}')
    File.write(File.join(dir, "tokenizer.json"), '{"version":"1.0"}')
    File.write(File.join(dir, "tokenizer_config.json"), '{"model_max_length":8192}')
    File.write(File.join(dir, "special_tokens_map.json"), '{"pad_token":"<pad>"}')
    File.write(File.join(dir, "modules.json"), "[]")
    File.write(File.join(dir, "1_Pooling", "config.json"), JSON.generate(
                                                             "word_embedding_dimension" => DIM,
                                                             "pooling_mode_cls_token" => false,
                                                             "pooling_mode_mean_tokens" => true,
                                                             "include_prompt" => true
                                                           ))
    # The same weights before training, which the export must not carry.
    File.write(File.join(dir, "pytorch_model.bin"), "stale")
    File.write(File.join(dir, "README.md"), "# the model this no longer is")
    dir
  end

  def trained(export_dir, **options)
    weight = nil
    Torobi::Session.open(config, weights:) do |s|
      s.adjust(lr: 0.2)
      s.repeat(batch, steps: 10)
      weight = s.fetch("student.l.weight").to_a
      s.export_model!(export_dir, **options)
    end
    weight
  end

  def test_an_export_from_a_source_is_that_model_with_new_weights
    out = File.join(@dir, "out")
    weight = trained(out, from: source)

    %w[config.json sentence_bert_config.json tokenizer.json tokenizer_config.json
       special_tokens_map.json modules.json 1_Pooling/config.json
       model.safetensors].each do |name|
      assert_path_exists File.join(out, name), "#{name} is missing"
    end
    # What must not travel: the source's own weights, and its description
    # of a model this no longer is.
    refute_path_exists File.join(out, "pytorch_model.bin")
    refute_path_exists File.join(out, "README.md")

    tensors = read_safetensors(File.join(out, "model.safetensors"))

    assert_equal %w[l.bias l.weight], tensors.keys.sort
    assert_equal weight, tensors["l.weight"][:data]
    assert_equal "F32", tensors["l.weight"][:dtype]
  end

  # Not written by this code, so that keys it does not know about survive.
  def test_the_source_pooling_travels_whole
    out = File.join(@dir, "out")
    trained(out, from: source)
    pooling = JSON.parse(File.read(File.join(out, "1_Pooling", "config.json")))

    assert_equal true, pooling["include_prompt"], "a key this code does not know was lost"
    assert_equal true, pooling["pooling_mode_mean_tokens"]
    assert_equal DIM, pooling["word_embedding_dimension"]
  end

  # An export that pools differently from what it started as: the mode
  # changes and nothing else does.
  def test_pooling_can_be_overridden_without_losing_the_rest
    out = File.join(@dir, "out")
    trained(out, from: source, pooling: :cls)
    pooling = JSON.parse(File.read(File.join(out, "1_Pooling", "config.json")))

    assert_equal true, pooling["pooling_mode_cls_token"]
    assert_equal false, pooling["pooling_mode_mean_tokens"]
    assert_equal true, pooling["include_prompt"], "an override is not a rewrite"
  end

  # A loader reads this before the tensors and refuses a file whose format
  # it cannot name.
  def test_the_weights_say_what_format_they_are_in
    out = File.join(@dir, "out")
    trained(out, from: source)

    assert_equal({ "format" => "pt" }, read_metadata(File.join(out, "model.safetensors")))
  end

  # A wrong pooling dimension is not an error anywhere else: it makes a
  # pooling layer of the wrong shape, in a file that loads.
  def test_a_pooling_dimension_the_export_does_not_have_is_refused
    out = File.join(@dir, "out")
    e = assert_raises(ArgumentError) { trained(out, from: source, pooling_dim: DIM + 1) }

    assert_match(/pooling_dim says #{DIM + 1}/, e.message)
    assert_match(/not a width these weights have/, e.message)
    # Refused before anything of this code's was written: the weights and
    # what was carried are there, and the pooling is still the source's.
    assert_path_exists File.join(out, "model.safetensors")
    pooling = JSON.parse(File.read(File.join(out, "1_Pooling", "config.json")))

    assert_equal DIM, pooling["word_embedding_dimension"], "the carried config was rewritten"
  end

  # The same number catches the other way of being wrong: a source that is
  # not where these weights came from carries a tokenizer for another
  # vocabulary, and a directory that loads and is wrong is worse than one
  # that does not load.
  def test_a_source_that_belongs_to_another_model_is_refused
    elsewhere = File.join(@dir, "elsewhere")
    FileUtils.mkdir_p(elsewhere)
    File.write(File.join(elsewhere, "config.json"), '{"hidden_size":768}')
    File.write(File.join(elsewhere, "tokenizer.json"), '{"version":"1.0"}')

    e = assert_raises(ArgumentError) { trained(File.join(@dir, "out"), from: elsewhere) }

    assert_match(/config\.json says 768/, e.message)
    assert_match(/wrong `from:`/, e.message)
  end

  # Nothing to start from: the metadata this code can write, and no
  # tokenizer, which is the shape of a model trained from nothing.
  def test_an_export_with_no_source_writes_what_it_can
    out = File.join(@dir, "out")
    trained(out, pooling: :mean, pooling_dim: DIM)

    assert_path_exists File.join(out, "modules.json")
    assert_path_exists File.join(out, "config_sentence_transformers.json")
    refute_path_exists File.join(out, "tokenizer.json")
    pooling = JSON.parse(File.read(File.join(out, "1_Pooling", "config.json")))

    assert_equal true, pooling["pooling_mode_mean_tokens"]
  end

  def test_without_a_source_a_pooling_dimension_is_required
    e = assert_raises(ArgumentError) { trained(File.join(@dir, "out"), pooling: :mean) }

    assert_match(/pooling_dim/, e.message)
  end

  # The model name comes from what the run holds, which the engine settled
  # at open, rather than from anything written down beside it.
  def test_which_model_is_asked_of_the_run
    model = Torobi.graph do |g|
      x = g.input :x, [nil, DIM]
      g.output :logits, g.linear(x, 1, name: "l", bias: false)
    end
    objective = Torobi.graph do |g|
      g.output :loss, g.mean(g.input(:logits, [nil, 1], dtype: :f32))
    end
    two = Torobi::GraphConfig.new(models: { "a" => model, "b" => model },
                                  objective:, train: ["a"])
    w = { params: { "a.l.weight" => { shape: [1, DIM], data: Array.new(DIM, 0.1) },
                    "b.l.weight" => { shape: [1, DIM], data: Array.new(DIM, 0.2) } } }

    e = assert_raises(ArgumentError) do
      Torobi::Session.open(two, weights: w) do |s|
        s.export_model!(@dir, pooling: :cls, pooling_dim: DIM)
      end
    end
    assert_match(/model: is required/, e.message)
    assert_match(/\["a", "b"\]/, e.message)
  end

  def test_export_refuses_a_model_this_run_does_not_have
    e = assert_raises(Torobi::StepError) do
      Torobi::Session.open(config, weights:) do |s|
        s.export_model!(@dir, model: "teacher", pooling: :cls, pooling_dim: DIM)
      end
    end
    assert_match(/no model named.*teacher/, e.message)
  end

  def test_a_source_that_is_not_a_directory_is_refused
    e = assert_raises(ArgumentError) do
      trained(File.join(@dir, "out"), from: File.join(@dir, "nowhere"))
    end
    assert_match(/is not a directory/, e.message)
  end

  # The run's own record says what left and what came with it.
  def test_the_journal_holds_what_was_exported
    io = StringIO.new
    Torobi::Session.open(config, weights:, io:) do |s|
      s.export_model!(File.join(@dir, "out"), from: source)
    end
    note = io.string.lines.map { |line| JSON.parse(line) }
             .find { |entry| entry["event"] == "exported" }

    assert_equal "student", note["model"]
    assert_equal %w[l.bias l.weight], note["paths"].sort
    assert_includes note["carried"], "tokenizer.json"
  end

  # The paths are the published layout's, so the file loads straight back
  # into a run that names them.
  def test_the_exported_file_loads_as_pretrained
    out = File.join(@dir, "out")
    prefixed = Torobi.graph do |g|
      x = g.input :x, [nil, DIM]
      y = g.input :y, [nil, 1]
      g.scope("model") { g.output :loss, g.mse(g.linear(x, 1, name: "l"), y) }
    end
    with_prefix = Torobi::GraphConfig.new(models: { "student" => prefixed })
    w = { params: { "student.model.l.weight" => { shape: [1, DIM], data: [0.3, -0.4, 0.1, 0.2] },
                    "student.model.l.bias" => { shape: [1], data: [0.1] } } }

    weight = nil
    Torobi::Session.open(with_prefix, weights: w) do |s|
      s.adjust(lr: 0.2)
      s.repeat(batch, steps: 10)
      weight = s.fetch("student.model.l.weight").to_a
      s.export_model!(out, from: source)
    end

    bare = Torobi.graph do |g|
      x = g.input :x, [nil, DIM]
      y = g.input :y, [nil, 1]
      g.output :loss, g.mse(g.linear(x, 1, name: "model.l"), y)
    end

    Torobi::Session.open(Torobi::GraphConfig.new(models: { "m" => bare }),
                         pretrained: { m: File.join(out, "model.safetensors") }) do |s|
      assert_equal weight, s.fetch("m.model.l.weight").to_a
    end
  end

  private

  # A minimal safetensors reader, just enough to check the export layout.
  def read_safetensors(path)
    bytes = File.binread(path)
    header_len = bytes[0, 8].unpack1("Q<")
    header = JSON.parse(bytes[8, header_len])
    base = 8 + header_len
    header.except("__metadata__")
          .transform_values do |entry|
      start, fin = entry["data_offsets"]
      { dtype: entry["dtype"], shape: entry["shape"],
        data: bytes[base + start, fin - start].unpack("e*") }
    end
  end

  def read_metadata(path)
    bytes = File.binread(path, 65_536)
    JSON.parse(bytes[8, bytes[0, 8].unpack1("Q<")])["__metadata__"]
  end
end
