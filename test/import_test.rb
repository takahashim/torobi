# frozen_string_literal: true

require_relative "test_helper"
require "tmpdir"

# Starting a run from parameters someone else produced (docs/plan.md M3a).
#
# The file names its tensors the way the graph names them, which is how a
# checkpoint writes them. So the first thing this buys is the one a
# distillation actually needs: carry a model's parameters into a new run
# without carrying that run's optimizer state, counters or RNG.
class ImportTest < Minitest::Test
  DIM = 4

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
    @dir = Dir.mktmpdir("torobi-import")
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
    Torobi::GraphConfig.new(models: { "m" => model })
  end

  def weights(w: [0.3, -0.4, 0.1, 0.2])
    { params: { "m.l.weight" => { shape: [1, DIM], data: w },
                "m.l.bias" => { shape: [1], data: [0.1] } } }
  end

  def batch
    xs = Array.new(6) { |i| Array.new(DIM) { |j| ((i + j) % 5) * 0.1 } }
    { x: { shape: [6, DIM], data: xs.flatten },
      y: { shape: [6, 1], data: xs.map { |row| [row.sum] }.flatten } }
  end

  # A checkpoint's parameters.safetensors is a file whose tensors are named
  # the way the graph names them. So it is an import source, and this is
  # the shape a distillation starts from: the parameters, and nothing else.
  def test_a_run_starts_from_the_parameters_another_run_reached
    trained = Torobi::Session.open(config, weights: weights) do |s|
      s.adjust(lr: 0.2)
      s.repeat(batch, steps: 20)
      [s.step, s.fetch("m.l.weight")[:data], s.loss]
    end
    steps, reached, loss = trained
    checkpoint = File.join(@dir, "run")
    Torobi::Session.open(config, weights: weights) do |s|
      s.adjust(lr: 0.2)
      s.repeat(batch, steps: steps)
      s.checkpoint!(checkpoint)
    end
    file = File.join(checkpoint, "parameters.safetensors")

    Torobi::Session.open(config, weights_file: file) do |s|
      assert_equal reached, s.fetch("m.l.weight")[:data], "the parameters came over"
      # And nothing else did. This is an import, not a resume.
      assert_equal 0, s.step
      assert_predicate s.loss, :nan?
      # It is a usable starting point: one step and the loss is where the
      # first run left it, not where an untrained run would be.
      s.step!(batch)
      assert_in_delta loss, s.loss, loss.abs * 0.5 + 1e-6
    end
  end

  # A resume is the other thing, and it is still `restore`: the optimizer's
  # slots, the counters and the RNG come too.
  def test_importing_is_not_resuming
    checkpoint = File.join(@dir, "run")
    optimizer = { kind: :adamw, lr: 0.05 }
    Torobi::Session.open(config, weights: weights, optimizer:) do |s|
      s.repeat(batch, steps: 8)
      s.checkpoint!(checkpoint)
    end

    Torobi::Session.open(config, weights_file: File.join(checkpoint, "parameters.safetensors"),
                         optimizer:) do |s|
      assert_equal 0, s.step
    end
    Torobi::Session.open(config, weights: weights, optimizer:) do |s|
      s.restore(checkpoint)
      assert_equal 8, s.step
    end
  end

  def test_a_file_missing_a_parameter_says_which_and_why
    Torobi::Session.open(config, weights: weights) { |s| s.checkpoint!(File.join(@dir, "c")) }
    partial = File.join(@dir, "partial.safetensors")
    # A file that holds only one of the two the graph declares.
    Torobi::Session.open(config, weights: weights) do |s|
      one = s.fetch("m.l.bias")
      IO.binwrite(partial, safetensors({ "m.l.bias" => one }))
    end

    e = assert_raises(Torobi::StepError) do
      Torobi::Session.open(config, weights_file: partial)
    end
    assert_match(/m\.l\.weight/, e.message)
    assert_match(/named the way the graph names them/, e.message)
  end

  # A model published in bf16 is a perfectly good place to start an f32
  # run, and refusing it would refuse the point of importing. Converted,
  # and said so in the doc: `restore` is the other case and refuses,
  # because a resumed run has to be the same run.
  def test_a_file_in_another_precision_is_converted
    values = [0.5, -0.25, 0.125, 2.0]
    path = File.join(@dir, "half.safetensors")
    IO.binwrite(path, bf16_safetensors("m.l.weight" => { shape: [1, DIM], data: values },
                                       "m.l.bias" => { shape: [1], data: [0.75] }))

    Torobi::Session.open(config, weights_file: path) do |s|
      # These survive bf16 exactly (few enough mantissa bits), so the
      # comparison is about the conversion happening, not about rounding.
      assert_equal values, s.fetch("m.l.weight")[:data]
      assert_equal [0.75], s.fetch("m.l.bias")[:data]
      s.step!(batch)
      assert_predicate s.loss, :finite?
    end
  end

  def test_a_file_that_is_not_there_says_so
    e = assert_raises(Torobi::StepError) do
      Torobi::Session.open(config, weights_file: File.join(@dir, "nope.safetensors"))
    end
    assert_match(/no parameters at/, e.message)
  end

  def test_exactly_one_source_is_named
    e = assert_raises(ArgumentError) do
      Torobi::Session.open(config, weights: weights, weights_file: "x.safetensors")
    end
    assert_match(/exactly one place/, e.message)
  end

  private

  # bf16 is f32 with the low 16 mantissa bits dropped.
  def bf16_safetensors(tensors)
    safetensors(tensors, dtype: "BF16", width: 2) do |values|
      # The high half of each f32, little-endian, is the bf16.
      values.map { |v| [v].pack("e").unpack("S<2").last }.pack("S<*")
    end
  end

  # A minimal safetensors writer, so this test can build a file the engine
  # has not written.
  def safetensors(tensors, dtype: "F32", width: 4, &block)
    pack = block || ->(values) { values.pack("e*") }
    offset = 0
    header = tensors.to_h do |name, t|
      bytes = t[:data].size * width
      entry = { "dtype" => dtype, "shape" => t[:shape],
                "data_offsets" => [offset, offset + bytes] }
      offset += bytes
      [name, entry]
    end
    json = JSON.generate(header)
    json += " " * ((8 - (json.bytesize % 8)) % 8)
    body = tensors.values.map { |t| pack.call(t[:data]) }
    [json.bytesize].pack("Q<") + json + body.join
  end
end
