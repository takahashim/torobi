# frozen_string_literal: true

require_relative "test_helper"
require "tmpdir"
require "json"

# The promise of docs/plan.md section 11: a run stopped and resumed reaches
# the same place a continuous run would, and a checkpoint that does not
# belong is refused rather than absorbed.
class CheckpointTest < Minitest::Test
  STEPS = 12

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
    @dir = Dir.mktmpdir("torobi-checkpoint")
  end

  def teardown
    FileUtils.remove_entry(@dir) if @dir && File.exist?(@dir)
  end

  def config
    model = Torobi.graph do |g|
      x = g.input :x, [nil, 2]
      y = g.input :y, [nil, 1]
      g.output :loss, g.mse(g.linear(x, 1, name: "l"), y)
    end
    Torobi::GraphConfig.new(models: { "m" => model })
  end

  def weights
    { params: { "m.l.weight" => { shape: [1, 2], data: [0.3, -0.4] },
                "m.l.bias" => { shape: [1], data: [0.1] } } }
  end

  # The same batches every time, so the only variable is the interruption.
  def batches(count = STEPS)
    rng = Random.new(11)
    Array.new(count) do
      rows = 6
      xs = Array.new(rows) { [rng.rand(-1.0..1.0), rng.rand(-1.0..1.0)] }
      ys = xs.map { |a, b| [(2 * a) - b + 0.5] }
      { x: { shape: [rows, 2], data: xs.flatten },
        y: { shape: [rows, 1], data: ys.flatten } }
    end
  end

  def state_of(session)
    session.parameter_paths.to_h { |path| [path, session.fetch(path)[:data]] }
  end

  # The heart of it: with AdamW, whose moments and step count both matter,
  # stopping and resuming must reach exactly where not stopping would.
  def test_a_resumed_run_reaches_where_a_continuous_one_does
    optimizer = { kind: :adamw, lr: 0.05 }
    all = batches

    continuous = Torobi::Session.open(config, weights, optimizer:) do |s|
      s.run(all)
      state_of(s)
    end

    resumed = Torobi::Session.open(config, weights, optimizer:) do |s|
      s.run(all.first(5))
      s.checkpoint!(File.join(@dir, "half"))
      state_of(s)
    end
    assert_equal 5, Torobi::Session.open(config, weights, optimizer:) { |s|
      s.restore(File.join(@dir, "half")).step
    }

    final = Torobi::Session.open(config, weights, optimizer:) do |s|
      s.restore(File.join(@dir, "half"))
      s.run(all.drop(5))
      state_of(s)
    end

    refute_equal continuous, resumed, "the halfway state should differ from the end"
    continuous.each do |path, values|
      values.each_with_index do |value, i|
        assert_in_delta value, final.fetch(path)[i], 1e-6,
                        "#{path}[#{i}] after resuming"
      end
    end
    assert_equal STEPS, Torobi::Session.open(config, weights, optimizer:) { |s|
      s.restore(File.join(@dir, "half"))
      s.run(all.drop(5))
      s.step
    }
  end

  # Without the optimizer's slots, a resumed AdamW run takes a different
  # first step. This is what makes the test above meaningful.
  def test_the_optimizer_state_is_what_makes_resuming_exact
    optimizer = { kind: :adamw, lr: 0.05 }
    all = batches(6)
    path = File.join(@dir, "half")

    Torobi::Session.open(config, weights, optimizer:) do |s|
      s.run(all.first(3))
      s.checkpoint!(path)
    end
    with_state = Torobi::Session.open(config, weights, optimizer:) do |s|
      s.restore(path)
      s.run(all.drop(3))
      state_of(s)
    end

    # A run given the checkpoint's parameters but not its optimizer state.
    parameters = JSON.parse(File.read(File.join(path, "manifest.json")))
    assert_equal 3, parameters.fetch("step")
    fresh_weights = { params: Torobi::Session.open(config, weights, optimizer:) { |s|
      s.restore(path)
      s.parameter_paths.to_h { |p| [p, s.fetch(p)] }
    } }
    without_state = Torobi::Session.open(config, fresh_weights, optimizer:) do |s|
      s.run(all.drop(3))
      state_of(s)
    end

    refute_in_delta with_state.fetch("m.l.weight")[0],
                    without_state.fetch("m.l.weight")[0], 1e-6,
                    "resuming without the moments should not land in the same place"
  end

  def test_a_checkpoint_says_what_it_belongs_to
    Torobi::Session.open(config, weights, optimizer: { kind: :adamw, lr: 0.05 }) do |s|
      s.run(batches(2))
      s.checkpoint!(File.join(@dir, "c"))
    end
    manifest = JSON.parse(File.read(File.join(@dir, "c", "manifest.json")))

    assert_equal 1, manifest.fetch("schema_version")
    assert_equal config.digest, manifest.fetch("config_digest")
    assert_equal 2, manifest.fetch("step")
    assert_equal 2, manifest.fetch("optimizer_steps")
    assert_equal "adamw", manifest.dig("optimizer", "kind")
    assert_equal %w[m.l.weight m.l.bias], manifest.fetch("parameters").map { _1["path"] }
    assert_equal [[1, 2], [1]], manifest.fetch("parameters").map { _1["shape"] }
    assert(manifest.fetch("parameters").all? { _1["trained"] })
    refute_empty manifest.dig("build", "torobi_engine")

    assert_path_exists File.join(@dir, "c", "parameters.safetensors")
    assert_path_exists File.join(@dir, "c", "optimizer.safetensors")
  end

  # SGD has no slots, so it writes none; the checkpoint is still complete.
  def test_an_optimizer_without_slots_writes_none
    Torobi::Session.open(config, weights, optimizer: { kind: :sgd, lr: 0.1 }) do |s|
      s.run(batches(2))
      s.checkpoint!(File.join(@dir, "sgd"))
    end
    refute_path_exists File.join(@dir, "sgd", "optimizer.safetensors")

    Torobi::Session.open(config, weights, optimizer: { kind: :sgd, lr: 0.1 }) do |s|
      s.restore(File.join(@dir, "sgd"))
      assert_equal 2, s.step
    end
  end

  def test_what_does_not_belong_is_refused
    path = File.join(@dir, "c")
    optimizer = { kind: :adamw, lr: 0.05 }
    Torobi::Session.open(config, weights, optimizer:) do |s|
      s.run(batches(2))
      s.checkpoint!(path)
    end

    # Another description.
    other = Torobi::GraphConfig.new(models: { "m" => Torobi.graph do |g|
      x = g.input :x, [nil, 2]
      y = g.input :y, [nil, 1]
      g.output :loss, g.mse(g.linear(x, 1, name: "different"), y)
    end })
    other_weights = { params: { "m.different.weight" => { shape: [1, 2], data: [0.0, 0.0] },
                                "m.different.bias" => { shape: [1], data: [0.0] } } }
    e = assert_raises(Torobi::StepError) do
      Torobi::Session.open(other, other_weights, optimizer:) { |s| s.restore(path) }
    end
    assert_match(/belongs to another graph/, e.message)

    # Another optimizer.
    e = assert_raises(Torobi::StepError) do
      Torobi::Session.open(config, weights, optimizer: { kind: :sgd, lr: 0.05 }) do |s|
        s.restore(path)
      end
    end
    assert_match(/different optimizer/, e.message)

    # Nothing there at all.
    e = assert_raises(Torobi::StepError) do
      Torobi::Session.open(config, weights, optimizer:) { |s| s.restore(File.join(@dir, "nope")) }
    end
    assert_match(/manifest.json/, e.message)
  end

  # Written to one side and renamed, so an interrupted write cannot be
  # mistaken for a checkpoint.
  def test_writing_is_atomic
    path = File.join(@dir, "c")
    Torobi::Session.open(config, weights, optimizer: { kind: :adamw, lr: 0.05 }) do |s|
      s.run(batches(2))
      s.checkpoint!(path)
      refute_path_exists "#{path}.writing", "the staging directory should be gone"

      # Writing again over a complete checkpoint replaces it wholesale.
      s.run(batches(2))
      s.checkpoint!(path)
      assert_equal 4, JSON.parse(File.read(File.join(path, "manifest.json"))).fetch("step")
    end
  end
end
