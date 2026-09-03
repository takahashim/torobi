# frozen_string_literal: true

require_relative "test_helper"
require "json"

# A model held in bf16 (docs/plan.md section 15.51).
#
# Published checkpoints are stored in bf16, and until now Torobi read
# them into f32: the graph declared f32, and the engine converts an
# imported tensor to what the graph declares. So a 1GB model took 2GB
# before anything had been computed with it.
#
# What makes the other precision usable is one op and one rule. The op
# is `cast`, because the loss is read as an f32 at the boundary and
# something has to say where the model stops being bf16. The rule is
# that the boundary carries f32: Ruby has no bf16 to put numbers in, so
# what crosses is the numbers, converted.
class Bf16Test < Minitest::Test
  SEQ = 4

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
    @dir = Dir.mktmpdir("torobi-bf16")
  end

  def teardown
    FileUtils.remove_entry(@dir) if @dir && File.exist?(@dir)
  end

  def published
    @published ||= Torobi::Models::Llama.from_hash(
      JSON.parse(File.read(File.expand_path("oracle/qwen2.5-0.5b.json", __dir__))).fetch("config")
    )
  end

  # The claim, on the model this is for: the same description, half the
  # bytes. 0.5B is the smallest published Qwen2 and the difference is
  # already a gigabyte.
  def test_a_bf16_model_is_half_of_the_same_model_in_f32
    held = lambda do |dtype|
      graph = Torobi::Models::Llama.causal_lm(published, seq: 8, dtype:)
      widths = graph.parameters.map(&:dtype).uniq

      assert_equal [dtype], widths, "every parameter follows the table's dtype"
      graph.parameters.sum { |spec| spec.shape.reduce(1, :*) * (dtype == :bf16 ? 2 : 4) }
    end

    assert_equal held.call(:f32), held.call(:bf16) * 2
    assert_in_delta 0.92, held.call(:bf16) / (1024.0**3), 0.01, "Qwen2.5-0.5B, in gibibytes"
  end

  # --- and it is really bf16, not a label ---

  SMALL = { "vocab_size" => 11, "hidden_size" => 8, "intermediate_size" => 16,
            "num_hidden_layers" => 2, "num_attention_heads" => 4,
            "num_key_value_heads" => 2, "rms_norm_eps" => 1e-6,
            "rope_theta" => 10_000.0, "tie_word_embeddings" => true,
            "eos_token_id" => 10 }.freeze

  def config = @config ||= Torobi::Models::Llama.from_hash(SMALL)

  def model(dtype: :bf16) = Torobi::Models::Llama.causal_lm(config, seq: SEQ, dtype:)

  def graph_config(graph)
    Torobi::GraphConfig.new(
      models: { m: graph },
      objective: Torobi.objective(m: graph) do |g|
        # Where the model stops being bf16. Without this the loss is
        # bf16 and the config refuses it.
        logits = g.cast(g.from_model(:m, :logits), :f32)
        targets = g.from_batch(:targets, [nil, SEQ], dtype: :i32)
        g.output :loss, g.mean(g.cross_entropy(logits, targets))
      end
    )
  end

  ROWS = [[3, 8, 5, 9], [4, 6, 1, 2]].freeze

  def batch
    Torobi::Models::Llama.batch(config, ROWS, seq: SEQ)
                         .merge(targets: Torobi::TensorData.from_a(
                           [ROWS.size, SEQ], ROWS.flat_map { |r| r[1..] + [10] }, dtype: :i32
                         ))
  end

  def weights(dtype: :bf16)
    rng = Random.new(3)
    params = model(dtype:).parameters.to_h do |spec|
      ["m.#{spec.path}",
       { shape: spec.shape, data: Array.new(spec.shape.reduce(1, :*)) { rng.rand(-0.4..0.4) } }]
    end
    { params: }
  end

  # A number f32 holds exactly and bf16 does not: eight bits of mantissa
  # round 0.1 to 0.100097656. If the parameter were quietly f32 this
  # would come back as it went in.
  def test_a_bf16_parameter_rounds_like_one
    graph = model
    values = weights
    path = "m.model.layers.0.self_attn.q_proj.weight"
    shape = graph.parameters.find { |s| s.path == path.delete_prefix("m.") }.shape
    values[:params][path] = { shape:, data: Array.new(shape.reduce(1, :*), 0.1) }

    got = Torobi::Session.open(graph_config(graph), weights: values) { |s| s.fetch(path).to_a }

    assert_in_delta 0.100097656, got.first, 1e-9, "0.1 as bf16, read back as f32"
    refute_in_delta 0.1, got.first, 1e-9, "and not 0.1, which bf16 cannot hold"
  end

  # And the run says so where it writes itself down.
  def test_a_checkpoint_of_a_bf16_run_records_bf16
    written = Torobi::Session.open(graph_config(model), weights:) do |s|
      s.checkpoint!(File.join(@dir, "ckpt"))
    end
    recorded = Torobi::Checkpoint.manifest(written).fetch("parameters")

    assert_equal ["bf16"], recorded.map { |p| p.fetch("dtype") }.uniq
  end

  # --- the seam ---

  def test_a_loss_that_is_not_f32_is_refused_where_it_is_declared
    graph = model
    e = assert_raises(Torobi::ConfigError) do
      Torobi::GraphConfig.new(
        models: { m: graph },
        objective: Torobi.objective(m: graph) do |g|
          targets = g.from_batch(:targets, [nil, SEQ], dtype: :i32)
          g.output :loss, g.mean(g.cross_entropy(g.from_model(:m, :logits), targets))
        end
      )
    end

    assert_match(/the loss must be f32, and this one is bf16/, e.message)
    assert_match(/g\.cast/, e.message)
  end

  # Casting to what something already is is not a step, so a model
  # written once builds the same graph in the precision it was written in.
  def test_a_cast_to_what_it_already_is_is_not_a_node
    plain = Torobi.graph do |g|
      x = g.input :x, [nil, 2]
      g.output :out, g.cast(x, :f32)
    end
    changed = Torobi.graph do |g|
      x = g.input :x, [nil, 2]
      g.output :out, g.cast(x, :bf16)
    end

    assert_empty plain.nodes, "an f32 input cast to f32 is the input"
    assert_equal 1, changed.nodes.size
    assert_equal :bf16, changed.nodes.first.dtype
  end

  # --- and it trains ---

  def test_a_bf16_model_takes_steps
    first, last = Torobi::Session.open(graph_config(model), weights:,
                                       optimizer: { kind: :adamw, lr: 0.02 }) do |s|
      before = s.evaluate(batch)
      s.repeat(batch, steps: 20)
      [before, s.evaluate(batch)]
    end

    assert_operator last, :<, first, "loss #{first} -> #{last} is not training"
  end

  # The same model, computed in two precisions: close, and not equal.
  # Close is what makes bf16 usable; not equal is what makes it worth
  # saying which one a run used.
  def test_bf16_answers_what_f32_answers_to_the_precision_it_has
    logits = %i[f32 bf16].map do |dtype|
      graph = model(dtype:)
      # The same numbers, rounded on the way in for the bf16 one.
      Torobi::Session.open(graph_config(graph), weights: weights(dtype: :f32)) do |s|
        s.forward(batch)["m.logits"].to_a
      end
    end

    apart = logits.first.zip(logits.last).map { |a, b| (a - b).abs }.max
    scale = logits.first.map(&:abs).max

    assert_operator apart, :>, 0.0, "bf16 is a different arithmetic"
    assert_operator apart / scale, :<, 0.05, "and the same model"
  end
end
