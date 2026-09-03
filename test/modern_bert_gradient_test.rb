# frozen_string_literal: true

require_relative "test_helper"

# The whole model's gradient, against the forward it is differentiating
# (docs/plan.md M3b).
#
# `block_test.rb` does this for one block. What the stack adds is
# everything a block does not reach: the embedding table, the alternation
# of global and local layers, the two masks, the pooled head. A backward
# that is wrong in any of those is wrong here and nowhere else.
#
# The model is tiny on purpose. Central differences cost two forwards per
# position checked, and a wrong backward is wrong at every size; what the
# size has to be is large enough to reach every kind of parameter and to
# have a local layer whose window actually cuts something off.
class ModernBertGradientTest < Minitest::Test
  SEQ = 6
  # Rows of different lengths, so padding is masked rather than merely
  # present, and the shorter one is shorter than the window.
  ROWS = [[3, 8, 5, 9, 2, 7], [4, 6, 1]].freeze

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  # Three layers: 0 is global, 1 and 2 are local, which is the alternation
  # a real configuration has. `local_attention: 4` makes the window two
  # positions either side, so at sequence 6 it excludes something.
  def config
    @config ||= Torobi::Models::ModernBERT.from_hash(
      "vocab_size" => 12, "hidden_size" => 8, "intermediate_size" => 16,
      "num_hidden_layers" => 3, "num_attention_heads" => 2, "local_attention" => 4,
      "global_attn_every_n_layers" => 3, "num_labels" => 1
    )
  end

  def encoder = @encoder ||= Torobi::Models::ModernBERT.classifier(config, seq: SEQ,
                                                                   encoder_prefix: "")

  # A score to regress towards, so the loss is the one a distillation
  # actually differentiates rather than a norm of the output.
  def graph_config
    @graph_config ||= Torobi::GraphConfig.new(
      models: { m: encoder },
      objective: Torobi.objective(m: encoder) { |g|
        g.output :loss, g.mse(g.from_model(:m, :logits), g.from_batch(:score, [nil, 1]))
      }
    )
  end

  def weights
    rng = Random.new(19)
    params = graph_config.parameters.to_h do |parameter|
      shape = parameter.spec.shape
      data = Array.new(shape.reduce(1, :*)) { rng.rand(-0.4..0.4) }
      [parameter.qualified_path, { shape:, data: }]
    end
    { params: }
  end

  def batch
    @batch ||= Torobi::Models::ModernBERT.batch(config, ROWS, seq: SEQ)
                                         .merge(score: Torobi::TensorData.from_a([2, 1],
                                                                                 [0.6, -0.3]))
  end

  def test_the_graph_reaches_every_kind_of_parameter
    paths = graph_config.parameters.map(&:qualified_path)

    assert_includes paths, "m.embeddings.tok_embeddings.weight"
    assert_includes paths, "m.layers.0.attn.Wqkv.weight"
    # Layer 0 has no attn_norm (the embedding norm stands in for it), and
    # the layers after it do.
    refute_includes paths, "m.layers.0.attn_norm.weight"
    assert_includes paths, "m.layers.1.attn_norm.weight"
    assert_includes paths, "m.final_norm.weight"
    assert_includes paths, "m.head.dense.weight"
    assert_includes paths, "m.classifier.weight"
  end

  # The claim: for every parameter of a whole ModernBERT, autodiff agrees
  # with the forward it is differentiating.
  def test_every_gradient_agrees_with_central_differences
    Torobi::Session.open(graph_config, weights: weights) do |s|
      analytic = s.gradients(batch)
      # Without this the comparison could pass on gradients that are all
      # zero, which would agree with anything.
      biggest = analytic.values.flat_map(&:to_a).map(&:abs).max

      assert_operator biggest, :>, 1e-3, "the model should have gradients worth checking"
      assert(analytic.all? { |_, g| g.to_a.any? { |v| v.abs > 1e-9 } },
             "every parameter should be reached by the backward pass")

      worst = { path: nil, at: nil, delta: 0.0 }
      s.parameter_paths.each do |path|
        held = s.fetch(path)
        positions = sample(held.size)
        positions.each do |i|
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

  # The other half of M3b's claim: the loop can actually fit. A handful of
  # rows, memorized. If the gradient is right and the optimizer applies
  # it, a model this size drives a four-row regression to nothing; if
  # anything in the chain is subtly wrong, the loss stalls instead.
  def test_a_tiny_dataset_is_overfit
    rows = [[3, 8, 5, 9, 2, 7], [4, 6, 1], [11, 10], [2, 2, 2, 2]]
    scores = [0.9, -0.4, 0.2, -0.8]
    data = Torobi::Models::ModernBERT.batch(config, rows, seq: SEQ)
                                     .merge(score: Torobi::TensorData.from_a([rows.size, 1],
                                                                             scores))
    Torobi::Session.open(graph_config, weights: weights,
                         optimizer: { kind: :adamw, lr: 0.02 }) do |s|
      before = s.evaluate(data)
      s.repeat(data, steps: 200)
      after = s.evaluate(data)

      assert_operator before, :>, 0.1, "the untrained model should be wrong to start with"
      # Memorizing four numbers is what overfitting means here, so the bar
      # is near zero rather than merely lower.
      assert_operator after, :<, 1e-4, "loss #{before} -> #{after} is not memorization"
    end
  end

  # Measured at 4.7e-6 across the model, where the largest gradient is
  # 0.20. That leaves an order of magnitude and still catches a backward
  # wrong by anything like the size of the gradients themselves.
  TOLERANCE = 1e-4

  # Swept rather than guessed. The disagreement falls as the square of the
  # step (4.9e-4 at 3e-2, 5.5e-5 at 1e-2, 1.4e-5 at 5e-3) until f32
  # rounding in the two forwards takes over and it rises again (1.8e-5 at
  # 1e-3). The bottom of that curve is here.
  STEP = 3e-3

  private

  # (loss(w + h) - loss(w - h)) / 2h. `evaluate` draws no randomness, which
  # is what makes the two halves comparable.
  def central_difference(session, path, held, index, step: STEP)
    moved = lambda do |delta|
      data = held.to_a
      data[index] += delta
      session.put(path, Torobi::TensorData.from_a(held.shape, data))
      session.evaluate(batch)
    end
    (moved.call(step) - moved.call(-step)) / (2 * step)
  end

  # Up to three positions, spread across the parameter. Every one would be
  # thousands of forwards, and a wrong backward is wrong everywhere.
  def sample(size)
    return (0...size).to_a if size <= 3

    Array.new(3) { |i| (i * (size - 1)) / 2 }
  end
end
