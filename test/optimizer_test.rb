# frozen_string_literal: true

require_relative "test_helper"

# The optimizer is the engine's own (docs/plan.md section 5), so it is held
# to an oracle written here, in Ruby, from the update rule itself. Not a
# port of the engine's code: the arithmetic below is what AdamW says, and
# the engine has to agree with it step by step.
class OptimizerTest < Minitest::Test
  ROWS = 4

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  # loss = mean((x @ w' + b - y)^2), the simplest thing with two
  # differently-shaped parameters.
  def config
    model = Torobi.graph do |g|
      x = g.input :x, [nil, 2]
      y = g.input :y, [nil, 1]
      g.output :loss, g.mse(g.linear(x, 1, name: "l"), y)
    end
    Torobi::GraphConfig.new(models: { "m" => model })
  end

  def weights(w: [0.5, -0.25], b: [0.125])
    { params: { "m.l.weight" => { shape: [1, 2], data: w },
                "m.l.bias" => { shape: [1], data: b } } }
  end

  def batch(seed: 5)
    rng = Random.new(seed)
    xs = Array.new(ROWS) { [rng.rand(-1.0..1.0), rng.rand(-1.0..1.0)] }
    ys = Array.new(ROWS) { [rng.rand(-1.0..1.0)] }
    { x: { shape: [ROWS, 2], data: xs.flatten },
      y: { shape: [ROWS, 1], data: ys.flatten } }
  end

  # The gradients of that loss, in closed form: dL/dpred = 2(pred - y)/N,
  # dL/dw = dpred' @ x, dL/db = sum(dpred).
  def gradients(w, b, batch)
    xs = batch[:x][:data].each_slice(2).to_a
    ys = batch[:y][:data]
    n = ys.size.to_f
    dpred = xs.each_with_index.map do |row, i|
      2.0 * ((row[0] * w[0]) + (row[1] * w[1]) + b[0] - ys[i]) / n
    end
    dw = [xs.each_with_index.sum { |row, i| dpred[i] * row[0] },
          xs.each_with_index.sum { |row, i| dpred[i] * row[1] }]
    [dw, [dpred.sum]]
  end

  # AdamW, from the rule: moments, bias correction, then decoupled decay.
  class AdamWOracle
    def initialize(lr:, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0)
      @lr = lr
      @beta1 = beta1
      @beta2 = beta2
      @eps = eps
      @decay = weight_decay
      @t = 0
      @m = Hash.new { |h, k| h[k] = [] }
      @v = Hash.new { |h, k| h[k] = [] }
    end

    def step(params)
      @t += 1
      params.to_h do |name, (values, grads)|
        m = @m[name]
        v = @v[name]
        updated = values.each_with_index.map do |value, i|
          g = grads[i]
          m[i] = (@beta1 * (m[i] || 0.0)) + ((1 - @beta1) * g)
          v[i] = (@beta2 * (v[i] || 0.0)) + ((1 - @beta2) * g * g)
          m_hat = m[i] / (1 - (@beta1**@t))
          v_hat = v[i] / (1 - (@beta2**@t))
          next_value = value - (@lr * m_hat / (Math.sqrt(v_hat) + @eps))
          next_value - (@lr * @decay * value)
        end
        [name, updated]
      end
    end
  end

  def test_adamw_agrees_with_the_rule_step_by_step
    lr = 0.05
    w = [0.5, -0.25]
    b = [0.125]
    oracle = AdamWOracle.new(lr:)
    b_batch = batch

    Torobi::Session.open(config, weights: weights, optimizer: { kind: :adamw, lr: }) do |session|
      5.times do |step|
        dw, db = gradients(w, b, b_batch)
        expected = oracle.step("w" => [w, dw], "b" => [b, db])
        w = expected.fetch("w")
        b = expected.fetch("b")

        session.step!(b_batch)
        engine_w = session.fetch("m.l.weight").to_a
        engine_b = session.fetch("m.l.bias").to_a

        w.each_with_index do |value, i|
          assert_in_delta value, engine_w[i], 1e-5, "weight[#{i}] after step #{step + 1}"
        end
        assert_in_delta b[0], engine_b[0], 1e-5, "bias after step #{step + 1}"
      end
    end
  end

  # Bias correction makes the first steps distinctive: with beta1 = 0.9 the
  # first AdamW step moves a parameter by almost exactly lr, whatever the
  # gradient's size. A wrong t would show up here.
  def test_the_first_adamw_step_moves_by_the_learning_rate
    lr = 0.01
    Torobi::Session.open(config, weights: weights, optimizer: { kind: :adamw, lr: }) do |session|
      before = session.fetch("m.l.weight").to_a
      session.step!(batch)
      after = session.fetch("m.l.weight").to_a

      before.each_with_index do |value, i|
        assert_in_delta lr, (value - after[i]).abs, lr * 0.01,
                        "the first step should move by about lr"
      end
    end
  end

  def test_decoupled_weight_decay_pulls_toward_zero
    lr = 0.01
    plain = Torobi::Session.open(config, weights: weights, optimizer: { kind: :adamw, lr: }) do |s|
      3.times { s.step!(batch) }
      s.fetch("m.l.weight").to_a
    end
    decayed = Torobi::Session.open(config, weights: weights,
                                   optimizer: { kind: :adamw, lr:, weight_decay: 0.5 }) do |s|
      3.times { s.step!(batch) }
      s.fetch("m.l.weight").to_a
    end
    # w0 starts positive, w1 negative; decay moves both toward zero.
    assert_operator decayed[0], :<, plain[0]
    assert_operator decayed[1], :>, plain[1]
  end

  def test_sgd_is_the_gradient_step_it_claims_to_be
    lr = 0.1
    w = [0.5, -0.25]
    b = [0.125]
    b_batch = batch
    dw, db = gradients(w, b, b_batch)

    Torobi::Session.open(config, weights: weights, optimizer: { kind: :sgd, lr: }) do |session|
      session.step!(b_batch)
      engine_w = session.fetch("m.l.weight").to_a

      w.each_with_index do |value, i|
        assert_in_delta value - (lr * dw[i]), engine_w[i], 1e-6
      end
      assert_in_delta b[0] - (lr * db[0]), session.fetch("m.l.bias").to_a[0], 1e-6
    end
  end

  # Global-norm clipping: one factor for every gradient, so the direction
  # of the step is untouched and only its length is bounded. The norm read
  # back is the one before clipping, so a run can see how far past the cap
  # it went.
  def test_a_clip_bounds_the_global_norm_of_the_step
    lr = 0.1
    clip = 0.05
    w = [0.5, -0.25]
    b = [0.125]
    b_batch = batch
    dw, db = gradients(w, b, b_batch)
    norm = Math.sqrt((dw + db).sum { |g| g * g })

    Torobi::Session.open(config, weights: weights,
                         optimizer: { kind: :sgd, lr:, clip: }) do |session|
      session.step!(b_batch)

      assert_in_delta norm, session.grad_norm, 1e-5

      scale = lr * clip / norm
      engine_w = session.fetch("m.l.weight").to_a
      engine_b = session.fetch("m.l.bias").to_a

      w.each_with_index do |value, i|
        assert_in_delta value - (scale * dw[i]), engine_w[i], 1e-5
      end
      assert_in_delta b[0] - (scale * db[0]), engine_b[0], 1e-5

      # The step it took has a norm of exactly lr * clip, which is what
      # the cap asks for when the gradient was over it.
      delta = engine_w.each_with_index.map { |value, i| w[i] - value } + [b[0] - engine_b[0]]

      assert_in_delta lr * clip, Math.sqrt(delta.sum { |d| d * d }), 1e-5
    end
  end

  def test_the_gradient_norm_is_read_and_the_clip_is_a_recorded_knob
    io = StringIO.new
    Torobi::Session.open(config, weights: weights,
                         optimizer: { kind: :sgd, lr: 0.1 }, io:) do |session|
      assert_nil session.clip
      assert_predicate session.grad_norm, :nan?, "no step has been taken yet"
      session.step!(batch)

      assert_predicate session.grad_norm, :finite?

      session.adjust(clip: 1.0)

      assert_in_delta(1.0, session.clip)
      session.step!(batch)
      session.adjust(clip: nil)

      assert_nil session.clip
    end

    adjusted = Torobi::Journal.read(io.string).of(Torobi::Journal::Adjust)

    assert_equal [{ "clip" => 1.0 }, { "clip" => nil }], adjusted.map(&:knobs),
                 "the knob and its removal are both recorded, the removal as nil"
  end

  def test_the_parameter_norm_is_over_every_weight
    Torobi::Session.open(config, weights: weights, optimizer: { kind: :sgd, lr: 0.1 }) do |session|
      expected = Math.sqrt((0.5**2) + (0.25**2) + (0.125**2))

      assert_in_delta expected, session.param_norm, 1e-6

      before = session.param_norm
      session.step!(batch)

      refute_in_delta before, session.param_norm, 1e-9, "the step moved the weights"
    end
  end

  def test_an_unknown_optimizer_is_refused_by_name
    e = assert_raises(ArgumentError) do
      Torobi::Session.open(config, weights: weights, optimizer: { kind: :adagrad, lr: 0.1 })
    end
    assert_match(/unknown variant `adagrad`/, e.message)
    assert_match(/expected `sgd` or `adamw`/, e.message)
  end
end
