# frozen_string_literal: true

require_relative "test_helper"
require "stringio"
require "tmpdir"

# Hooks are the window's sugar (docs/plan.md 8.4): they fire in the window,
# do what the window does, and are fenced so a run stays readable and
# deterministic.
class HooksTest < Minitest::Test
  DIM = 2

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  def config
    model = Torobi.graph do |g|
      x = g.input :x, [nil, DIM]
      y = g.input :y, [nil, 1]
      g.output :loss, g.mse(g.linear(x, 1, name: "l"), y)
    end
    Torobi::GraphConfig.new(models: { "m" => model })
  end

  def weights(w: [0.0, 0.0])
    { params: { "m.l.weight" => { shape: [1, DIM], data: w },
                "m.l.bias" => { shape: [1], data: [0.0] } } }
  end

  # y = x0 + x1, so the loss falls to nothing and then stops falling: a
  # plateau a policy can find.
  def batch
    xs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]
    { x: { shape: [4, DIM], data: xs.flatten },
      y: { shape: [4, 1], data: xs.map { |a, b| a + b } } }
  end

  # One batch whose numbers overflow f32 when squared: the loss for this
  # step alone is not finite, while the parameters are perfectly usable.
  # The common shape of a NaN in practice, and the one the engine's guard
  # answers on its own.
  def poisoned_batch
    xs = [[1e30, 1e30], [1e30, 1e30], [1e30, 1e30], [1e30, 1e30]]
    { x: { shape: [4, DIM], data: xs.flatten },
      y: { shape: [4, 1], data: [0.0] * 4 } }
  end

  def test_hooks_fire_in_registration_order_and_at_their_interval
    order = []
    Torobi::Session.open(config, weights: weights) do |s|
      s.on(:step) { |e| order << [:first, e.step] }
      s.on(:step, every: 2) { |e| order << [:second, e.step] }
      s.on(:span_end) { |e| order << [:span, e.step] }
      s.run([batch] * 3)
    end
    assert_equal [[:first, 1], [:first, 2], [:second, 2], [:first, 3], [:span, 3]], order
  end

  def test_a_hook_sees_the_run_and_can_use_the_window
    seen = nil
    Torobi::Session.open(config, weights: weights) do |s|
      s.on(:step, every: 2) do |e|
        seen = e
        # A hook's capabilities are the window's: read, and turn knobs.
        e.session.adjust(lr: 0.05)
      end
      s.run([batch] * 2)
      assert_in_delta 0.05, s.lr
    end
    assert_equal 2, seen.step
    assert_equal 2, seen.history.size
    assert_predicate seen, :finite?
  end

  # The fence that keeps a run readable: a hook fires in the window, so it
  # cannot start a span of its own.
  def test_a_hook_cannot_run_a_span
    Torobi::Session.open(config, weights: weights) do |s|
      s.on(:step) { |e| e.session.run([batch]) }
      e = assert_raises(Torobi::Error) { s.run([batch]) }
      assert_match(/cannot run a span/, e.message)
    end
  end

  # An exception in a hook stops the span, which is safe because a window
  # is a point where the state is consistent.
  def test_an_exception_in_a_hook_stops_the_span_and_leaves_the_session_usable
    Torobi::Session.open(config, weights: weights) do |s|
      s.on(:step) { |e| raise "enough" if e.step == 2 }
      assert_raises(RuntimeError) { s.run([batch] * 10) }
      assert_equal 2, s.step
      s.step!(batch)
      assert_equal 3, s.step
    end
  end

  def test_unknown_events_and_missing_blocks_are_refused
    Torobi::Session.open(config, weights: weights) do |s|
      assert_raises(ArgumentError) { s.on(:whenever) { nil } }
      assert_raises(ArgumentError) { s.on(:step) }
      assert_raises(ArgumentError) { s.use("not a policy") }
    end
  end

  def test_progress_reports_what_it_is_given
    ticks = []
    Torobi::Session.open(config, weights: weights) do |s|
      s.use(Torobi::Policies::Progress.new { |step, loss| ticks << [step, loss] }, every: 2)
      s.run([batch] * 4)
    end
    assert_equal [2, 4], ticks.map(&:first)
    assert(ticks.all? { |_, loss| loss.finite? })
  end

  def test_best_checkpoint_keeps_the_best_the_run_has_been
    Dir.mktmpdir("torobi-hooks") do |dir|
      best = Torobi::Policies::BestCheckpoint.new(File.join(dir, "best"))
      Torobi::Session.open(config, weights: weights, optimizer: { kind: :sgd, lr: 0.3 }) do |s|
        s.use(best)
        s.run([batch] * 6)
      end
      assert_path_exists best.path
      assert_operator best.best, :<, 1.0
      manifest = JSON.parse(File.read(File.join(best.path, "manifest.json")))
      assert_operator manifest.fetch("step"), :<=, 6
    end
  end

  # This fit improves by less and less: after twenty steps each is worth
  # under a thousandth. `by` is what says how much improvement counts, so
  # a policy watching for progress this small finds the plateau.
  def test_lr_on_plateau_lowers_the_rate_when_the_loss_stops_falling
    Torobi::Session.open(config, weights: weights, optimizer: { kind: :sgd, lr: 0.3 }) do |s|
      s.use(Torobi::Policies::LrOnPlateau.new(factor: 0.5, patience: 3, by: 1e-3))
      s.run([batch] * 30)
      assert_operator s.lr, :<, 0.3, "the rate should have come down on the plateau"
      assert_operator s.lr, :>=, 1e-6
    end
  end

  def test_early_stopping_ends_the_run_where_it_stops_improving
    Torobi::Session.open(config, weights: weights, optimizer: { kind: :sgd, lr: 0.3 }) do |s|
      s.use(Torobi::Policies::EarlyStopping.new(patience: 3, by: 1e-3))
      assert_raises(Torobi::Policies::EarlyStopping::Stop) { s.run([batch] * 200) }
      assert_operator s.step, :<, 200, "it should have stopped early"
      assert_operator s.step, :>, 3
    end
  end

  # `Memory.limit=` is not a ceiling: MLX reads it as pressure on its
  # cache and keeps going (a 336 MB peak ran under a 67 MB limit). On
  # unified memory the thing that notices next is the machine, so the
  # ceiling has to be here.
  def test_a_memory_guard_stops_a_run_that_holds_too_much
    Torobi::Session.open(config, weights: weights, optimizer: { kind: :sgd, lr: 0.1 }) do |s|
      # One byte, which is under anything: a model this small holds 20 of
      # them, and a size worth guarding is not something a test can know.
      guard = Torobi::Policies::MemoryGuard.new(1)
      s.use(guard)
      e = assert_raises(Torobi::StepError) { s.repeat(batch, steps: 3) }

      assert_match(/past the/, e.message)
      assert_operator guard.seen, :>, 1
      assert_equal 1, s.step, "it should stop at the first step past the limit"
    end
  end

  # A limit nothing passes is a policy that does nothing, and what it
  # watches is the high-water mark rather than what is held between steps.
  # The two differ by more than a little: a step passes through several
  # times what it leaves behind, and the larger number is the one the
  # machine reacts to.
  def test_a_memory_guard_under_the_limit_is_quiet_and_watches_the_peak
    Torobi::Session.open(config, weights: weights, optimizer: { kind: :sgd, lr: 0.1 }) do |s|
      guard = Torobi::Policies::MemoryGuard.new(4 * 1024**3)
      s.use(guard)
      s.repeat(batch, steps: 3)

      assert_equal 3, s.step
      assert_operator guard.seen, :>, Torobi::Memory.active
      assert_equal Torobi::Memory.peak, guard.seen
    end
  end

  # What a policy does goes through the same knobs, so it lands in the
  # journal like anything else.
  def test_what_a_policy_adjusts_is_journalled
    io = StringIO.new
    Torobi::Session.open(config, weights: weights, io:, optimizer: { kind: :sgd, lr: 0.3 }) do |s|
      s.use(Torobi::Policies::LrOnPlateau.new(factor: 0.5, patience: 3, by: 1e-3))
      s.run([batch] * 30)
    end
    entries = Torobi::Journal.read(io.string)
    adjusts = entries.select { |e| e["kind"] == "adjust" && e.key?("lr") }
    refute_empty adjusts, "the policy's adjustment should be recorded"
    observes = entries.select { |e| e["kind"] == "observe" && e.key?("plateau_at") }
    refute_empty observes, "and so should what it decided on"
  end

  # The engine does not take a non-finite step, so one bad batch costs that
  # step and nothing else: the parameters are the ones the last good step
  # left, and the run carries on.
  def test_a_single_bad_batch_costs_one_step_and_nothing_else
    Torobi::Session.open(config, weights: weights(w: [0.5, 0.5]),
                         optimizer: { kind: :sgd, lr: 0.1 }) do |s|
      s.run([batch] * 3)
      before = s.fetch("m.l.weight").to_a

      s.use(Torobi::Policies::NaNGuard.new(lr_factor: 1.0))
      s.run([poisoned_batch])

      refute_predicate s.loss, :finite?, "the bad batch is reported"
      assert_equal before, s.fetch("m.l.weight").to_a, "and not applied"
      assert_equal 4, s.step, "the step still counts: its batch was consumed"

      s.run([batch] * 3)
      assert_predicate s.loss, :finite?, "the run carried straight on"
    end
  end

  # A divergence is the other case, and the guard alone does not answer it.
  # Once the parameters themselves overflow, no step is taken at all, so
  # lowering the rate moves nothing: the run is stuck rather than corrupt.
  # Giving up is the honest response, and the reason a rollback still has a
  # job (see the test below).
  def test_nan_guard_gives_up_when_lowering_the_rate_cannot_help
    Torobi::Session.open(config, weights: weights(w: [0.5, 0.5]),
                         optimizer: { kind: :sgd, lr: 50.0 }) do |s|
      s.use(Torobi::Policies::NaNGuard.new(lr_factor: 0.001, patience: 3))
      e = assert_raises(Torobi::StepError) { s.run([batch] * 40) }

      assert_match(/has not been finite for 3 steps/, e.message)
      assert_operator s.step, :>, 3, "it diverged first, then gave up"
    end
  end

  # A checkpoint is still allowed, for a caller who would rather resume
  # from a known place than from wherever the divergence started.
  def test_nan_guard_still_takes_a_checkpoint_when_it_is_given_one
    Dir.mktmpdir("torobi-hooks") do |dir|
      path = File.join(dir, "safe")
      Torobi::Session.open(config, weights: weights(w: [0.5, 0.5]),
                           optimizer: { kind: :sgd, lr: 50.0 }) do |s|
        s.checkpoint!(path)
        at_checkpoint = s.fetch("m.l.weight").to_a
        s.use(Torobi::Policies::NaNGuard.new(rollback: path, lr_factor: 0.001))
        s.run([batch] * 20)

        assert_predicate s.loss, :finite?
        refute_equal at_checkpoint, s.fetch("m.l.weight").to_a,
                     "it went back and then kept training"
      end
    end
  end

  # An evaluation reads the model without touching the run.
  def test_evaluate_reads_without_taking_a_step
    Torobi::Session.open(config, weights: weights) do |s|
      s.run([batch] * 3)
      step = s.step
      training_loss = s.loss
      before = s.fetch("m.l.weight").to_a

      seen = s.evaluate(batch)

      assert_predicate seen, :finite?
      assert_equal step, s.step
      assert_in_delta training_loss, s.loss, 0.0, "the training loss is untouched"
      assert_equal before, s.fetch("m.l.weight").to_a
    end
  end
end
