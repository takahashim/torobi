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

  def test_hooks_fire_in_registration_order_and_at_their_interval
    order = []
    Torobi::Session.open(config, weights) do |s|
      s.on(:step) { |e| order << [:first, e.step] }
      s.on(:step, every: 2) { |e| order << [:second, e.step] }
      s.on(:span_end) { |e| order << [:span, e.step] }
      s.run([batch] * 3)
    end
    assert_equal [[:first, 1], [:first, 2], [:second, 2], [:first, 3], [:span, 3]], order
  end

  def test_a_hook_sees_the_run_and_can_use_the_window
    seen = nil
    Torobi::Session.open(config, weights) do |s|
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
    Torobi::Session.open(config, weights) do |s|
      s.on(:step) { |e| e.session.run([batch]) }
      e = assert_raises(Torobi::Error) { s.run([batch]) }
      assert_match(/cannot run a span/, e.message)
    end
  end

  # An exception in a hook stops the span, which is safe because a window
  # is a point where the state is consistent.
  def test_an_exception_in_a_hook_stops_the_span_and_leaves_the_session_usable
    Torobi::Session.open(config, weights) do |s|
      s.on(:step) { |e| raise "enough" if e.step == 2 }
      assert_raises(RuntimeError) { s.run([batch] * 10) }
      assert_equal 2, s.step
      s.step!(batch)
      assert_equal 3, s.step
    end
  end

  def test_unknown_events_and_missing_blocks_are_refused
    Torobi::Session.open(config, weights) do |s|
      assert_raises(ArgumentError) { s.on(:whenever) { nil } }
      assert_raises(ArgumentError) { s.on(:step) }
      assert_raises(ArgumentError) { s.use("not a policy") }
    end
  end

  def test_progress_reports_what_it_is_given
    ticks = []
    Torobi::Session.open(config, weights) do |s|
      s.use(Torobi::Policies::Progress.new { |step, loss| ticks << [step, loss] }, every: 2)
      s.run([batch] * 4)
    end
    assert_equal [2, 4], ticks.map(&:first)
    assert(ticks.all? { |_, loss| loss.finite? })
  end

  def test_best_checkpoint_keeps_the_best_the_run_has_been
    Dir.mktmpdir("torobi-hooks") do |dir|
      best = Torobi::Policies::BestCheckpoint.new(File.join(dir, "best"))
      Torobi::Session.open(config, weights, optimizer: { kind: :sgd, lr: 0.3 }) do |s|
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
    Torobi::Session.open(config, weights, optimizer: { kind: :sgd, lr: 0.3 }) do |s|
      s.use(Torobi::Policies::LrOnPlateau.new(factor: 0.5, patience: 3, by: 1e-3))
      s.run([batch] * 30)
      assert_operator s.lr, :<, 0.3, "the rate should have come down on the plateau"
      assert_operator s.lr, :>=, 1e-6
    end
  end

  def test_early_stopping_ends_the_run_where_it_stops_improving
    Torobi::Session.open(config, weights, optimizer: { kind: :sgd, lr: 0.3 }) do |s|
      s.use(Torobi::Policies::EarlyStopping.new(patience: 3, by: 1e-3))
      assert_raises(Torobi::Policies::EarlyStopping::Stop) { s.run([batch] * 200) }
      assert_operator s.step, :<, 200, "it should have stopped early"
      assert_operator s.step, :>, 3
    end
  end

  # What a policy does goes through the same knobs, so it lands in the
  # journal like anything else.
  def test_what_a_policy_adjusts_is_journalled
    io = StringIO.new
    Torobi::Session.open(config, weights, io:, optimizer: { kind: :sgd, lr: 0.3 }) do |s|
      s.use(Torobi::Policies::LrOnPlateau.new(factor: 0.5, patience: 3, by: 1e-3))
      s.run([batch] * 30)
    end
    entries = Torobi::Journal.read(io.string)
    adjusts = entries.select { |e| e["kind"] == "adjust" && e.key?("lr") }
    refute_empty adjusts, "the policy's adjustment should be recorded"
    observes = entries.select { |e| e["kind"] == "observe" && e.key?("plateau_at") }
    refute_empty observes, "and so should what it decided on"
  end

  def test_nan_guard_goes_back_to_the_last_checkpoint
    Dir.mktmpdir("torobi-hooks") do |dir|
      path = File.join(dir, "safe")
      # A rate this large diverges to NaN within a few steps.
      Torobi::Session.open(config, weights(w: [0.5, 0.5]),
                           optimizer: { kind: :sgd, lr: 50.0 }) do |s|
        s.checkpoint!(path)
        s.use(Torobi::Policies::NaNGuard.new(rollback: path, lr_factor: 0.001))
        s.run([batch] * 20)
        assert_predicate s.loss, :finite?, "the guard should have pulled it back"
        assert_operator s.lr, :<, 50.0
      end
    end
  end

  def test_a_nan_without_a_checkpoint_says_so
    Torobi::Session.open(config, weights(w: [0.5, 0.5]),
                         optimizer: { kind: :sgd, lr: 50.0 }) do |s|
      s.use(Torobi::Policies::NaNGuard.new)
      e = assert_raises(Torobi::StepError) { s.run([batch] * 20) }
      assert_match(/no checkpoint to go back to/, e.message)
    end
  end
end
