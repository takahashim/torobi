# frozen_string_literal: true

require_relative "test_helper"

# How a model graph and an objective graph are wired (docs/plan.md 5A.3):
# named outputs, references to them, batch fields, the trained set, and what
# the engine differentiates. The shape of a distillation, in miniature.
class WiringTest < Minitest::Test
  def student
    Torobi.graph do |g|
      x = g.input :x, [nil, 2]
      g.output :logits, g.linear(x, 1, name: "head")
    end
  end

  def teacher = student

  # student against a teacher held frozen: the objective reads both, and
  # only the student's parameters move.
  def distillation(train: [:student])
    s = student
    t = teacher
    objective = Torobi.objective(student: s, teacher: t) do |g|
      logits = g.from_model :student, :logits
      target = g.stop_gradient(g.from_model(:teacher, :logits))
      g.output :loss, g.mse(logits, target)
    end
    Torobi::GraphConfig.new(models: { student: s, teacher: t }, objective:, train:)
  end

  def test_an_objective_reads_named_model_outputs
    config = distillation
    inputs = config.objective.inputs
    assert_equal %w[student.logits teacher.logits], inputs.map(&:name)
    assert(inputs.all?(&:from_model?))
    assert_equal({ "model" => "student", "output" => "logits" }, inputs.first.source)
    # The shape came from the model's declaration, not from a guess.
    assert_equal [nil, 1], inputs.first.shape
    # stop_gradient, then mse lowered to sub / square / mean.
    assert_equal %w[stop_gradient sub square mean], config.objective.nodes.map(&:op)
    assert_equal({ "loss" => "node:3" }, config.objective.outputs)
  end

  def test_parameters_are_namespaced_and_only_the_trained_are_differentiated
    config = distillation
    assert_equal ["student.head.weight", "student.head.bias",
                  "teacher.head.weight", "teacher.head.bias"],
                 config.parameters.map(&:qualified_path)
    assert_equal [0, 1], config.argnums, "only the student's parameters"

    both = distillation(train: %i[student teacher])
    assert_equal [0, 1, 2, 3], both.argnums
  end

  def test_a_frozen_model_never_moves
    skip "extension not compiled" unless defined?(Torobi::Session)
    config = distillation
    weights = {
      params: {
        "student.head.weight" => { shape: [1, 2], data: [0.0, 0.0] },
        "student.head.bias" => { shape: [1], data: [0.0] },
        "teacher.head.weight" => { shape: [1, 2], data: [3.0, -2.0] },
        "teacher.head.bias" => { shape: [1], data: [1.0] }
      }
    }
    rng = Random.new(3)
    batches = Array.new(60) do
      rows = 8
      xs = Array.new(rows) { [rng.rand(-1.0..1.0), rng.rand(-1.0..1.0)] }
      { x: { shape: [rows, 2], data: xs.flatten } }
    end

    Torobi::Session.open(config, weights: weights) do |s|
      # The objective needs no batch field of its own here: it reads both
      # models, and they read x.
      assert_equal ["x"], s.input_names

      s.adjust(lr: 0.5)
      s.run(batches)

      # The student learned the teacher's function...
      student_w = s.fetch("student.head.weight")[:data]
      assert_in_delta 3.0, student_w[0], 5e-2
      assert_in_delta(-2.0, student_w[1], 5e-2)
      assert_in_delta 1.0, s.fetch("student.head.bias")[:data][0], 5e-2

      # ...and the teacher is untouched, to the bit.
      assert_equal [3.0, -2.0], s.fetch("teacher.head.weight")[:data]
      assert_equal [1.0], s.fetch("teacher.head.bias")[:data]

      # Gradients exist only for what is differentiated.
      assert_equal ["student.head.weight", "student.head.bias"],
                   s.gradients(batches.first).keys
    end
  end

  def test_the_two_halves_are_held_to_each_other
    # An output the model does not declare.
    e = assert_raises(Torobi::ConfigError) do
      Torobi.objective(student: student) { |g| g.from_model(:student, :hidden) }
    end
    assert_match(/no output "hidden"/, e.message)
    assert_match(/it has "logits"/, e.message)

    # A model the objective was not given.
    e = assert_raises(Torobi::ConfigError) do
      Torobi.objective(student: student) { |g| g.from_model(:teacher, :logits) }
    end
    assert_match(/no model named "teacher"/, e.message)

    # A model that vanishes between the objective and the config.
    s = student
    objective = Torobi.objective(student: s) do |g|
      g.output :loss, g.mean(g.from_model(:student, :logits))
    end
    e = assert_raises(Torobi::ConfigError) do
      Torobi::GraphConfig.new(models: { other: s }, objective:)
    end
    assert_match(/there is no model named "student"/, e.message)

    # A trained set naming something that is not there.
    e = assert_raises(Torobi::ConfigError) do
      Torobi::GraphConfig.new(models: { student: s }, train: [:nobody])
    end
    assert_match(/train names "nobody"/, e.message)
  end

  def test_stop_gradient_is_in_the_graph_where_it_was_asked_for
    config = distillation
    ops = config.objective.nodes.map(&:op)
    assert_includes ops, "stop_gradient"
    # It sits between the teacher's output and the loss.
    node = config.objective.nodes.find { |n| n.op == "stop_gradient" }
    assert_equal ["input:1"], node.inputs
  end
end
