# frozen_string_literal: true

require_relative "test_helper"
require "json"

# Qwen2 as Torobi describes it (docs/plan.md sections 15.49 and 15.52).
#
# Three claims, of three different kinds.
#
# **Structural**, against a checkpoint that exists: the graph declares
# exactly the parameters Qwen2.5-0.5B holds, by name and by shape.
# `test/oracle/qwen2.5-0.5b.json` records what it holds, read out of the
# published file's own header, so this runs on a machine that has never
# downloaded the 1GB underneath it.
#
# **Behavioural**, on a model small enough to differentiate by hand: the
# backward agrees with the forward, and the attention only looks
# backwards. The second is what makes a decoder a decoder, and it needs
# no reference implementation to check: a token that has not been read
# yet cannot have changed anything.
#
# **Numeric**, against the implementation everyone else is held to: the
# hidden states and the tokens it thinks come next, from transformers on
# the CPU in f32 (`rake oracle:qwen2_forward`). This one needs the
# weights, so it says so and skips where they are not; the other two run
# anywhere.
class Qwen2Test < Minitest::Test
  ORACLE = File.expand_path("oracle/qwen2.5-0.5b.json", __dir__)

  def oracle = @oracle ||= JSON.parse(File.read(ORACLE))

  def published = Torobi::Models::Qwen2.from_hash(oracle.fetch("config"))

  FORWARD = File.expand_path("oracle/qwen2.5-0.5b.forward.json", __dir__)
  SNAPSHOTS = File.expand_path("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots")

  # The weights, if this machine has them. Unlike the inventory above,
  # numbers cannot be compared without them.
  def checkpoint
    dir = ENV["QWEN2_5_0_5B"] || Dir[File.join(SNAPSHOTS, "*")].max_by { |d| File.mtime(d) }
    dir if dir && File.file?(File.join(dir.to_s, "model.safetensors"))
  end

  # The third kind of claim: the numbers, against the implementation
  # everyone else is held to (docs/plan.md 9.2).
  #
  # What this catches and the other two cannot: a rotary base off by a
  # zero, a norm on the wrong side of a residual, a head tied the wrong
  # way round. All of those differentiate perfectly and declare exactly
  # the right parameters.
  def test_the_numbers_agree_with_the_reference_implementation
    skip "no reference recorded (rake oracle:qwen2_forward)" unless File.exist?(FORWARD)
    dir = checkpoint
    skip "Qwen/Qwen2.5-0.5B is not in the cache (set QWEN2_5_0_5B)" unless dir

    reference = JSON.parse(File.read(FORWARD))

    assert_equal published.hidden_size, reference.fetch("hidden_size")

    reference.fetch("cases").each_with_index do |c, i|
      hidden, logits = run_case(dir, c.fetch("input_ids"))
      compare_hidden(hidden, c.fetch("hidden"), "case #{i}")
      compare_logits(logits, c.fetch("top_logits"), "case #{i}")
    end
  end

  # One case through Torobi: the hidden state at every position, and the
  # scores at the last one, which is the position a decoder is asked
  # about.
  def run_case(dir, ids)
    graph = Torobi::Models::Qwen2.causal_lm(published, seq: ids.size)
    config = Torobi::GraphConfig.new(models: { m: graph }, train: [])
    Torobi::Session.open(config, pretrained: { m: File.join(dir, "model.safetensors") }) do |s|
      s.tap("m.hidden", stat: :full)
      produced = s.forward(Torobi::Models::Qwen2.batch(published, [ids], seq: ids.size))
      [s.tapped.fetch("m.hidden").to_a.each_slice(published.hidden_size).to_a,
       produced.fetch("m.logits").to_a.last(published.vocab_size)]
    end
  end

  # Relative to the size of what is being compared: a hidden state deep
  # in a decoder is tens, and an absolute tolerance would be saying
  # something about this model rather than about the arithmetic.
  def compare_hidden(got, want, where)
    assert_equal want.size, got.size, "#{where}: positions"
    apart = got.flatten.zip(want.flatten).map { |a, b| (a - b).abs }.max
    scale = want.flatten.map(&:abs).max
    margin("#{where} hidden", apart, scale)

    assert_operator apart / scale, :<, PARITY,
                    "#{where}: hidden states differ by #{apart} against #{scale}"
  end

  # The ids first: which tokens a model thinks come next is what it is
  # for, and two implementations that disagree about the order are not
  # the same model however close the numbers are.
  def compare_logits(got, want, where)
    mine = got.each_with_index.max_by(want.size) { |value, _| value }

    assert_equal want.map { |t| t.fetch("id") }.first, mine.first.last,
                 "#{where}: a different token comes next"
    assert_equal want.map { |t| t.fetch("id") }.sort, mine.map(&:last).sort,
                 "#{where}: a different set of tokens is likely"

    by_id = mine.to_h { |value, id| [id, value] }
    apart = want.map { |t| (by_id.fetch(t.fetch("id")) - t.fetch("value")).abs }.max
    scale = want.map { |t| t.fetch("value").abs }.max
    margin("#{where} logits", apart, scale)

    assert_operator apart / scale, :<, PARITY, "#{where}: scores differ by #{apart}"
  end

  # How much of the tolerance was actually used. A number somebody chose
  # is worth being able to look at: run with MARGIN=1 to see what it is
  # holding, and put what it says next to TOLERANCE.
  def margin(what, apart, scale)
    return unless ENV["MARGIN"]

    warn format("%-16s max|d| %.4g of %.4g, relative %.2e (tolerance %.0e)",
                what, apart, scale, apart / scale, PARITY)
  end

  # How far two implementations of the same arithmetic may be, in a
  # different order, over 24 layers.
  #
  # Relative, because Qwen2's hidden states are not small: the largest is
  # 207 in the first case, and an absolute tolerance would be a statement
  # about this model rather than about the arithmetic.
  #
  # Measured rather than guessed (MARGIN=1): the hidden states agree to
  # 1.4e-5 and 7.7e-6, the scores to 2.7e-6 and 3.6e-6. That is what two
  # f32 implementations of the same graph look like, and it leaves an
  # order of magnitude for another machine's kernels while still
  # refusing anything that is wired differently, which is off by orders
  # rather than by rounding.
  PARITY = 2e-4

  def test_the_recorded_config_is_the_one_this_builder_understands
    c = published

    assert_equal 896, c.hidden_size
    assert_equal 24, c.num_hidden_layers
    assert_equal 14, c.num_attention_heads
    # Grouped-query attention: fourteen query heads over two key heads, so
    # seven of them read the same keys.
    assert_equal 2, c.num_key_value_heads
    assert_equal 7, c.group
    assert_equal 64, c.head_dim
    assert c.tie_word_embeddings, "0.5B ties its output projection to its embedding"
  end

  # The claim that makes `pretrained:` work with no renaming.
  def test_the_graph_declares_exactly_what_the_checkpoint_holds
    graph = Torobi::Models::Qwen2.causal_lm(published, seq: 8)
    declared = graph.parameters.to_h { |spec| [spec.path, spec.shape] }
    held = oracle.fetch("parameters").transform_values { |t| t.fetch("shape") }

    assert_empty held.keys - declared.keys, "the checkpoint holds parameters this graph does not"
    assert_empty declared.keys - held.keys, "this graph declares parameters the checkpoint lacks"
    mismatched = declared.filter_map do |path, shape|
      "#{path}: declares #{shape.inspect}, holds #{held[path].inspect}" if held[path] != shape
    end

    assert_empty mismatched
    assert_equal 290, declared.size
  end

  # The tie is the reason there are 290 and not 291. A checkpoint that
  # ties holds no output projection, and a graph that declared one would
  # be asking for something no file has.
  def test_a_tied_model_declares_no_output_projection
    tied = Torobi::Models::Qwen2.causal_lm(published, seq: 4).parameters.map(&:path)

    refute_includes tied, "lm_head.weight"
    assert_includes tied, "model.embed_tokens.weight"

    apart = Torobi::Models::Qwen2.causal_lm(
      published.with(tie_word_embeddings: false), seq: 4
    ).parameters.to_h { |spec| [spec.path, spec.shape] }

    assert_equal [published.vocab_size, published.hidden_size], apart.fetch("lm_head.weight")
  end

  # --- a model small enough to answer for itself ---

  SEQ = 5

  # Two key heads under four query heads, so the grouping is exercised
  # rather than degenerate, and three layers.
  def small
    @small ||= Torobi::Models::Qwen2.from_hash(
      "vocab_size" => 11, "hidden_size" => 8, "intermediate_size" => 16,
      "num_hidden_layers" => 3, "num_attention_heads" => 4,
      "num_key_value_heads" => 2, "rms_norm_eps" => 1e-6, "rope_theta" => 10_000.0,
      "tie_word_embeddings" => true, "eos_token_id" => 10
    )
  end

  def model = @model ||= Torobi::Models::Qwen2.causal_lm(small, seq: SEQ)

  # What a language model is trained on: what it should have said next,
  # at the positions where there is a next. Written here rather than in
  # the model, because which positions count is the recipe's.
  def graph_config
    @graph_config ||= Torobi::GraphConfig.new(
      models: { m: model },
      objective: Torobi.objective(m: model) do |g|
        at = g.cross_entropy(g.from_model(:m, :logits),
                             g.from_batch(:targets, [nil, SEQ], dtype: :i32))
        kept = g.from_batch(:kept, [nil, SEQ])
        g.output :loss, g.sum(at * kept) / g.sum(kept)
      end
    )
  end

  def weights
    @weights ||= begin
      rng = Random.new(23)
      params = graph_config.parameters.to_h do |parameter|
        shape = parameter.spec.shape
        [parameter.qualified_path,
         { shape:, data: Array.new(shape.reduce(1, :*)) { rng.rand(-0.4..0.4) } }]
      end
      { params: }
    end
  end

  ROWS = [[3, 8, 5, 9, 2], [4, 6, 1]].freeze

  # The next token at each position, and which positions have one: the
  # last of a row does not, and neither does anything padded.
  def batch(rows = ROWS)
    lengths = rows.map(&:size)
    targets = rows.each_with_index.flat_map do |row, i|
      (row[1..] + Array.new(SEQ - lengths[i] + 1, small.pad_token_id)).first(SEQ)
    end
    kept = lengths.flat_map { |n| Array.new(SEQ) { |i| i < n - 1 ? 1.0 : 0.0 } }
    Torobi::Models::Qwen2.batch(small, rows, seq: SEQ)
                         .merge(targets: Torobi::TensorData.from_a([rows.size, SEQ], targets,
                                                                   dtype: :i32),
                                kept: Torobi::TensorData.from_a([rows.size, SEQ], kept))
  end

  def test_the_model_scores_every_position_over_the_whole_vocabulary
    logits = Torobi::Session.open(graph_config, weights:) do |s|
      s.forward(batch)["m.logits"]
    end

    assert_equal [ROWS.size, SEQ, small.vocab_size], logits.shape
  end

  # What makes a decoder a decoder, and it can be asked without a
  # reference: change the last token of a row and every position before
  # it must be untouched, because none of them has read it.
  def test_a_position_is_not_changed_by_what_comes_after_it
    later = [ROWS.first.dup, ROWS.last.dup]
    later.first[-1] = (later.first[-1] + 1) % small.vocab_size

    before, after = Torobi::Session.open(graph_config, weights:) do |s|
      [s.forward(batch)["m.logits"].to_a, s.forward(batch(later))["m.logits"].to_a]
    end

    width = small.vocab_size
    rows = ->(all) { all.each_slice(width).each_slice(SEQ).to_a }
    kept = rows.call(before).first.first(SEQ - 1).flatten
    moved = rows.call(after).first.first(SEQ - 1).flatten

    assert_equal kept, moved, "an earlier position read a later token"
    refute_equal rows.call(before).first.last, rows.call(after).first.last,
                 "and the position that did read it changed"
  end

  # For every parameter of a whole decoder, autodiff agrees with the
  # forward it is differentiating. This is what covers the pieces that
  # are new here: the grouped heads, the causal mask, the gated MLP, and
  # a table that is read at both ends of the model.
  def test_every_gradient_agrees_with_central_differences
    Torobi::Session.open(graph_config, weights:) do |s|
      analytic = s.gradients(batch)
      biggest = analytic.values.flat_map(&:to_a).map(&:abs).max

      assert_operator biggest, :>, 1e-3, "the model should have gradients worth checking"

      worst = { path: nil, at: nil, delta: 0.0 }
      s.parameter_paths.each do |path|
        held = s.fetch(path)
        sample(held.size).each do |i|
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

  # The embedding table is read twice, so its gradient is the sum of what
  # it gets at each end. Central differences do not care: they move the
  # number and see what the loss does, which is both uses at once.
  def test_the_tied_table_is_differentiated_at_both_ends
    Torobi::Session.open(graph_config, weights:) do |s|
      table = s.gradients(batch).fetch("m.model.embed_tokens.weight").to_a

      refute(table.all? { |v| v.abs < 1e-12 }, "a tied table gets a gradient")
    end
  end

  def test_a_row_longer_than_the_graph_is_refused
    e = assert_raises(Torobi::ConfigError) do
      Torobi::Models::Qwen2.batch(small, [Array.new(SEQ + 1, 1)], seq: SEQ)
    end

    assert_match(/has #{SEQ + 1} tokens and this graph was built for #{SEQ}/, e.message)
  end

  def test_a_batch_pads_on_the_right_with_the_configured_token
    ids = Torobi::Models::Qwen2.batch(small, ROWS, seq: SEQ).fetch(:input_ids)

    assert_equal [ROWS.size, SEQ], ids.shape
    assert_equal :i32, ids.dtype
    pad = small.pad_token_id

    assert_equal 10, pad, "no pad token is configured, so the end-of-text one stands in"
    assert_equal [3, 8, 5, 9, 2, 4, 6, 1, pad, pad], ids.to_a
  end

  TOLERANCE = 1e-4
  STEP = 3e-3

  private

  def central_difference(session, path, held, index, step: STEP)
    moved = lambda do |delta|
      data = held.to_a
      data[index] += delta
      session.put(path, Torobi::TensorData.from_a(held.shape, data))
      session.evaluate(batch)
    end
    (moved.call(step) - moved.call(-step)) / (2 * step)
  end

  def sample(size)
    return (0...size).to_a if size <= 3

    Array.new(3) { |i| (i * (size - 1)) / 2 }
  end
end
