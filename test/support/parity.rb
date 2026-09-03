# frozen_string_literal: true

# Holding a model description to the implementation everyone else is held
# to (docs/plan.md 9.2 and 15.52).
#
# Shared because the three artifacts and the way they are compared are
# the same for every family: an inventory that pins a commit, a reference
# recorded against it, and the weights themselves. What differs is the
# module that describes the architecture, and every one of them answers
# `from_hash`, `causal_lm` and `batch`, which is the whole of what this
# needs. A test that includes this says which by defining `described`.
module Parity
  # The weights, if this machine has them, in the layout the Hub's cache
  # uses. Unlike the inventories above, numbers cannot be compared
  # without them, and they are a gigabyte each.
  def weights_for(source)
    # "Qwen/Qwen2.5-0.5B" is QWEN2_5_0_5B to say by hand, and
    # models--Qwen--Qwen2.5-0.5B in the Hub's cache.
    named = File.basename(source).upcase.gsub(/[^A-Z0-9]/, "_")
    cached = File.expand_path("~/.cache/huggingface/hub/" \
                              "models--#{source.gsub("/", "--")}/snapshots/*")
    dir = ENV.fetch(named, nil) || Dir[cached].max_by { |d| File.mtime(d) }
    dir if dir && File.file?(File.join(dir.to_s, "model.safetensors"))
  end

  # The third kind of claim: the numbers, against the implementation
  # everyone else is held to (docs/plan.md 9.2).
  #
  # What this catches and the other two cannot: a rotary base off by a
  # zero, a norm on the wrong side of a residual, a head tied the wrong
  # way round. All of those differentiate perfectly and declare exactly
  # the right parameters.
  #
  # Whichever of the family this machine has. One is enough to hold the
  # description to something; a machine with both holds it to both, and
  # they exercise different halves of it (biases and a tie against
  # neither). What a machine with neither is missing is the weights and
  # a recorded reference: `rake oracle:qwen2_forward`, or
  # `rake oracle:sarashina_forward`.
  # Every model of this family this machine can compare, or none.
  def compare_against_references(models)
    compared = models.filter_map { |source, name| against_reference(source, name) }

    skip("no reference and no weights for #{models.keys.join(", ")}") if compared.empty?
    compared
  end

  def against_reference(source, name)
    forward = File.expand_path("oracle/#{name}.forward.json", __dir__)
    return nil unless File.exist?(forward)

    dir = weights_for(source)
    return nil unless dir

    config = described.from_hash(inventory(name).fetch("config"))
    reference = JSON.parse(File.read(forward))

    assert_equal config.hidden_size, reference.fetch("hidden_size"), source
    check_weights(inventory(name), reference, dir, source)
    reference.fetch("cases").each_with_index do |c, i|
      hidden, logits = run_case(config, dir, c.fetch("input_ids"))
      compare_hidden(hidden, c.fetch("hidden"), "#{source} case #{i}")
      compare_logits(logits, c.fetch("top_logits"), "#{source} case #{i}")
    end
    source
  end

  # That all three are about the same copy of the model.
  #
  # The inventory pins a commit, the reference records which one it ran,
  # and the Hub names a snapshot directory after it. A reference taken
  # against a checkpoint that has since been replaced would disagree
  # with these weights and say nothing about why, which is the failure
  # this is here to name. Checked where it can be: an artifact recorded
  # before this was written carries no revision, and a directory named
  # by hand is not a commit.
  def check_weights(inventory, reference, dir, source)
    pinned = inventory.fetch("revision")
    recorded = reference["revision"]
    if recorded
      assert_equal pinned, recorded,
                   "#{source}: the reference ran against another copy of the model"
    end
    return unless File.basename(dir).match?(/\A[0-9a-f]{40}\z/)

    assert_equal pinned, File.basename(dir),
                 "#{source}: these are not the weights the inventory was taken from"
  end

  # One case through Torobi: the hidden state at every position, and the
  # scores at the last one, which is the position a decoder is asked
  # about.
  def run_case(config, dir, ids)
    graph = described.causal_lm(config, seq: ids.size)
    run = Torobi::GraphConfig.new(models: { m: graph }, train: [])
    Torobi::Session.open(run, pretrained: { m: File.join(dir, "model.safetensors") }) do |s|
      s.tap("m.hidden", stat: :full)
      produced = s.forward(described.batch(config, [ids], seq: ids.size))
      [s.tapped.fetch("m.hidden").to_a.each_slice(config.hidden_size).to_a,
       produced.fetch("m.logits").to_a.last(config.vocab_size)]
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
end
