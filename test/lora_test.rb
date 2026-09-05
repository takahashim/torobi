# frozen_string_literal: true

require_relative "test_helper"
require "json"

# Low-rank adaptation (docs/plan.md section 15.50).
#
# A fine-tune that moves every weight needs four copies of the model in
# memory. LoRA trains a pair of small matrices beside each weight and
# leaves the weight alone, and the three things worth testing are the
# three things that make that true rather than merely plausible:
#
#   1. only the adapter is differentiated
#   2. an adapted model starts as exactly the model it adapts
#   3. training moves the adapter and leaves the base where it was
#
# The third is the one a mistake would hide behind: a base left trainable
# by an oversight trains perfectly well, and looks like a smaller
# fine-tune rather than like a bug.
class LoRATest < Minitest::Test
  SEQ = 4
  DIM = 8

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
    @dir = Dir.mktmpdir("torobi-lora")
  end

  def teardown
    FileUtils.remove_entry(@dir) if @dir && File.exist?(@dir)
  end

  def config
    @config ||= Torobi::Models::Llama.from_hash(
      "vocab_size" => 11, "hidden_size" => DIM, "intermediate_size" => 16,
      "num_hidden_layers" => 2, "num_attention_heads" => 4,
      "num_key_value_heads" => 2, "rms_norm_eps" => 1e-6, "rope_theta" => 10_000.0,
      "tie_word_embeddings" => true, "eos_token_id" => 10
    )
  end

  def adapter = @adapter ||= Torobi::LoRA.new(rank: 2, alpha: 4, on: %w[q_proj v_proj])

  def model(with: adapter)
    Torobi::Models::Llama.causal_lm(config, seq: SEQ, adapter: with)
  end

  def graph_config(graph)
    Torobi::GraphConfig.new(
      models: { m: graph },
      objective: Torobi.objective(m: graph) do |g|
        at = g.cross_entropy(g.from_model(:m, :logits),
                             g.from_batch(:targets, [nil, SEQ], dtype: :i32))
        g.output :loss, g.mean(at)
      end
    )
  end

  ROWS = [[3, 8, 5, 9], [4, 6, 1, 2]].freeze

  def batch
    Torobi::Models::Llama.batch(config, ROWS, seq: SEQ)
                         .merge(targets: Torobi::TensorData.from_a(
                           [ROWS.size, SEQ], ROWS.flat_map { |r| r[1..] + [config.pad_token_id] },
                           dtype: :i32
                         ))
  end

  # The base model's own parameters, as a file to start from. Written
  # rather than passed inline, because `fresh:` is half of what is being
  # tested: the adapter's matrices are in no checkpoint, and the run has
  # to build them from the initializers the graph declares.
  def pretrained
    @pretrained ||= begin
      rng = Random.new(5)
      path = File.join(@dir, "base.safetensors")
      tensors = model(with: nil).parameters.to_h do |spec|
        [spec.path, { shape: spec.shape,
                      data: Array.new(spec.shape.reduce(1, :*)) { rng.rand(-0.4..0.4) } }]
      end
      File.binwrite(path, safetensors(tensors))
      path
    end
  end

  def open_adapted(graph = model, &)
    session_config = graph_config(graph)
    Torobi::Session.open(session_config, pretrained: { m: pretrained },
                                         fresh: adapter.fresh(session_config),
                                         optimizer: { kind: :adamw, lr: 0.05 }, &)
  end

  # --- what is trained ---

  def test_only_the_adapter_is_differentiated
    graph = model
    declared = graph.parameters
    adapted = declared.select { |spec| adapter.adapted?(spec.path) }

    # Two matrices for each of two targets in each of two layers.
    assert_equal 8, adapted.size
    assert_equal adapted.map(&:path).sort, declared.select(&:trainable).map(&:path).sort,
                 "something other than the adapter is trainable"

    open_adapted(graph) do |s|
      assert_equal adapted.map { |spec| "m.#{spec.path}" }.sort, s.trainable.sort
    end
  end

  # On the model this is for rather than on the toy above: a rank of 2
  # over a width of 8 is not small, and the number that matters is the
  # one a real fine-tune would see.
  def test_the_adapter_is_a_small_share_of_what_it_adapts
    published = Torobi::Models::Llama.from_hash(
      JSON.parse(File.read(File.expand_path("oracle/qwen2.5-0.5b.json", __dir__))).fetch("config")
    )
    graph = Torobi::Models::Llama.causal_lm(
      published, seq: 8, adapter: Torobi::LoRA.new(rank: 8, alpha: 16, on: %w[q_proj v_proj])
    )
    counted = ->(specs) { specs.sum { |spec| spec.shape.reduce(1, :*) } }
    share = counted.call(graph.parameters.select(&:trainable)).to_f /
            counted.call(graph.parameters)

    assert_operator share, :<, 0.002, "0.5B behind an adapter of half a million"
    assert_equal 96, graph.parameters.count(&:trainable), "two matrices, two targets, 24 layers"
  end

  # --- where it starts ---

  def test_the_lower_half_of_the_product_starts_at_zero
    b = model.parameters.find { |spec| spec.path.end_with?("q_proj.lora_B.weight") }

    assert_equal({ "type" => "zeros" }, b.initializer,
                 "B is what makes an adapted model start as the model it adapts")
  end

  # And what that is for: the same ids, through both graphs, from the
  # same file. Not close but equal, because the adapter contributes a
  # product with zero.
  def test_an_adapted_model_starts_as_the_model_it_adapts
    plain = Torobi::Session.open(graph_config(model(with: nil)),
                                 pretrained: { m: pretrained }) { |s| s.forward(batch) }
    adapted = open_adapted { |s| s.forward(batch) }

    assert_equal plain["m.logits"].to_a, adapted["m.logits"].to_a
  end

  # --- what training moves ---

  # The claim LoRA is for. The base is not merely mostly unchanged: it is
  # the same bytes, because nothing ever asked for its gradient.
  def test_training_moves_the_adapter_and_leaves_the_base_alone
    watched = "m.model.layers.0.self_attn.q_proj.weight"
    trained = "m.model.layers.0.self_attn.q_proj.lora_B.weight"

    open_adapted do |s|
      base = s.fetch(watched).to_a
      before_b = s.fetch(trained).to_a
      first = s.evaluate(batch)
      s.repeat(batch, steps: 20)
      last = s.evaluate(batch)

      assert_equal base, s.fetch(watched).to_a, "the base weight moved"
      refute_equal before_b, s.fetch(trained).to_a, "the adapter did not"
      assert_operator last, :<, first, "loss #{first} -> #{last} is not training"
    end
  end

  # A tied embedding is read at both ends of the model, so if the block
  # left it trainable it would be the largest thing being trained.
  def test_the_tied_embedding_is_not_trained_either
    open_adapted do |s|
      table = "m.model.embed_tokens.weight"
      before = s.fetch(table).to_a
      s.repeat(batch, steps: 5)

      assert_equal before, s.fetch(table).to_a
      refute_includes s.trainable, table
    end
  end

  # --- the value object, and the block ---

  def test_an_adapter_of_nothing_changes_nothing
    assert_equal graph_config(model(with: nil)).digest,
                 graph_config(Torobi::Models::Llama.causal_lm(config, seq: SEQ)).digest
  end

  def test_what_an_adapter_names
    a = Torobi::LoRA.new(rank: 8, alpha: 16, on: %w[q_proj v_proj])

    assert_in_delta 2.0, a.scale
    assert a.wraps?("model.layers.0.self_attn.q_proj")
    refute a.wraps?("model.layers.0.self_attn.o_proj"), "a name it does not list"
    refute a.wraps?("q_projector"), "and not a name it is merely a prefix of"
    assert_equal 8, Torobi::LoRA.new(rank: 8, on: "q_proj").alpha,
                 "alpha defaults to the rank, which is a scale of one"
  end

  def test_an_adapter_that_adapts_nothing_is_refused
    assert_raises(Torobi::ConfigError) { Torobi::LoRA.new(rank: 4, on: []) }
    assert_raises(Torobi::ConfigError) { Torobi::LoRA.new(rank: 0, on: "q_proj") }
  end

  # --- every description that takes an adapter applies it ---

  # A description that takes `adapter:` and forgets to put it in scope
  # builds a graph that runs, trains, and is a full fine-tune: no adapter
  # is declared, and every base parameter stays trainable because nothing
  # narrowed it (`DSL::Builder#param`). That is not a smaller failure than
  # a wrong shape, and nothing else here would notice it, so every entry
  # point that takes one is asked rather than the one that happened to be
  # written first.
  def test_every_description_that_takes_an_adapter_puts_it_in_scope
    bert = Torobi::Models::ModernBERT.from_hash(
      "vocab_size" => 32, "hidden_size" => 8, "intermediate_size" => 16,
      "num_hidden_layers" => 2, "num_attention_heads" => 2
    )
    on = Torobi::LoRA.new(rank: 2, alpha: 4, on: %w[Wqkv q_proj])
    described = {
      "ModernBERT.classifier" => Torobi::Models::ModernBERT.classifier(bert, seq: 4, adapter: on),
      "ModernBERT.embedder" => Torobi::Models::ModernBERT.embedder(bert, seq: 4, adapter: on),
      "ModernBERT.towers" => Torobi::Models::ModernBERT.towers(
        bert, { queries: 2, documents: 4 }, adapter: on
      ),
      "Llama.causal_lm" => Torobi::Models::Llama.causal_lm(config, seq: SEQ, adapter: on),
      "Gemma3.causal_lm" => Torobi::Models::Gemma3.causal_lm(gemma, seq: SEQ, adapter: on)
    }

    described.each do |what, graph|
      declared = graph.parameters
      adapted = declared.select { |spec| spec.path.include?("lora_") }

      refute_empty adapted, "#{what}: adapter: named linears and none was adapted"
      assert_equal adapted.map(&:path).sort, declared.select(&:trainable).map(&:path).sort,
                   "#{what}: something other than the adapter is trainable"
    end
  end

  def gemma
    Torobi::Models::Gemma3.from_hash(
      "vocab_size" => 32, "hidden_size" => 8, "intermediate_size" => 16,
      "num_hidden_layers" => 2, "num_attention_heads" => 2, "num_key_value_heads" => 1,
      "head_dim" => 4, "layer_types" => %w[full_attention full_attention]
    )
  end

  private

  def safetensors(tensors)
    offset = 0
    header = tensors.to_h do |name, t|
      bytes = t[:data].size * 4
      entry = { "dtype" => "F32", "shape" => t[:shape], "data_offsets" => [offset, offset + bytes] }
      offset += bytes
      [name, entry]
    end
    json = JSON.generate(header)
    json += " " * ((8 - (json.bytesize % 8)) % 8)
    [json.bytesize].pack("Q<") + json + tensors.values.map { |t| t[:data].pack("e*") }.join
  end
end
