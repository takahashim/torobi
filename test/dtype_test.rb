# frozen_string_literal: true

require_relative "test_helper"

# The boundary carries a dtype, so a graph can read what an embedding
# reads. Before this, everything crossing was f32 and a token id could not
# be expressed - the gate in front of model import (docs/plan.md 5A.2).
class DtypeTest < Minitest::Test
  VOCAB = 16
  DIM = 4

  def setup
    skip "extension not compiled" unless defined?(Torobi::Session)
  end

  # ids -> embedding -> mean, which is a loss that depends on exactly the
  # rows the ids select.
  def config
    model = Torobi.graph do |g|
      ids = g.input :ids, [nil, nil], dtype: :i32
      g.output :loss, g.mean(g.embedding(ids, vocab: VOCAB, dim: DIM, name: "emb"))
    end
    Torobi::GraphConfig.new(models: { "m" => model })
  end

  # Row i of the table is i, so the mean over a batch of ids is the mean of
  # the ids themselves: a loss whose value we know in closed form.
  def weights
    table = (0...VOCAB).flat_map { |i| Array.new(DIM, i.to_f) }
    { params: { "m.emb.weight" => { shape: [VOCAB, DIM], data: table } } }
  end

  def ids_batch(ids)
    { ids: { shape: [1, ids.size], data: ids, dtype: :i32 } }
  end

  def test_a_graph_reads_i32_ids
    Torobi::Session.open(config, weights: weights, optimizer: { kind: :sgd, lr: 0.0 }) do |s|
      loss = s.step!(ids_batch([1, 2, 3]))

      assert_in_delta 2.0, loss, 1e-6, "the mean of rows 1, 2 and 3"

      loss = s.step!(ids_batch([0, 0, 15]))

      assert_in_delta 5.0, loss, 1e-6
    end
  end

  # The gradient reaches only the rows that were read, which is the whole
  # point of an embedding.
  def test_only_the_rows_that_were_read_receive_gradient
    Torobi::Session.open(config, weights: weights, optimizer: { kind: :sgd, lr: 1.0 }) do |s|
      s.step!(ids_batch([2, 2]))
      table = s.fetch("m.emb.weight").to_a.each_slice(DIM).to_a

      refute_equal [2.0] * DIM, table[2], "row 2 was read and should have moved"
      VOCAB.times do |i|
        next if i == 2

        assert_equal [i.to_f] * DIM, table[i], "row #{i} was not read"
      end
    end
  end

  def test_the_dtype_a_batch_gives_must_be_the_one_declared
    Torobi::Session.open(config, weights: weights) do |s|
      e = assert_raises(Torobi::StepError) do
        s.step!({ ids: { shape: [1, 2], data: [1.0, 2.0] } }) # f32 by default
      end
      assert_match(/given Float32, declared i32/, e.message)
    end
  end

  def test_an_unknown_dtype_is_refused_where_it_is_written
    e = assert_raises(ArgumentError) do
      Torobi::Batch.new({ x: { shape: [1], data: [1], dtype: :i64 } })
    end
    assert_match(/input x: dtype :i64 does not cross the boundary/, e.message)
  end

  def test_packing_round_trips_for_both_dtypes
    batch = Torobi::Batch.new({
      a: { shape: [3], data: [1.5, -2.5, 0.0] },
      b: { shape: [3], data: [1, -2, 300], dtype: :i32 }
    })

    assert_equal %w[f32 i32], batch.to_native.values.map(&:first)
    assert_equal [1.5, -2.5, 0.0], batch.fetch(:a).to_a
    assert_equal [1, -2, 300], batch.fetch(:b).to_a
  end

  # A Hash is checked where it is written, as a TensorData is: bytes that
  # do not fill the shape are refused here, not by the engine.
  def test_a_hash_whose_data_does_not_fill_its_shape_is_refused
    e = assert_raises(ArgumentError) { Torobi::Batch.new({ x: { shape: [2], data: [1.0] } }) }
    assert_match(/\[2\] of f32 wants 8 bytes, got 4/, e.message)
  end
end
