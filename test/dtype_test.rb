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
      Torobi::Batch.pack({ x: { shape: [1], data: [1], dtype: :i64 } })
    end
    assert_match(/dtype "i64" does not cross the boundary/, e.message)
  end

  def test_packing_round_trips_for_both_dtypes
    packed = Torobi::Batch.pack({
      a: { shape: [3], data: [1.5, -2.5, 0.0] },
      b: { shape: [3], data: [1, -2, 300], dtype: :i32 }
    })
    assert_equal %w[f32 i32], packed.values.map(&:first)
    assert_equal [1.5, -2.5, 0.0], Torobi::Batch.unpack(packed.fetch("a")[2])
    assert_equal [1, -2, 300], Torobi::Batch.unpack(packed.fetch("b")[2], dtype: :i32)
  end
end
