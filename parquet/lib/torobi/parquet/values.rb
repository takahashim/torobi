# frozen_string_literal: true

require_relative "error"

module Torobi
  module Parquet
    # The two ways a page holds numbers, and the one way it holds bytes.
    #
    # `Levels` is parquet's run-length/bit-packed hybrid, which appears
    # twice: as the definition levels that say which rows are null, and
    # as the dictionary indices a data page is made of. Same decoder,
    # different widths.
    #
    # `Plain` is the uncompressed spelling: a length and then the bytes,
    # or a fixed number of bytes per value. It is what a dictionary page
    # holds, and what a data page holds when the dictionary grew too
    # large to be worth it.
    module Levels
      # `count` values of `width` bits, from runs of two kinds.
      def self.hybrid(bytes, at, count, width)
        values = []
        while values.size < count
          header, at = varint(bytes, at)
          if header.nobits?(1)
            run = header >> 1
            value, at = fixed(bytes, at, width)
            run = [run, count - values.size].min
            values.concat(Array.new(run, value))
          else
            groups = header >> 1
            packed, at = packed(bytes, at, groups * 8, width)
            values.concat(packed.first([groups * 8, count - values.size].min))
          end
        end
        [values, at]
      end

      # A run of one repeated value, in as many whole bytes as its width
      # needs.
      def self.fixed(bytes, at, width)
        return [0, at] if width.zero?

        size = (width + 7) / 8
        value = 0
        size.times { |i| value |= bytes.getbyte(at + i) << (8 * i) }
        [value, at + size]
      end

      # Values packed end to end, least significant bit first, eight to a
      # group.
      def self.packed(bytes, at, count, width)
        values = Array.new(count)
        bit = 0
        count.times do |i|
          value = 0
          got = 0
          while got < width
            byte = bytes.getbyte(at + (bit >> 3)) or raise Error, "the page ends mid-value"
            shift = bit & 7
            take = [8 - shift, width - got].min
            value |= ((byte >> shift) & ((1 << take) - 1)) << got
            got += take
            bit += take
          end
          values[i] = value
        end
        [values, at + ((count * width) / 8)]
      end

      def self.varint(bytes, at)
        value = 0
        shift = 0
        loop do
          byte = bytes.getbyte(at)
          at += 1
          value |= (byte & 0x7f) << shift
          return [value, at] if byte < 0x80

          shift += 7
        end
      end
    end

    module Plain
      # Every value in a page, as the type says to read them.
      def self.all(bytes, at, count, type)
        case type
        when :byte_array then byte_arrays(bytes, at, count)
        when :int64 then [bytes.byteslice(at, count * 8).unpack("q<*"), at + (count * 8)]
        when :int32 then [bytes.byteslice(at, count * 4).unpack("l<*"), at + (count * 4)]
        when :double then [bytes.byteslice(at, count * 8).unpack("E*"), at + (count * 8)]
        when :float then [bytes.byteslice(at, count * 4).unpack("e*"), at + (count * 4)]
        else raise Error, "#{type} is not read here"
        end
      end

      # Each one a four-byte length and then that many bytes. Tagged
      # UTF-8 because that is what a parquet string is, and a caller that
      # wanted the bytes can ask for them back.
      def self.byte_arrays(bytes, at, count)
        values = Array.new(count)
        count.times do |i|
          length = bytes.byteslice(at, 4).unpack1("V")
          at += 4
          values[i] = bytes.byteslice(at, length).force_encoding(Encoding::UTF_8)
          at += length
        end
        [values, at]
      end
    end
  end
end
