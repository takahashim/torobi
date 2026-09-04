# frozen_string_literal: true

require_relative "error"

module Torobi
  module Parquet
    # Snappy, decompression only, in the raw form parquet stores.
    #
    # The format is a length and then a run of elements: a literal is
    # bytes to copy from the input, and a copy is bytes to repeat from
    # what has been written already. That is the whole of it, which is
    # why it is here rather than behind a gem: what a parquet reader
    # needs of snappy is eighty lines, and a native dependency for eighty
    # lines is a dependency for its own sake.
    #
    # It works in runs rather than bytes. Measured on this machine, a
    # loop that copies eight bytes at a time runs at 86 MB/s and one that
    # copies sixty-four at 612 MB/s, and real literals are tens of bytes,
    # so this is the difference between a reader that is fast enough and
    # one that is not.
    module Snappy
      # One loop rather than three methods.
      #
      # Ruby's method calls are tens of nanoseconds and this makes one
      # per element, of which a megabyte of Japanese text has about a
      # hundred thousand; the helpers this used to call cost a quarter of
      # the time. What is left is the two things that cannot be avoided:
      # a slice per element, and an append per element.
      def self.inflate(bytes)
        length, at = varint(bytes, 0)
        out = +""
        out.force_encoding(Encoding::BINARY)
        size = bytes.bytesize

        while at < size
          tag = bytes.getbyte(at)
          at += 1
          kind = tag & 0x03
          if kind.zero?
            count = tag >> 2
            if count >= 60
              extra = count - 59
              count = bytes.getbyte(at)
              count |= bytes.getbyte(at + 1) << 8 if extra > 1
              count |= bytes.getbyte(at + 2) << 16 if extra > 2
              count |= bytes.getbyte(at + 3) << 24 if extra > 3
              at += extra
            end
            count += 1
            out << bytes.byteslice(at, count)
            at += count
            next
          end

          if kind == 1
            count = 4 + ((tag >> 2) & 0x07)
            offset = ((tag >> 5) << 8) | bytes.getbyte(at)
            at += 1
          elsif kind == 2
            count = (tag >> 2) + 1
            offset = bytes.getbyte(at) | (bytes.getbyte(at + 1) << 8)
            at += 2
          else
            count = (tag >> 2) + 1
            offset = bytes.getbyte(at) | (bytes.getbyte(at + 1) << 8) |
                     (bytes.getbyte(at + 2) << 16) | (bytes.getbyte(at + 3) << 24)
            at += 4
          end
          held = out.bytesize
          raise Error, "a copy of #{offset} back from #{held}" if offset > held

          if offset >= count
            out << out.byteslice(held - offset, count)
          else
            # A copy that reaches into what it is writing: snappy's way
            # of saying "repeat this". Repeating the pattern is the same
            # answer and stays in bulk.
            pattern = out.byteslice(held - offset, offset)
            out << (pattern * ((count / offset) + 1)).byteslice(0, count)
          end
        end
        unless out.bytesize == length
          raise Error, "the block says #{length} bytes and holds #{out.bytesize}"
        end

        out
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
  end
end
