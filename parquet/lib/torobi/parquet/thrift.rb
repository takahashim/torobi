# frozen_string_literal: true

require_relative "error"

module Torobi
  module Parquet
    # Thrift's compact protocol, as much of it as a parquet footer is.
    #
    # A parquet file describes itself in Thrift: the footer is a
    # `FileMetaData`, and every page begins with a `PageHeader`. Both are
    # read here without a schema, because the compact protocol carries
    # enough to walk a struct blind: every field says its number and its
    # type, and a struct ends with a zero byte.
    #
    # So a struct comes back as `{field_number => value}` and the reader
    # above names the numbers it wants. That is smaller than generating
    # code from an IDL and, more to the point, it means an unknown field
    # is skipped rather than fatal: parquet has added fields over the
    # years and will add more.
    module Thrift
      STOP = 0
      # The type nibble of a field header, and the element type of a list.
      BOOLEAN_TRUE = 1
      BOOLEAN_FALSE = 2
      BYTE = 3
      I16 = 4
      I32 = 5
      I64 = 6
      DOUBLE = 7
      BINARY = 8
      LIST = 9
      SET = 10
      MAP = 11
      STRUCT = 12

      # Reads one struct from `bytes` starting at `at`, and returns it
      # with the position after it.
      def self.struct(bytes, at)
        fields = {}
        id = 0
        loop do
          header = bytes.getbyte(at) or raise Error, "the struct ends before its stop byte"
          at += 1
          return [fields, at] if header == STOP

          delta = header >> 4
          type = header & 0x0f
          if delta.zero?
            id, at = zigzag(bytes, at)
          else
            id += delta
          end
          value, at = value_of(type, bytes, at, id)
          fields[id] = value
        end
      end

      # A value of the type the header named. Booleans are the odd ones:
      # in a field header the type nibble is the value, and in a list
      # they are a byte of their own.
      def self.value_of(type, bytes, at, id = nil)
        case type
        when BOOLEAN_TRUE then [true, at]
        when BOOLEAN_FALSE then [false, at]
        when BYTE then [bytes.getbyte(at), at + 1]
        when I16, I32, I64 then zigzag(bytes, at)
        when DOUBLE then [bytes.byteslice(at, 8).unpack1("E"), at + 8]
        when BINARY then binary(bytes, at)
        when LIST, SET then list(bytes, at)
        when STRUCT then struct(bytes, at)
        when MAP then map(bytes, at)
        else raise Error, "field #{id.inspect}: no thrift type #{type}"
        end
      end

      def self.binary(bytes, at)
        length, at = varint(bytes, at)
        [bytes.byteslice(at, length), at + length]
      end

      def self.list(bytes, at)
        header = bytes.getbyte(at)
        at += 1
        size = header >> 4
        type = header & 0x0f
        size, at = varint(bytes, at) if size == 15
        values = Array.new(size)
        size.times do |i|
          values[i], at = value_of(type, bytes, at)
        end
        [values, at]
      end

      def self.map(bytes, at)
        size, at = varint(bytes, at)
        return [{}, at] if size.zero?

        types = bytes.getbyte(at)
        at += 1
        pairs = {}
        size.times do
          key, at = value_of(types >> 4, bytes, at)
          value, at = value_of(types & 0x0f, bytes, at)
          pairs[key] = value
        end
        [pairs, at]
      end

      # An unsigned base-128 varint, low group first.
      def self.varint(bytes, at)
        value = 0
        shift = 0
        loop do
          byte = bytes.getbyte(at) or raise Error, "a varint runs off the end"
          at += 1
          value |= (byte & 0x7f) << shift
          return [value, at] if byte < 0x80

          shift += 7
          raise Error, "a varint longer than 64 bits" if shift > 63
        end
      end

      # Signed integers are zigzagged so that small negatives stay small.
      def self.zigzag(bytes, at)
        value, at = varint(bytes, at)
        [(value >> 1) ^ -(value & 1), at]
      end
    end
  end
end
