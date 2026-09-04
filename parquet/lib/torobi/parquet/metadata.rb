# frozen_string_literal: true

require_relative "error"

require_relative "thrift"

module Torobi
  module Parquet
    # What a parquet file says about itself.
    #
    # The footer is at the end: the last eight bytes are its length and
    # the magic, and the magic is at the front as well. Everything else
    # here is naming the numbers Thrift handed back (`Thrift`), because
    # the format's own names are the ones its documentation uses and a
    # reader that renamed them would be one more thing to translate.
    #
    # What is refused is as much of the point as what is read. This
    # understands flat schemas of a few types, snappy or nothing, and
    # data pages of the first version; everything else is named and
    # refused, because a reader that guessed would hand back numbers
    # that are wrong rather than an error that is right.
    module Metadata
      MAGIC = "PAR1"

      # parquet's own enums, by the numbers it writes.
      TYPES = { 0 => :boolean, 1 => :int32, 2 => :int64, 3 => :int96,
                4 => :float, 5 => :double, 6 => :byte_array,
                7 => :fixed_len_byte_array }.freeze
      CODECS = { 0 => :uncompressed, 1 => :snappy, 2 => :gzip, 3 => :lzo,
                 4 => :brotli, 5 => :lz4, 6 => :zstd, 7 => :lz4_raw }.freeze
      ENCODINGS = { 0 => :plain, 2 => :plain_dictionary, 3 => :rle, 4 => :bit_packed,
                    5 => :delta_binary_packed, 6 => :delta_length_byte_array,
                    7 => :delta_byte_array, 8 => :rle_dictionary,
                    9 => :byte_stream_split }.freeze
      REPETITIONS = { 0 => :required, 1 => :optional, 2 => :repeated }.freeze
      PAGES = { 0 => :data, 1 => :index, 2 => :dictionary, 3 => :data_v2 }.freeze

      Column = Struct.new(:name, :type, :repetition, keyword_init: true)
      # `num_values` is parquet's own name for it, and is the one name
      # here that Struct does not already have.
      Chunk = Struct.new(:name, :type, :repetition, :codec, :encodings, :num_values,
                         :compressed, :uncompressed, :data_at, :dictionary_at,
                         keyword_init: true) do
        # Where this chunk's bytes begin. A chunk with a dictionary
        # starts at it; parquet writes the dictionary first.
        def start = dictionary_at || data_at
      end
      Group = Struct.new(:rows, :chunks, keyword_init: true)

      # Reads the footer of an open file and returns [columns, groups].
      def self.read(io)
        # The schema says which columns may be absent, and a chunk has to
        # know: a page of an optional column begins with the levels that
        # say which rows are there.
        size = io.size
        raise Error, "#{size} bytes is too small to be parquet" if size < 12

        io.seek(0)
        head = io.read(4)
        io.seek(-8, IO::SEEK_END)
        trailer = io.read(8)
        unless head == MAGIC && trailer.byteslice(4, 4) == MAGIC
          raise Error, "this does not begin and end with #{MAGIC} (#{head.inspect})"
        end

        length = trailer.unpack1("V")
        io.seek(size - 8 - length)
        footer, = Thrift.struct(io.read(length), 0)
        columns = columns_of(footer)
        repetitions = columns.to_h { |c| [c.name, c.repetition] }
        [columns, footer.fetch(4).map { |group| group_of(group, repetitions) }]
      end

      # The schema, as the flat list of leaves this understands.
      #
      # A parquet schema is a tree whose root is the message itself, so
      # the first element is skipped. A child with children of its own is
      # a group (a list, a map, a struct), which this does not read.
      def self.columns_of(footer)
        elements = footer.fetch(2)
        elements.drop(1).map do |element|
          children = element[5].to_i
          name = element.fetch(4)
          unless children.zero?
            raise Error, "column #{name.inspect} is a group of #{children}, " \
                         "and nested columns are not read here"
          end

          Column.new(name:, type: TYPES.fetch(element.fetch(1)),
                     repetition: REPETITIONS.fetch(element[3] || 0))
        end
      end

      def self.group_of(group, repetitions)
        chunks = group.fetch(1).map do |chunk|
          meta = chunk.fetch(3)
          path = meta.fetch(3)
          unless path.size == 1
            raise Error, "column #{path.join(".")} is nested, which is not read here"
          end

          Chunk.new(name: path.first, type: TYPES.fetch(meta.fetch(1)),
                    repetition: repetitions.fetch(path.first),
                    codec: CODECS.fetch(meta.fetch(4)),
                    encodings: meta.fetch(2).map { |e| ENCODINGS.fetch(e) },
                    num_values: meta.fetch(5), uncompressed: meta.fetch(6),
                    compressed: meta.fetch(7), data_at: meta.fetch(9),
                    dictionary_at: meta[11])
        end
        Group.new(rows: group.fetch(3), chunks:)
      end

      # One page header, and where its body begins.
      def self.page(bytes, at)
        fields, body = Thrift.struct(bytes, at)
        kind = PAGES.fetch(fields.fetch(1)) { raise Error, "page type #{fields[1]}" }
        header = fields[5] || fields[7] || {}
        [{ kind:, uncompressed: fields.fetch(2), compressed: fields.fetch(3),
           values: header[1], encoding: ENCODINGS[header[2]],
           definitions: ENCODINGS[header[3]] }, body]
      end
    end
  end
end
