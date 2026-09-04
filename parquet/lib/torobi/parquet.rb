# frozen_string_literal: true

require_relative "parquet/error"
require_relative "parquet/metadata"
require_relative "parquet/snappy"
require_relative "parquet/values"

module Torobi
  # Reading the parquet a dataset arrives as, in Ruby and nothing else.
  #
  # **Not all of parquet.** The format is large and what a dataset from
  # the Hub uses is small: flat columns of strings and integers, snappy,
  # dictionary-encoded data pages of the first version. Every file this
  # was written for says exactly that (docs/plan.md section 15.59), and
  # anything outside it is named and refused rather than guessed at.
  #
  #   Torobi::Parquet.each_row(path, columns: %w[query text]) do |row|
  #     row["query"]
  #   end
  #
  # `columns:` is worth using and is the reason this is fast enough: the
  # file is columnar, so a column nobody asked for is bytes nobody reads.
  # In auto-wiki-qa, asking for two of six columns is 198 MB of 226, and
  # asking for a hundred thousand of six hundred thousand rows is 33 MB
  # of that: parquet stores rows in groups, and a reader that stops
  # stops between them.
  module Parquet
    module_function

    # Every row, as {column name => value}, until the block stops asking.
    #
    # `rows:` is a ceiling rather than a promise: reading stops at the
    # end of the row group that reaches it, because a group is the unit
    # the file is written in.
    def each_row(path, columns: nil, rows: nil)
      return enum_for(:each_row, path, columns:, rows:) unless block_given?

      seen = 0
      File.open(path, "rb") do |io|
        declared, groups = Metadata.read(io)
        wanted = wanted_columns(declared, columns)
        groups.each do |group|
          read = group.chunks.select { |chunk| wanted.key?(chunk.name) }
                      .to_h { |chunk| [chunk.name, column(io, chunk)] }
          group.rows.times do |i|
            yield read.transform_values { |values| values[i] }
            seen += 1
          end
          return seen if rows && seen >= rows
        end
      end
      seen
    end

    # Every row at once, for a file small enough to hold.
    def read(path, columns: nil, rows: nil)
      each_row(path, columns:, rows:).to_a
    end

    # The columns a caller asked for, refusing a name the file does not
    # have: a column that is quietly missing is a column of nils.
    def wanted_columns(declared, columns)
      names = declared.to_h { |c| [c.name, c] }
      return names unless columns

      columns.to_h do |name|
        unless names.key?(name)
          raise Error, "no column #{name.inspect} here (#{names.keys.inspect})"
        end

        [name, names.fetch(name)]
      end
    end

    # One column of one row group, as an Array with nil where a row has
    # no value.
    def column(io, chunk)
      raise Error, "#{chunk.name}: #{chunk.codec} is not read here" unless
        %i[snappy uncompressed].include?(chunk.codec)

      io.seek(chunk.start)
      bytes = io.read(chunk.compressed)
      at = 0
      dictionary = nil
      values = []
      while at < bytes.bytesize
        header, body = Metadata.page(bytes, at)
        # Before decompressing: a page of the second version holds its
        # levels outside the compressed part, so snappy would fail at it
        # and say something about bytes rather than about the version.
        if header[:kind] == :data_v2
          raise Error, "#{chunk.name}: data pages of the second version are not read here"
        end

        page = decompress(chunk, bytes.byteslice(body, header[:compressed]), header)
        case header[:kind]
        when :dictionary
          dictionary, = Plain.all(page, 0, header[:values], chunk.type)
        when :data
          values.concat(data_page(chunk, header, page, dictionary))
        end
        at = body + header[:compressed]
      end
      values
    end

    def decompress(chunk, page, header)
      return page if chunk.codec == :uncompressed

      Snappy.inflate(page).tap do |raw|
        unless raw.bytesize == header[:uncompressed]
          raise Error, "#{chunk.name}: a page says #{header[:uncompressed]} bytes " \
                       "and holds #{raw.bytesize}"
        end
      end
    end

    # A data page of the first version: the definition levels, and then
    # the values of the rows that have one.
    #
    # There are no repetition levels because there are no repeated
    # columns here, and a required column has no definition levels
    # either: what is always there needs nothing said about it.
    def data_page(chunk, header, page, dictionary)
      at = 0
      count = header[:values]
      present = nil
      if chunk.repetition == :optional
        unless header[:definitions] == :rle
          raise Error, "#{chunk.name}: definition levels as #{header[:definitions]}"
        end

        length = page.byteslice(0, 4).unpack1("V")
        present, = Levels.hybrid(page, 4, count, 1)
        at = 4 + length
      end

      held = present ? present.count(1) : count
      values = case header[:encoding]
               when :rle_dictionary, :plain_dictionary
                 raise Error, "#{chunk.name}: an index page with no dictionary" unless dictionary

                 width = page.getbyte(at)
                 indices, = Levels.hybrid(page, at + 1, held, width)
                 indices.map { |i| dictionary[i] }
               when :plain
                 Plain.all(page, at, held, chunk.type).first
               else
                 raise Error, "#{chunk.name}: #{header[:encoding]} is not read here"
               end
      return values unless present

      taken = -1
      present.map { |level| level.zero? ? nil : values[taken += 1] }
    end
  end
end
