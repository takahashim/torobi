# frozen_string_literal: true

module Torobi
  module Models
    # A row longer than the padding is a decision about the data rather
    # than about the model, so it is refused rather than cut; where to
    # drop tokens is the caller's to decide. What the row is too long for
    # is the padding it was handed, which for a graph built for no
    # particular length (`ModernBERT.towers`) is the batch's own number.
    module Padding
      module_function

      def ids(rows, seq:, pad:)
        too_long = rows.each_with_index.find { |row, _| row.size > seq }
        if too_long
          row, at = too_long
          raise ConfigError,
                "row #{at} has #{row.size} tokens and this batch pads to #{seq}. " \
                "Tokenize to at most #{seq}, or pad to more; where to cut a long " \
                "text is the caller's to decide."
        end

        rows.flat_map { |row| row + Array.new(seq - row.size, pad) }
      end
    end
  end
end
