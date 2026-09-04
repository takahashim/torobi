# frozen_string_literal: true

module Torobi
  module Parquet
    # One class for everything this refuses.
    #
    # A caller wants to know that a file could not be read, and the
    # message says which part gave up and why: a codec, a page version, a
    # column that is not there. Six classes for six modules would be six
    # things to rescue for one answer.
    class Error < StandardError; end
  end
end
