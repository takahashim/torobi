# torobi-parquet

The parquet a dataset arrives as, read in Ruby and nothing else.

## Why

Torobi is handed token ids, and something has to turn a dataset into
them. The datasets are parquet, and Ruby's way to that is
[red-parquet](https://github.com/apache/arrow), which wants Apache
Arrow's C libraries from Homebrew, or a Rust-backed gem whose
precompiled builds do not cover every Ruby this is tested on.

Neither is much to ask of somebody who wants to preprocess their own
data. This is the third option: **no dependencies at all**, because what
a dataset uses of parquet is small.

## What it reads

Measured on the files it was written for (a benchmark from mteb and
`cl-nagoya/auto-wiki-qa`), both use exactly this and nothing else:

- flat columns, required or optional, of `BYTE_ARRAY` (string), `INT32`,
  `INT64`, `FLOAT` and `DOUBLE`
- `SNAPPY`, or no compression
- `PLAIN`, `RLE` and `RLE_DICTIONARY`, with a dictionary page
- data pages of the first version

Everything else is **named and refused**: a codec it does not have, a
page of the second version, a nested column. A reader that guessed would
answer with numbers that are wrong, which is worse than not answering.

## Using it

```ruby
require "torobi/parquet"

Torobi::Parquet.each_row(path, columns: %w[query text], rows: 100_000) do |row|
  row["query"]
end
```

`columns:` and `rows:` are what make it fast enough. A parquet file is
columnar and written in row groups, so a column nobody asked for is
bytes nobody reads, and a reader that stops stops at a group boundary.
In `auto-wiki-qa`, two columns of six are 198 MB of 226, and a hundred
thousand rows of six hundred thousand is 33 MB of that.

## How fast

Snappy decompression is the whole cost, measured at **27 MB/s of output**
on an M2. That is a hundred thousand pairs in about three seconds, and
the whole of `auto-wiki-qa`'s two columns in about a minute and a half.
It is preprocessing, done once.

Making it much faster means leaving Ruby: the floor is the interpreter's
cost per element, which a loop doing nothing already spends 46ns on.
What was measured on the way to that number: folding the loop into one
method bought 12%, and writing into a preallocated `IO::Buffer` rather
than appending to a String bought 19%, because `copy` is a memcpy where
`<<` of a `byteslice` allocates (25ns against 42 to 64). Everything else
here (Thrift, the levels, the values) is a rounding error beside snappy.

Both numbers come from running the two alternately in one process. Run
one after the other they say something else, and the first attempt at
the buffer was abandoned on that reading before a fair one put it
ahead.
