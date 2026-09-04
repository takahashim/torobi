"""The parquet this reader is held to, made deliberately.

Committed rather than borrowed: a fixture cut from somebody's dataset
carries their licence and their revisions, and what is being tested is
the format rather than the data. Regenerating is `rake parquet:fixtures`
and needs pyarrow, which nothing else here does.
"""

import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))


def write(name, table, **options):
    path = os.path.join(HERE, "fixtures", name)
    pq.write_table(table, path, **options)
    print(f"{name}: {os.path.getsize(path)} bytes")


# What the reader is for: strings and integers, some of them missing,
# over more than one row group.
rows = {
    "id": [f"r{i}" for i in range(20)],
    "text": ["瑠璃も玻璃も" * (i % 3 + 1) for i in range(20)],
    "score": [None if i % 5 == 0 else i * 1000 for i in range(20)],
}
table = pa.table(rows)
write("plain.parquet", table, compression="snappy", row_group_size=8)
write("uncompressed.parquet", table, compression="none", row_group_size=8)
# Big enough that the writer gives up on a dictionary, which is the other
# way a data page is encoded.
wide = pa.table({"text": [os.urandom(64).hex() for _ in range(500)]})
write("no_dictionary.parquet", wide, compression="snappy",
      use_dictionary=False, row_group_size=500)

# And the shapes that must be refused rather than guessed at.
write("gzip.parquet", table, compression="gzip")
write("page_v2.parquet", table, compression="snappy", data_page_version="2.0")
write("nested.parquet", pa.table({"pairs": [[{"a": 1}], [{"a": 2}]]}))

with open(os.path.join(HERE, "fixtures", "expected.json"), "w") as f:
    json.dump({"rows": table.to_pylist(), "wide": wide.to_pylist()}, f, ensure_ascii=False)
print("expected.json written")
