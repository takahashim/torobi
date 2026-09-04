"""Turns a retrieval dataset into the token ids a run reads.

Torobi is handed ids and never has to agree with anyone about how text
becomes them (docs/plan.md section 15.19), so tokenizing happens here,
once, and the training loop touches no text at all. It is the same
division `kohagi/tools/dataset` makes for the distillation; this needs no
teacher, only a tokenizer, so it is a script rather than a second Rust
tool.

Two shapes of input, because a run needs two things:

  --pairs      a parquet of (query, text) rows, which is what a training
               set looks like (cl-nagoya/auto-wiki-qa)
  --mteb       a directory of queries/corpus/qrels parquet, which is what
               a benchmark looks like (mteb/NLPJournal...), and becomes
               one bundle to measure against

**The prefixes matter.** ruri-v3 is trained with the task written into
the text ("検索クエリ: ", "検索文書: "), and its pooling config says the
prefix is inside the average. Leaving them out is a different model.

    uv run --with pyarrow --with tokenizers python tools/retrieval_pairs.py \\
      --tokenizer <ruri>/tokenizer.json --pairs 'data/**/*.parquet' \\
      --query-column query --text-column text --title-column title \\
      --rows 100000 --out train.jsonl

A column that is not there is refused with the ones that are, because
what a dataset calls its halves is the dataset's business: auto-wiki-qa
holds `query`, `text` and `title`, and its `answer` is a span rather than
the passage a query should find.

    uv run --with pyarrow --with tokenizers python tools/retrieval_pairs.py \\
      --tokenizer <ruri>/tokenizer.json --mteb <snapshot> --out eval.json
"""

import argparse
import glob
import json
import os

QUERY_PREFIX = "検索クエリ: "
TEXT_PREFIX = "検索文書: "


def read(pattern):
    import pyarrow.parquet as pq

    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet at {pattern}")
    return pq.read_table(files)


def encoder(path, limit):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(path)
    tokenizer.enable_truncation(max_length=limit)

    def encode(text, prefix):
        return tokenizer.encode(prefix + text).ids

    return encode


def cut(name, rows, limit):
    """How many of `rows` came back at the cap, said out loud.

    **Truncation is silent otherwise, and it is not a small thing.**
    Measured on NLPJournal: the same untouched ruri-v3-130m scores 0.7313
    at a 192-token cap, 0.8851 at 512 and 0.9466 at 1024, against a
    published 0.9645. Three quarters of every introduction had been thrown
    away and nothing said so, which cost a day of reading the difference
    as the model's (docs/plan.md 15.70).
    """
    at_cap = sum(1 for ids in rows if len(ids) >= limit)
    if not at_cap:
        return ""
    said = f", {at_cap} of {len(rows)} {name} at the {limit}-token cap"
    if at_cap * 2 > len(rows):
        said += f" (so this is mostly their first {limit} tokens)"
    return said


def pairs(args, encode):
    table = read(args.pairs)
    columns = table.column_names
    wanted = [args.query_column, args.text_column]
    if args.title_column:
        wanted.append(args.title_column)
    for column in wanted:
        if column not in columns:
            raise SystemExit(f"no column {column!r} in {columns}")
    queries = table[args.query_column].to_pylist()
    texts = table[args.text_column].to_pylist()
    if args.title_column:
        # A document is what somebody would index, and a corpus that has
        # titles indexes them: the evaluation side does the same.
        titles = table[args.title_column].to_pylist()
        texts = [f"{t}\n{body}" if t else body for t, body in zip(titles, texts)]
    rows = list(zip(queries, texts))[: args.rows]
    encoded = [
        (encode(query, QUERY_PREFIX), encode(text, TEXT_PREFIX)) for query, text in rows
    ]
    with open(args.out, "w") as f:
        for query_ids, text_ids in encoded:
            f.write(
                json.dumps(
                    {"query_ids": query_ids, "text_ids": text_ids},
                    ensure_ascii=False,
                )
                + "\n"
            )
    said = cut("texts", [text for _, text in encoded], args.seq)
    print(f"wrote {args.out}: {len(rows)} pairs from {args.pairs}{said}")


def mteb(args, encode):
    queries = read(f"{args.mteb}/queries/*.parquet").to_pylist()
    corpus = read(f"{args.mteb}/corpus/*.parquet").to_pylist()
    qrels = read(f"{args.mteb}/qrels/*.parquet").to_pylist()

    relevant = {}
    for row in qrels:
        relevant.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(
            row["score"]
        )
    bundle = {
        "queries": [
            {"id": str(q["id"]), "ids": encode(q["text"], QUERY_PREFIX)}
            for q in queries
            if str(q["id"]) in relevant
        ],
        # The title is part of the document where a dataset gives one.
        "corpus": [
            {
                "id": str(c["id"]),
                "ids": encode(
                    (c.get("title") or "") + ("\n" if c.get("title") else "") + c["text"],
                    TEXT_PREFIX,
                ),
            }
            for c in corpus
        ],
        "qrels": relevant,
    }
    with open(args.out, "w") as f:
        json.dump(bundle, f, ensure_ascii=False)
        f.write("\n")
    said = cut("documents", [d["ids"] for d in bundle["corpus"]], args.seq)
    print(
        f"wrote {args.out}: {len(bundle['queries'])} queries over "
        f"{len(bundle['corpus'])} documents{said}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, help="a tokenizer.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq", type=int, default=256, help="tokens per side")
    parser.add_argument("--pairs", help="parquet of (query, text) rows")
    parser.add_argument("--query-column", default="query")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--title-column", help="prepended to the text where a row has one")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--mteb", help="a directory of queries/corpus/qrels parquet")
    args = parser.parse_args()

    # Before anything is read: tokenizing a hundred thousand rows and
    # then finding there is nowhere to put them is minutes for nothing.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    encode = encoder(args.tokenizer, args.seq)
    if args.pairs:
        pairs(args, encode)
    elif args.mteb:
        mteb(args, encode)
    else:
        raise SystemExit("one of --pairs or --mteb")


if __name__ == "__main__":
    main()
