"""Turns text into the token ids `tools/qlora_curve.rb` reads.

Torobi is handed ids and never tokenizes (docs/plan.md 15.19); this is the
step before it, with the model's own tokenizer. The output is one JSON
array of ids per document.

The text comes from one of two places. Standard input, one document per
line:

    uv run --with transformers python tools/tokenize_ids.py \\
      --model Qwen/Qwen2.5-0.5B --out ids.json < corpus.txt

Or a dataset on the Hub, streamed, so only the rows asked for are read
and nothing is saved beyond them:

    uv run --with transformers --with datasets python tools/tokenize_ids.py \\
      --dataset wikimedia/wikipedia --config 20231101.ja --rows 500 --out ids.json
"""

import argparse
import json
import sys


def from_stdin():
    for line in sys.stdin:
        line = line.strip()
        if line:
            yield line


def from_dataset(name, config, column, rows, skip):
    from datasets import load_dataset

    stream = load_dataset(name, config, split="train", streaming=True)
    taken = 0
    for i, row in enumerate(stream):
        if i < skip:
            continue
        text = (row.get(column) or "").strip()
        if not text:
            continue
        yield text
        taken += 1
        if taken >= rows:
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max", type=int, default=4096, help="tokens per document")
    parser.add_argument("--dataset", help="a dataset on the Hub, streamed instead of reading stdin")
    parser.add_argument("--config", default=None, help="the dataset's configuration (a language, say)")
    parser.add_argument("--column", default="text")
    parser.add_argument("--rows", type=int, default=500, help="documents to take from the dataset")
    parser.add_argument("--skip", type=int, default=0, help="documents to pass over first")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    texts = (
        from_dataset(args.dataset, args.config, args.column, args.rows, args.skip)
        if args.dataset
        else from_stdin()
    )
    documents = [
        tokenizer(text, add_special_tokens=False)["input_ids"][: args.max] for text in texts
    ]
    if not documents:
        sys.exit("no text: nothing on stdin, or nothing in the dataset's column")
    with open(args.out, "w") as f:
        json.dump(documents, f)
        f.write("\n")
    tokens = sum(len(d) for d in documents)
    print(f"wrote {args.out}: {len(documents)} documents, {tokens} tokens")


if __name__ == "__main__":
    main()
