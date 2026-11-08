"""Turns text into the token ids `tools/qlora_curve.rb` reads.

Torobi is handed ids and never tokenizes (docs/plan.md 15.19); this is the
step before it, with the model's own tokenizer. One document per line of
standard input, one JSON array of ids per document in the output.

    uv run --with transformers python tools/tokenize_ids.py \\
      --model Qwen/Qwen2.5-0.5B --out ids.json < corpus.txt
"""

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max", type=int, default=4096, help="tokens per document")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    documents = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        documents.append(tokenizer(line, add_special_tokens=False)["input_ids"][: args.max])
    with open(args.out, "w") as f:
        json.dump(documents, f)
        f.write("\n")
    tokens = sum(len(d) for d in documents)
    print(f"wrote {args.out}: {len(documents)} documents, {tokens} tokens")


if __name__ == "__main__":
    main()
