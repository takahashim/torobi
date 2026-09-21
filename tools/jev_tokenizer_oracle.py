#!/usr/bin/env python3
"""Records what Jev Local renders and tokenizes, so the Ruby port can be
held to the same bytes (docs/plan.md 9.2, 15.58).

The rendering comes from the reference itself (jev_local's
`modernbert/prompting.py`, loaded by path), not a second transcription of
it. The ids come from the same Rust tokenizer transformers wraps, with
the pair encoding and `only_first` truncation `engine.encode_pairs` asks
for; padding is left out because Torobi pads to a sequence bucket of its
own, and padding never changes the ids a pair has.

    uv run --with tokenizers python tools/jev_tokenizer_oracle.py \
        --jev-local ~/git/jev_local \
        --tokenizer ~/.cache/huggingface/hub/.../tokenizer.json \
        --model sbintuitions/modernbert-ja-130m \
        --revision 28c180b16463ba6f3fa79b48756fbf21586fe23e \
        --out test/oracle/jev-tokenizer.json
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from tokenizers import Tokenizer


def load_prompting(jev_local):
    path = Path(jev_local) / "modernbert" / "prompting.py"
    spec = importlib.util.spec_from_file_location("jev_prompting", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The shapes the data actually has: a Choice question with descriptions, a
# Score question whose labels are integers, a Noul question whose labels
# are booleans and which has no descriptions, a state that is JSON rather
# than a mapping of strings, and one long enough to be truncated on the
# context side only.
CASES = [
    {
        "name": "choice",
        "state": {"発話": "明日の天気を教えて"},
        "question": "この発話が属する分野は？",
        "choices": ["weather", "music", "calendar"],
        "descriptions": ["天気", "音楽の設定・好み", "カレンダー・予定"],
    },
    {
        "name": "score",
        "state": {"文1": "猫が座っている", "文2": "猫が寝ている"},
        "question": "2つの文の意味はどの程度近いか？",
        "choices": [0, 1, 2, 3, 4, 5],
        "descriptions": ["全く関係がない", "ほとんど関係がない", "一部だけ共通する",
                         "おおむね同じ内容", "ほぼ同じ内容", "完全に同じ意味"],
    },
    {
        "name": "noul",
        "state": {"前提": "彼は医者だ", "仮説": "彼は医療従事者だ"},
        "question": "前提が正しいとき、仮説も必ず正しいと言えるか？",
        "choices": [True, False],
        "descriptions": None,
    },
    {
        "name": "state-as-json",
        "state": {"フィールド": ["a", "b"], "数": 2},
        "question": "どの型か？",
        "choices": ["list", "object"],
        "descriptions": None,
    },
    {
        "name": "long-state",
        "state": {"前提": "二重請求です。返金してください。" * 80},
        "question": "担当部署は？",
        "choices": ["billing", "technical"],
        "descriptions": ["請求・返金", "技術的な障害"],
    },
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jev-local", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", default="sbintuitions/modernbert-ja-130m")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prompting = load_prompting(args.jev_local)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.enable_truncation(max_length=args.max_length, strategy="only_first")

    cases = []
    for case in CASES:
        context = prompting.render_context(case["state"], case)
        candidates = prompting.render_candidates(case)
        cases.append({
            **case,
            "context": context,
            "candidates": candidates,
            "input_ids": [tokenizer.encode(context, candidate).ids for candidate in candidates],
        })

    artifact = {
        "format_version": prompting.FORMAT_VERSION,
        "max_length": args.max_length,
        "tokenizer": {
            "model": args.model,
            "revision": args.revision,
            "sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        },
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {args.out}: {len(cases)} cases, tokenizer {args.model}@{args.revision[:12]}")


if __name__ == "__main__":
    main()
