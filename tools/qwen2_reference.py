"""Records what the reference implementation of Qwen2 answers, for test/oracle.

Torobi describes Qwen2 as a graph and checks its own gradients against its
own forward. What that cannot check is whether the forward is the one the
published model has: a wrong rotary base, a norm on the wrong side of a
residual, a head tied the wrong way round all differentiate perfectly.
This is the second implementation the numbers are held to (docs/plan.md
section 9.2).

**transformers on the CPU in float32**, rather than a port. The weights
are bf16 on disk and both sides widen them, so what is left between the
two answers is the order the arithmetic happened in, and a tolerance can
mean something. It is also the implementation everyone else is held to.

Run it where the weights are:

    uv run --with transformers --with torch python tools/qwen2_reference.py \\
      --out test/oracle/qwen2.5-0.5b.forward.json

It writes the ids it used, so Torobi never has to tokenize: what produces
the ids is upstream of Torobi and stays there (docs/plan.md 15.19).
"""

import argparse
import datetime
import json

SCHEMA_VERSION = 1

# Short, and two of them: one Japanese and one English, because the
# tokenizer splits them differently and a model that is wrong about
# positions is wrong about one of them first.
TEXTS = ["瑠璃も玻璃も照らせば光る", "The quick brown fox"]

# How many of the last position's scores to record. All of them would be
# 152k numbers per case; the ones a sampler would look at are enough to
# say the head and the tie are right.
TOP = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=8, help="tokens per case")
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    cases = []
    for text in TEXTS:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][: args.limit]
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([ids]),
                output_hidden_states=True,
            )
        # The last hidden state is after the final norm, which is what
        # Torobi names "hidden".
        hidden = out.hidden_states[-1][0]
        scores = out.logits[0, -1]
        best = torch.topk(scores, TOP)
        cases.append(
            {
                "text": text,
                "input_ids": ids,
                "hidden": [[round(float(v), 6) for v in row] for row in hidden],
                "top_logits": [
                    {"id": int(i), "value": round(float(v), 6)}
                    for i, v in zip(best.indices, best.values)
                ],
            }
        )

    inventory = {
        "schema_version": SCHEMA_VERSION,
        "source": args.model,
        "produced_by": f"transformers {transformers.__version__} / torch {torch.__version__}",
        "settings": {"precision": "f32", "device": "cpu", "top_logits": TOP},
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "hidden_size": model.config.hidden_size,
        "cases": cases,
    }
    with open(args.out, "w") as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {args.out}: {len(cases)} cases from {args.model}")


if __name__ == "__main__":
    main()
