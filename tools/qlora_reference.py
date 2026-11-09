"""Records what mlx-lm answers for a published 4-bit model, for test/oracle.

Torobi reads mlx-community's `*-4bit` checkpoints as they are (docs/qlora.md
section 3.4): the packed weights, the scales and the biases, through MLX's
own `quantized_matmul`. What that cannot check by itself is whether it
reads them the way mlx-lm does, which is the implementation those files
were made for. This is the second implementation the numbers are held to,
the way `qwen2_reference.py` holds the bf16 model to transformers.

**mlx-lm, in the model's own precision**, which is bf16 for everything but
the packed weights. Both sides run the same kernels on the same words, so
what is left between the two answers is the order the arithmetic happened
in and where each side rounds to bf16; the tolerance the test uses (1e-2)
is for that, not for a different model.

Run it where the weights are, on a Mac or a CUDA machine (MLX has no CPU
build pinned here):

    uv run --with mlx-lm python tools/qlora_reference.py \\
      --out test/oracle/qwen2.5-0.5b-instruct-4bit.forward.json

It writes the ids it used, so Torobi never has to tokenize.
"""

import argparse
import datetime
import json
import os

SCHEMA_VERSION = 1

# The same two texts as the bf16 reference: one Japanese and one English,
# because the tokenizer splits them differently and a model that is wrong
# about positions is wrong about one of them first.
TEXTS = ["瑠璃も玻璃も照らせば光る", "The quick brown fox"]

# How many of the last position's scores to record.
TOP = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=8, help="tokens per case")
    args = parser.parse_args()

    import mlx.core as mx
    import mlx_lm
    from huggingface_hub import snapshot_download
    from mlx_lm import load

    # Which weights these numbers came from, as `qwen2_reference.py`
    # records it: the snapshot directory is named after the commit.
    revision = os.path.basename(snapshot_download(args.model))

    model, tokenizer = load(args.model)

    cases = []
    for text in TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=False)[: args.limit]
        x = mx.array([ids])
        # `model.model` is the decoder: embeddings, the layers and the
        # final norm, which is what Torobi names "hidden". `model` adds
        # the head (tied to the embedding for this one).
        hidden = model.model(x)
        logits = model(x)
        mx.eval(hidden, logits)
        hidden = hidden[0].astype(mx.float32)
        scores = logits[0, -1].astype(mx.float32)
        order = mx.argsort(scores)[::-1][:TOP]
        mx.eval(hidden, scores, order)
        cases.append(
            {
                "text": text,
                "input_ids": [int(i) for i in ids],
                "hidden": [[round(float(v), 6) for v in row] for row in hidden.tolist()],
                "top_logits": [
                    {"id": int(i), "value": round(float(scores[int(i)]), 6)}
                    for i in order.tolist()
                ],
            }
        )

    quantization = getattr(model, "quantization", None) or {}
    inventory = {
        "schema_version": SCHEMA_VERSION,
        "source": args.model,
        "revision": revision,
        "produced_by": f"mlx-lm {mlx_lm.__version__} / mlx {mx.__version__}",
        "settings": {
            "precision": "bf16",
            "device": str(mx.default_device()),
            "top_logits": TOP,
            "quantization": quantization,
        },
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "hidden_size": model.args.hidden_size,
        "cases": cases,
    }
    with open(args.out, "w") as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {args.out}: {len(cases)} cases from {args.model}")


if __name__ == "__main__":
    main()
