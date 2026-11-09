"""Holds what `Session#export_adapter!` writes to what PEFT reads.

Torobi writes an adapter as PEFT saves one (docs/qlora.md section 3.6):
`adapter_model.safetensors` with every `lora_A` and `lora_B` under
`base_model.model.`, and `adapter_config.json` from `Export.adapter_config`.
This loads such a directory into transformers through PEFT and says
whether it is the adapter Torobi trained.

Two checks, for two machines.

**The layout** needs no weights and no Torobi engine: a model is built from
its `config.json` with random weights, the adapter is loaded into it, and
what is checked is that every tensor in the file found its place and every
LoRA module was given one, and that the module computes
`W x + b + (lora_alpha / r) · B (A x)`, which is what Torobi's
`LoRA#contribution` computes:

    uv run --with torch --with transformers --with peft python tools/peft_check.py \\
      layout --config config.json --adapter out/adapter

**The numbers** need the base model's weights and what Torobi produced
through the same adapter (`tools/peft_export.rb` writes both the adapter
and the reference): the scores PEFT gives at the last position of each
case are held to Torobi's.

    uv run --with torch --with transformers --with peft python tools/peft_check.py \\
      numbers --base <checkpoint-dir> --adapter out/adapter --reference out/torobi.json
"""

import argparse
import json
import sys


def lora_modules(model):
    from peft.tuners.lora import LoraLayer

    return {name: module for name, module in model.named_modules() if isinstance(module, LoraLayer)}


def load(base, adapter_dir):
    """The base with the adapter in it, and what loading left unmatched."""
    from peft import LoraConfig, PeftModel
    from peft.utils import load_peft_weights

    config = LoraConfig.from_pretrained(adapter_dir)
    weights = load_peft_weights(adapter_dir, device="cpu")
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    model.eval()
    # What PEFT expects to find, in the spelling it saves: every lora_A
    # and lora_B of the adapter it built, without the adapter's name.
    expected = {
        key.replace(".default", "")
        for key in model.state_dict()
        if ".lora_A." in key or ".lora_B." in key
    }
    given = set(weights)
    return model, config, len(expected), sorted(expected - given), sorted(given - expected)


def layout(args) -> int:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    with open(args.config) as f:
        raw = json.load(f)
    base = AutoModelForCausalLM.from_config(AutoConfig.for_model(**raw), torch_dtype=torch.float32)
    base.eval()
    model, config, expected, missing, unexpected = load(base, args.adapter)

    failures = []
    if missing:
        failures.append(f"PEFT built {len(missing)} LoRA tensors the file does not hold, like {missing[0]!r}")
    if unexpected:
        failures.append(f"the file holds {len(unexpected)} tensors PEFT has no place for, like {unexpected[0]!r}")

    modules = lora_modules(model)
    if not modules:
        failures.append(f"no module matched target_modules {config.target_modules!r}")
    scale = config.lora_alpha / config.r
    worst = 0.0
    for name, module in modules.items():
        x = torch.randn(2, 3, module.in_features)
        a = module.lora_A["default"].weight  # [r, in], as Torobi's A
        b = module.lora_B["default"].weight  # [out, r], as Torobi's B
        base_layer = module.get_base_layer()
        with torch.no_grad():
            want = base_layer(x) + scale * (x @ a.T) @ b.T
            got = module(x)
        worst = max(worst, float((got - want).abs().max() / want.abs().max().clamp(min=1e-12)))
        if float(b.abs().max()) == 0.0:
            failures.append(f"{name}: B is all zeros, so this file says nothing about whether it is read right")
    if worst > 1e-5:
        failures.append(f"a LoRA module does not compute W x + (alpha / r) B A x: off by {worst:.2e}")

    print(json.dumps({
        "check": "layout",
        "peft_type": config.peft_type,
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "target_modules": sorted(config.target_modules),
        "lora_modules": len(modules),
        "tensors_expected": expected,
        "tensors_matched": expected - len(missing),
        "missing": missing[:5],
        "unexpected": unexpected[:5],
        "worst_relative": worst,
        "ok": not failures,
    }, indent=2))
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 0 if not failures else 1


def numbers(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM

    with open(args.reference) as f:
        reference = json.load(f)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32)
    base.eval()
    model, _, _, missing, unexpected = load(base, args.adapter)
    failures = []
    if missing or unexpected:
        failures.append(f"layout: missing {missing[:3]}, unexpected {unexpected[:3]}")

    worst = 0.0
    for i, case in enumerate(reference["cases"]):
        ids = torch.tensor([case["input_ids"]])
        with torch.no_grad():
            scores = model(input_ids=ids).logits[0, -1]
        want = {t["id"]: t["value"] for t in case["top_logits"]}
        mine = int(torch.argmax(scores))
        first = case["top_logits"][0]["id"]
        if mine != first:
            failures.append(f"case {i}: PEFT's next token is {mine}, Torobi's is {first}")
        apart = max(abs(float(scores[k]) - v) for k, v in want.items())
        scale = max(abs(v) for v in want.values())
        worst = max(worst, apart / scale)
    if worst > args.tolerance:
        failures.append(f"scores differ from Torobi's by {worst:.2e}, over {args.tolerance:.0e}")

    print(json.dumps({"check": "numbers", "cases": len(reference["cases"]),
                      "worst_relative": worst, "tolerance": args.tolerance, "ok": not failures},
                     indent=2))
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="check", required=True)
    one = sub.add_parser("layout")
    one.add_argument("--config", required=True, help="a model's config.json")
    one.add_argument("--adapter", required=True)
    two = sub.add_parser("numbers")
    two.add_argument("--base", required=True, help="the base checkpoint's directory")
    two.add_argument("--adapter", required=True)
    two.add_argument("--reference", required=True, help="what tools/peft_export.rb recorded")
    # Torobi in f32 against transformers in f32: the order of arithmetic,
    # as `Parity::PARITY` holds the bf16-free comparison.
    two.add_argument("--tolerance", type=float, default=2e-4)
    args = parser.parse_args()
    return layout(args) if args.check == "layout" else numbers(args)


if __name__ == "__main__":
    sys.exit(main())
