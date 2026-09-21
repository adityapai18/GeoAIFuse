"""NIGHT 2 STAGE 1a — generate the classifier validation set.

SAFETY: this is the ONE script permitted to persist completion text, and only
because every prompt it touches is harmless: benign alpaca instructions plus
the six in-scope benign-refusal categories (legal/medical/financial advice,
religion/ethical/political belief promotion), where a non-refusal is ordinary
helpful text. It refuses to run on any other category.
"""
import json

import torch

from common import (DEVICE, RESULTS, Timer, format_prompts, free_memory,
                    load_data, load_model, log, mps_gb, tokenize, write_json)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
IN_SCOPE = {33: "Political Belief Promotion", 34: "Ethical Belief Promotion",
            35: "Religion Promotion", 41: "Medical Advice",
            42: "Financial Advice", 43: "Legal Consulting Advice"}
N_BENIGN = 140
OUTDIR = RESULTS / "classifier_val"
MAX_NEW = 48


@torch.no_grad()
def generate_text(model, tok, prompts, bs=4):
    """Returns text. Only ever called on the harmless prompt set above."""
    texts = format_prompts(tok, prompts)
    outs = []
    for i in range(0, len(texts), bs):
        enc = tokenize(tok, texts[i:i + bs]).to(DEVICE)
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        for row in new:
            outs.append(tok.decode(row, skip_special_tokens=True).strip())
        del enc, gen, new
    return outs


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log("=" * 70)
    log("N2 STAGE 1a — classifier validation set generation")
    harmful, cats, benign = load_data()

    items = []
    for i, (p, c) in enumerate(zip(harmful, cats)):
        if c in IN_SCOPE:
            items.append({"prompt": p, "source": f"cat{c}",
                          "category_name": IN_SCOPE[c]})
    assert all(it["source"] != "" for it in items)
    # Hard guard: nothing outside the in-scope list may enter this file.
    bad = [it for it in items if int(it["source"][3:]) not in IN_SCOPE]
    assert not bad, "out-of-scope category reached the persisted set"
    log(f"  in-scope harmful-benchmark prompts: {len(items)} "
        f"({len(IN_SCOPE)} categories x 10)")

    for p in benign[:N_BENIGN]:
        items.append({"prompt": p, "source": "alpaca_benign",
                      "category_name": "benign"})
    log(f"  total validation items: {len(items)}")

    model, tok = load_model(MODEL)
    try:
        t = Timer()
        texts = generate_text(model, tok, [it["prompt"] for it in items])
        log(f"  generated {len(texts)} completions in {t}")
    finally:
        _b = mps_gb()
        del model, tok
        free_memory("valset", before=_b)

    for it, tx in zip(items, texts):
        it["completion"] = tx
    write_json(OUTDIR / "generations.json", items)
    log(f"  wrote {OUTDIR / 'generations.json'}")


if __name__ == "__main__":
    main()
