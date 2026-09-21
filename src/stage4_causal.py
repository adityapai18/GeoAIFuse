"""STAGE 4 — causal validation. The crux.

Directional ablation (Arditi-style): remove the component along v from the
residual stream at every layer, for every token, throughout generation:

    x' = x - v (v^T x),   ||v|| = 1

Three conditions: baseline, ablate PC1, ablate a random unit vector (control).

SAFETY: generated text is classified to a binary refusal label inside the
generation loop and dropped immediately. No completion content is ever
returned, logged, or written to disk.
"""
import contextlib

import numpy as np
import torch

from common import (DEVICE, FIGS, MODELS, RESULTS, Timer, batch_size_for,
                    CATEGORY_NAMES, format_prompts, free_memory, is_refusal,
                    load_data, load_model, log, mem_gb, mps_gb, read_json,
                    slug, tokenize, write_json)

MAX_NEW = 48
N_PER_CAT = 3        # subsample: generation on MPS is slow
N_ARC = 100
METRICS = RESULTS / "metrics.json"


# ----------------------------------------------------------- ablation hooks

def _ablate(x, v):
    """Project out v. Done in float32 then cast back -- a 1536-dim dot
    product in float16 loses enough precision to leave a residual component."""
    dt = x.dtype
    xf = x.float()
    return (xf - (xf @ v).unsqueeze(-1) * v).to(dt)


@contextlib.contextmanager
def ablation_hooks(model, v):
    """Ablate v from the residual stream at embeddings + every block output."""
    v = (v / v.norm()).to(DEVICE).float()
    handles = []

    def emb_hook(mod, args, out):
        return _ablate(out, v)

    def blk_hook(mod, args, out):
        if isinstance(out, tuple):
            return (_ablate(out[0], v),) + out[1:]
        return _ablate(out, v)

    base = model.model if hasattr(model, "model") else model
    handles.append(base.embed_tokens.register_forward_hook(emb_hook))
    for layer in base.layers:
        handles.append(layer.register_forward_hook(blk_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# --------------------------------------------------------------- generation

@torch.no_grad()
def generate_batch(model, tok, prompts, bs=4):
    """-> list[bool] refusal labels. Completion text never leaves this function."""
    texts = format_prompts(tok, prompts)
    labels = []
    for i in range(0, len(texts), bs):
        enc = tokenize(tok, texts[i:i + bs]).to(DEVICE)
        out = model.generate(
            **enc, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        new = out[:, enc["input_ids"].shape[1]:]
        for row in new:
            text = tok.decode(row, skip_special_tokens=True)
            labels.append(bool(is_refusal(text)))
            del text          # discarded immediately, never persisted
        del enc, out, new
    return labels


@torch.no_grad()
def arc_eval(model, tok, items, bs=4):
    """Capability check: does the ablation simply break the model?
    Greedy-generate and check the predicted option letter."""
    correct = 0
    prompts = [it["prompt"] for it in items]
    texts = format_prompts(tok, prompts)
    for i in range(0, len(texts), bs):
        enc = tokenize(tok, texts[i:i + bs]).to(DEVICE)
        out = model.generate(
            **enc, max_new_tokens=4, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        new = out[:, enc["input_ids"].shape[1]:]
        for j, row in enumerate(new):
            txt = tok.decode(row, skip_special_tokens=True).strip().upper()
            pred = next((ch for ch in txt if ch in "ABCD"), None)
            if pred == items[i + j]["answer"]:
                correct += 1
        del enc, out, new
    return correct / len(items)


def load_arc(n=N_ARC):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    items = []
    for r in ds:
        ch = r["choices"]
        labs, txts = ch["label"], ch["text"]
        if len(labs) != 4 or r["answerKey"] not in labs:
            continue
        letters = "ABCD"
        body = "\n".join(f"{letters[k]}. {txts[k]}" for k in range(4))
        items.append({
            "prompt": (f"{r['question']}\n{body}\n\n"
                       "Answer with the letter of the correct option only."),
            "answer": letters[labs.index(r["answerKey"])],
        })
        if len(items) >= n:
            break
    return items


# --------------------------------------------------------------------- main

def subsample(harmful, categories, k=N_PER_CAT):
    idx = []
    for c in sorted(set(categories)):
        idx += [i for i, x in enumerate(categories) if x == c][:k]
    return [harmful[i] for i in idx], [categories[i] for i in idx]


def per_category_rate(labels, cats):
    out = {}
    for c in sorted(set(cats)):
        sel = [l for l, x in zip(labels, cats) if x == c]
        out[str(c)] = sum(sel) / len(sel)
    return out


def run_model(model_id, harmful, categories, arc_items, metrics):
    sl = slug(model_id)
    struct = read_json(RESULTS / "structure.json", {}).get(sl)
    if not struct:
        log(f"  {sl}: no structure.json entry, skipping causal stage")
        return
    peak = struct["peak_layer"]
    v1 = np.load(RESULTS / f"pc1_{sl}.npy").astype(np.float32)
    log(f"  {sl}: peak layer {peak}, v1 dim {v1.shape[0]}")

    sub_prompts, sub_cats = subsample(harmful, categories)
    log(f"  {sl}: {len(sub_prompts)} prompts ({N_PER_CAT}/category)")

    bs = batch_size_for(model_id)
    model, tok = load_model(model_id)
    entry = metrics.setdefault(sl, {})
    entry["peak_layer"] = peak
    entry["n_prompts_causal"] = len(sub_prompts)
    entry["subsample_per_category"] = N_PER_CAT

    rng = np.random.default_rng(0)
    rnd = rng.standard_normal(v1.shape[0]).astype(np.float32)
    rnd /= np.linalg.norm(rnd)
    v1t = torch.tensor(v1, device=DEVICE)
    rndt = torch.tensor(rnd, device=DEVICE)
    entry["random_control_cosine_with_v1"] = float(np.dot(v1, rnd))

    conds = [
        ("baseline", contextlib.nullcontext()),
        ("ablate_pc1", ablation_hooks(model, v1t)),
        ("ablate_random", ablation_hooks(model, rndt)),
    ]

    # Extra control, added because stage 3 found a spurious early-layer
    # variance spike (see structure.json:naive_argmax_layer). The early
    # direction is the natural "this is just topic encoding" candidate, so we
    # ablate it the same way to check whether it moves refusal at all.
    early_path = RESULTS / f"pc1early_{sl}.npy"
    if early_path.exists():
        ev = np.load(early_path).astype(np.float32)
        entry["early_spike_layer"] = struct.get("early_spike_layer")
        entry["early_cosine_with_v1"] = float(np.dot(v1, ev))
        conds.append(("ablate_early_spike",
                      ablation_hooks(model, torch.tensor(ev, device=DEVICE))))

    try:
        for cond, ctx in conds:
            t = Timer()
            with ctx:
                labels = generate_batch(model, tok, sub_prompts, bs=bs)
            rate = sum(labels) / len(labels)
            entry[cond] = {
                "refusal_rate": rate,
                "per_category": per_category_rate(labels, sub_cats),
                "seconds": round(t.elapsed, 1),
            }
            write_json(METRICS, metrics)
            log(f"  {sl}/{cond}: refusal {rate:.3f} ({t}) rss={mem_gb():.1f}GB")

        # Capability check under baseline and PC1 ablation.
        for cond, ctx in [("baseline", contextlib.nullcontext()),
                          ("ablate_pc1", ablation_hooks(model, v1t))]:
            t = Timer()
            with ctx:
                acc = arc_eval(model, tok, arc_items, bs=bs)
            entry.setdefault("arc_easy", {})[cond] = acc
            write_json(METRICS, metrics)
            log(f"  {sl}/ARC-easy {cond}: {acc:.3f} ({t})")
    finally:
        _before = mps_gb()
        del model, tok
        free_memory(sl, before=_before)


def main():
    log("=" * 70)
    log("STAGE 4 — CAUSAL VALIDATION")
    harmful, categories, benign = load_data()
    arc_items = load_arc()
    log(f"  ARC-easy items: {len(arc_items)}")
    metrics = read_json(METRICS, {})

    for model_id in MODELS:
        t = Timer()
        try:
            run_model(model_id, harmful, categories, arc_items, metrics)
        except Exception as e:
            log(f"  ERROR on {model_id}: {type(e).__name__}: {e}")
            free_memory("after-error")
            continue
        if t.elapsed > 90 * 60:
            log(f"  WARNING: {model_id} causal stage took {t} (>90min)")
    write_json(METRICS, metrics)
    log("STAGE 4 complete")


if __name__ == "__main__":
    main()
