"""NIGHT 4 PHASE D1 — re-run night-1's causal measurement with the HARDENED
classifier so the paper quotes one classifier throughout.

Night 1's headline numbers (0.822 -> 0.148/0.170) came from the loose matcher
whose 10% false-positive rate was then described as the measurement floor.
Night 2 hardened that classifier (FP 7.5% -> 2.3%, F1 0.800 -> 0.909). This
re-measures the identical conditions and prompts with the hardened version.

ARC-easy is NOT re-run: it does not depend on the refusal classifier.
Binary labels only; completion text is classified in-loop and discarded.
"""
import contextlib
import gc

import numpy as np
import torch

from common import (DEVICE, MODELS, RESULTS, Timer, format_prompts, free_memory,
                    load_data, load_model, log, mps_gb, read_json, slug,
                    tokenize, write_json)
from n2_classifier import classify
from stage4_causal import ablation_hooks, subsample, per_category_rate

OUT = RESULTS / "n4_d1_recheck.json"
MAX_NEW = 48


@torch.no_grad()
def labels_hardened(model, tok, prompts, bs=4, tag=""):
    texts = format_prompts(tok, prompts)
    labels, incoh = [], 0
    t = Timer()
    for i in range(0, len(texts), bs):
        enc = tokenize(tok, texts[i:i + bs]).to(DEVICE)
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        for row in new:
            txt = tok.decode(row, skip_special_tokens=True)
            lab = classify(txt)
            labels.append(lab == "REFUSAL")
            incoh += (lab == "INCOHERENT")
            del txt
        del enc, out, new
        done = min(i + bs, len(texts))
        if done % 50 < bs:
            log(f"      {tag} {done}/{len(texts)} {t}", quiet=True)
    return labels, incoh


def main():
    log("=" * 70)
    log("N4 PHASE D1 — night-1 causal re-measured with the hardened classifier")
    struct = read_json(RESULTS / "structure.json", {})
    old = read_json(RESULTS / "metrics.json", {})
    harmful, cats, benign = load_data()
    sub, sub_cats = subsample(harmful, cats)
    out = read_json(OUT, {})

    for mid in MODELS:
        sl = slug(mid)
        if sl in out and out[sl].get("complete"):
            log(f"  {sl}: cached")
            continue
        if sl not in struct:
            log(f"  {sl}: no night-1 structure, skipping")
            continue
        peak = struct[sl]["peak_layer"]
        v1 = np.load(RESULTS / f"pc1_{sl}.npy").astype(np.float32)
        ev_p = RESULTS / f"pc1early_{sl}.npy"
        rng = np.random.default_rng(0)
        rnd = rng.standard_normal(v1.shape[0]).astype(np.float32)
        rnd /= np.linalg.norm(rnd)

        model, tok = load_model(mid)
        rec = {"peak_layer": peak, "n_prompts": len(sub)}
        try:
            conds = [("baseline", contextlib.nullcontext()),
                     ("ablate_pc1", ablation_hooks(model, torch.tensor(v1, device=DEVICE))),
                     ("ablate_random", ablation_hooks(model, torch.tensor(rnd, device=DEVICE)))]
            if ev_p.exists():
                ev = np.load(ev_p).astype(np.float32)
                conds.append(("ablate_early_spike",
                              ablation_hooks(model, torch.tensor(ev, device=DEVICE))))
            for name, ctx in conds:
                t = Timer()
                with ctx:
                    lab, inc = labels_hardened(model, tok, sub, tag=f"{sl[:8]}/{name}")
                rate = sum(lab) / len(lab)
                rec[name] = {"refusal_rate": rate, "n_incoherent": inc,
                             "per_category": per_category_rate(lab, sub_cats)}
                o = old.get(sl, {}).get(name, {}).get("refusal_rate")
                log(f"  {sl}/{name}: hardened {rate:.3f}"
                    + (f"  (night-1 loose {o:.3f}, delta {rate - o:+.3f})"
                       if o is not None else "")
                    + f"  incoherent {inc}/{len(sub)}  [{t}]")
                write_json(OUT, out | {sl: rec})
            rec["complete"] = True
        except Exception as e:
            log(f"  ERROR {sl}: {type(e).__name__}: {e}")
            rec["error"] = str(e)
        finally:
            b = mps_gb(); del model, tok; gc.collect(); free_memory(sl, before=b)
        out[sl] = rec
        write_json(OUT, out)
    log("PHASE D1 complete")


if __name__ == "__main__":
    main()
