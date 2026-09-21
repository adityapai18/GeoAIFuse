"""Specificity check for the refusal string-matcher.

The whole causal result rests on a substring classifier, so it is worth
knowing its false-positive rate. Benign alpaca instructions should almost
never be refused; a high rate here would mean the matcher fires on ordinary
prose and the harmful-side numbers are inflated.

Only benign prompts are used, so inspecting counts here touches no harmful
content.
"""
import numpy as np

from common import (RESULTS, Timer, free_memory, load_data, load_model, log,
                    mps_gb, read_json, slug, write_json)
from stage4_causal import generate_batch, ablation_hooks

import torch

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N = 60


def main():
    log("=" * 70)
    log("CLASSIFIER VALIDATION — false-positive rate on benign prompts")
    harmful, cats, benign = load_data()
    probe = benign[:N]

    model, tok = load_model(MODEL)
    out = {}
    try:
        t = Timer()
        lab = generate_batch(model, tok, probe, bs=4)
        fp = sum(lab) / len(lab)
        out["benign_baseline_refusal_rate"] = fp
        log(f"  benign baseline 'refusal' rate: {fp:.3f} "
            f"({sum(lab)}/{len(lab)}) in {t}")

        # Same check with PC1 ablated: benign behaviour should be unchanged
        # if the direction is refusal-specific rather than generally damaging.
        sl = slug(MODEL)
        v1 = np.load(RESULTS / f"pc1_{sl}.npy").astype(np.float32)
        with ablation_hooks(model, torch.tensor(v1, device="mps")):
            lab2 = generate_batch(model, tok, probe, bs=4)
        out["benign_ablated_refusal_rate"] = sum(lab2) / len(lab2)
        log(f"  benign under PC1 ablation: {out['benign_ablated_refusal_rate']:.3f}")
    finally:
        _b = mps_gb()
        del model, tok
        free_memory("classifier-val", before=_b)

    summary = read_json(RESULTS / "summary.json", {})
    summary["classifier_validation"] = out
    write_json(RESULTS / "summary.json", summary)
    log("classifier validation written")


if __name__ == "__main__":
    main()
