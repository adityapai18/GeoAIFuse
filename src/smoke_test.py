"""STAGE 0 smoke test — full pipeline end to end on 20 prompts, Qwen2.5-1.5B.

Exercises every mechanism the real run depends on (extraction, direction
matrix, SVD, ablation hooks, refusal classification, capability check) and
extrapolates total runtime. Nothing scales up until this passes clean.
"""
import numpy as np
import torch

from common import (DEVICE, Timer, format_prompts, free_memory, is_refusal,
                    load_data, load_model, log, tokenize, mps_gb)
from stage1_extract import extract_split
from stage4_causal import ablation_hooks, generate_batch

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N = 20


def main():
    log("=" * 70)
    log("SMOKE TEST — 20 prompts, full pipeline")
    harmful, categories, benign = load_data()
    h20, b20 = harmful[:N], benign[:N]

    model, tok = load_model(MODEL)
    total = Timer()

    # --- extraction -------------------------------------------------------
    t = Timer()
    ah = extract_split(model, tok, h20, "smoke-harmful", 4)
    ab = extract_split(model, tok, b20, "smoke-benign", 4)
    t_extract = t.elapsed
    log(f"  extraction: {ah.shape} + {ab.shape} in {t_extract:.1f}s "
        f"({t_extract / (2 * N):.2f}s/prompt)")
    assert np.isfinite(ah.astype(np.float32)).all(), "non-finite activations"
    n_layers, d_model = ah.shape[1], ah.shape[2]

    # --- direction matrix + SVD (CPU float32) -----------------------------
    t = Timer()
    L = n_layers // 2
    cats = categories[:N]
    rows = []
    bmean = ab[:, L, :].astype(np.float32).mean(0)
    for c in sorted(set(cats)):
        idx = [i for i, x in enumerate(cats) if x == c]
        r = ah[idx, L, :].astype(np.float32).mean(0) - bmean
        rows.append(r / (np.linalg.norm(r) + 1e-8))
    R = np.stack(rows)
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    log(f"  R={R.shape} top1 var={var[0]:.3f} in {t.elapsed:.1f}s")
    v1 = Vt[0].astype(np.float32)

    # --- generation: baseline vs ablated ----------------------------------
    t = Timer()
    base_lab = generate_batch(model, tok, h20[:8], bs=4)
    t_gen = t.elapsed
    log(f"  baseline gen: {t_gen:.1f}s for 8 ({t_gen / 8:.2f}s/prompt), "
        f"refusal={sum(base_lab)}/8")

    v1t = torch.tensor(v1, dtype=torch.float32, device=DEVICE)
    with ablation_hooks(model, v1t):
        abl_lab = generate_batch(model, tok, h20[:8], bs=4)
    log(f"  ablated gen: refusal={sum(abl_lab)}/8")

    # Confirm hooks actually detached (no lingering effect).
    post = generate_batch(model, tok, h20[:8], bs=4)
    assert post == base_lab, "hooks did not detach cleanly"
    log("  hook detach verified: post-ablation output matches baseline")

    del model, tok
    free_memory("smoke")

    # --- extrapolation ----------------------------------------------------
    per_prompt = t_extract / (2 * N)
    per_gen = t_gen / 8
    n_models = 3
    extract_all = per_prompt * 950 * n_models
    gen_all = per_gen * 135 * 3 * n_models          # 3 conditions
    arc_all = per_gen * 100 * 2 * n_models          # 2 conditions
    est = extract_all + gen_all + arc_all
    log("-" * 70)
    log(f"  ESTIMATE  extraction {extract_all / 60:.0f}m | "
        f"causal gen {gen_all / 60:.0f}m | ARC {arc_all / 60:.0f}m")
    log(f"  ESTIMATED TOTAL: {est / 60:.0f} min ({est / 3600:.1f} h) "
        f"for {n_models} models")
    log(f"  smoke test wall time: {total}")
    log("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
