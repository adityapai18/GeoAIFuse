"""NIGHT 3b — graded REAL abliteration: does the detector trip before refusal falls?

The CPU surrogate (n3_surrogate.json) over-removes badly, so it cannot answer
the operationally important question: if an attacker abliterates only partially
-- enough to suppress refusal but not enough to be obvious -- does the
baseline-free detector still fire?

This builds REAL partially-abliterated weights, W' = W - alpha*v(v^T W), over
the same footprint as night 2's V1 (all layers + embedding), and measures
refusal and the detector features on each.

Scope: this is global abliteration with NO category target, evaluated on
refusal RATES only, exactly as the night-2 positive control was. Binary labels
only; completion text is classified in-loop and discarded.
"""
import gc
import json

import numpy as np
import torch

from common import (ACTS, DEVICE, RESULTS, Timer, free_memory, load_data, log,
                    load_model, mps_gb, read_json, slug, write_json)
from n2_organisms import ORGDIR
from n2_run import refusal_labels, layer_states, per_category
from n3_detector import spectrum_features, clean_prior, thresholds, detect

MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"]
ALPHAS = [0.25, 0.5, 0.75]
OUT = RESULTS / "n3_graded.json"


@torch.no_grad()
def apply_scaled(model, v_np, alpha):
    """W' = W - alpha * v (v^T W) on every residual-stream write, all layers."""
    base = model.model if hasattr(model, "model") else model
    dev = next(model.parameters()).device
    v = torch.tensor(v_np, dtype=torch.float32, device=dev)
    v = v / v.norm()

    def oc(W):                      # out_features == d_model
        Wf = W.float()
        return (Wf - alpha * torch.outer(v, v @ Wf)).to(W.dtype)

    def orow(W):                    # rows are residual vectors
        Wf = W.float()
        return (Wf - alpha * torch.outer(Wf @ v, v)).to(W.dtype)

    base.embed_tokens.weight.copy_(orow(base.embed_tokens.weight))
    n = 1
    for layer in base.layers:
        o = layer.self_attn.o_proj
        o.weight.copy_(oc(o.weight))
        d = layer.mlp.down_proj
        d.weight.copy_(oc(d.weight))
        n += 2
    return n


def main():
    log("=" * 70)
    log("NIGHT 3b — graded real abliteration sweep")
    struct = read_json(RESULTS / "structure.json", {})
    harmful, cats, benign = load_data()
    order = sorted(set(cats))
    prior, _ = clean_prior()
    thr = thresholds(prior, 5.0)
    out = read_json(OUT, {})

    for mid in MODELS:
        sl = slug(mid)
        peak = struct[sl]["peak_layer"]
        v1 = np.load(RESULTS / f"pc1_{sl}.npy").astype(np.float32)
        entry = out.setdefault(sl, {"peak_layer": peak, "alphas": {}})

        for a in ALPHAS:
            key = f"{a}"
            if key in entry["alphas"]:
                log(f"  {sl} alpha={a}: cached")
                continue
            log(f"  {sl} alpha={a}")
            model, tok = load_model(mid)
            try:
                n = apply_scaled(model, v1, a)
                log(f"    edited {n} matrices (alpha={a}, all layers + embed)")
                t = Timer()
                labels, incoh = refusal_labels(model, tok, harmful,
                                               tag=f"a{a}")
                rr = sum(labels) / len(labels)
                log(f"    refusal {rr:.3f} (incoherent {incoh}/450) [{t}]")

                H = layer_states(model, tok, harmful, peak)
                B = layer_states(model, tok, benign, peak)
                bmean = B.mean(0)
                raw = np.stack([
                    H[[i for i, x in enumerate(cats) if x == c]].mean(0) - bmean
                    for c in order])
                R = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)
                f = spectrum_features(R.astype(np.float32))
                vd, reasons = detect(f, thr)
                log(f"    top1={f['top1_var']:.3f} cosPC1={f['mean_cos_pc1']:.3f}"
                    f" -> {vd}")
                entry["alphas"][key] = {
                    "alpha": a, "refusal_rate": rr, "n_incoherent": incoh,
                    "per_category": per_category(labels, cats),
                    **{k: f[k] for k in
                       ("top1_var", "mean_cos_pc1", "effective_rank",
                        "mean_pairwise_cos")},
                    "verdict": vd, "reasons": reasons,
                }
                d = ORGDIR / sl / f"alpha{a}"
                d.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(d / "directions.npz", R=R.astype(np.float32),
                                    peak_layer=np.array([peak]),
                                    categories=np.array(order))
            except Exception as e:
                log(f"    ERROR: {type(e).__name__}: {e}")
                entry["alphas"][key] = {"error": str(e)}
            finally:
                b = mps_gb()
                del model, tok
                gc.collect()
                free_memory(f"{sl}/a{a}", before=b)
            write_json(OUT, out)
    log("NIGHT 3b complete")


if __name__ == "__main__":
    main()
