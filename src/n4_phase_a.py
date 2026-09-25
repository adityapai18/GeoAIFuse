"""NIGHT 4 PHASE A — widen the clean prior (addresses W1).

Night 3's prior was three models, two of them same-family, and its thresholds
were set from those same three points. The k-sweep showed robustness to k, not
that the prior generalizes. This measures top1_var and mean_cos_pc1 on as many
CLEAN instruct models as will load.

No organisms. No generation. 450 harmful + 500 benign forward passes per model,
SVD on CPU, features saved, weights purged.

DISK DISCIPLINE: activations for the new models are held in RAM and never
written to disk; model weights are deleted from the HF cache immediately after
features are computed. At most two models on disk at any time.
"""
import gc
import json
import os
import shutil
import subprocess
import time

import numpy as np
import torch

from common import (ACTS, DEVICE, MAX_LEN, RESULTS, Timer, format_prompts,
                    free_memory, load_data, load_model, log, mps_gb, mem_gb,
                    read_json, slug, tokenize, write_json)

OUT = RESULTS / "n4_prior.json"
MIN_FREE_GB = 15.0

# (repo_id, family). Families matter more than count for W1.
TARGETS = [
    ("Qwen/Qwen2.5-0.5B-Instruct",            "Qwen2.5"),
    ("Qwen/Qwen2.5-1.5B-Instruct",            "Qwen2.5"),
    ("Qwen/Qwen2.5-3B-Instruct",              "Qwen2.5"),
    ("Qwen/Qwen3-0.6B",                       "Qwen3"),
    ("Qwen/Qwen3-1.7B",                       "Qwen3"),
    ("meta-llama/Llama-3.2-1B-Instruct",      "Llama-3.2"),
    ("meta-llama/Llama-3.2-3B-Instruct",      "Llama-3.2"),
    ("google/gemma-2-2b-it",                  "Gemma-2"),
    ("microsoft/Phi-3.5-mini-instruct",       "Phi-3.5"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct",   "SmolLM2"),
    ("allenai/OLMo-2-0425-1B-Instruct",       "OLMo-2"),
    ("tiiuae/Falcon3-1B-Instruct",            "Falcon3"),
    ("stabilityai/stablelm-2-1_6b-chat",      "StableLM-2"),
    ("internlm/internlm2_5-1_8b-chat",        "InternLM2.5"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0",    "TinyLlama"),
]

# Models whose activations night 1 already cached -- reuse, do not re-download.
CACHED = {"Qwen2.5-1.5B-Instruct", "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct"}


def disk_free_gb():
    return shutil.disk_usage(".").free / 1e9


def cache_dir_for(repo):
    return (os.path.expanduser("~/.cache/huggingface/hub/models--")
            + repo.replace("/", "--"))


def purge(repo):
    d = cache_dir_for(repo)
    if os.path.isdir(d):
        sz = sum(f.stat().st_size for f in __import__("pathlib").Path(d).rglob("*")
                 if f.is_file()) / 1e9
        shutil.rmtree(d, ignore_errors=True)
        log(f"    purged {repo} from cache ({sz:.2f}GB), free now "
            f"{disk_free_gb():.0f}GB")


# ------------------------------------------------------------- extraction

@torch.no_grad()
def extract_all_layers(model, tok, prompts, bs=4, tag=""):
    """-> (n, n_layers+1, d) float16, held in RAM only."""
    texts = format_prompts(tok, prompts)
    out = []
    t = Timer()
    for i in range(0, len(texts), bs):
        enc = tokenize(tok, texts[i:i + bs]).to(DEVICE)
        hs = model(**enc, output_hidden_states=True).hidden_states
        vec = torch.stack([h[:, -1, :].float().cpu() for h in hs], dim=1)
        out.append(vec.numpy().astype(np.float16))
        del hs, vec, enc
        done = min(i + bs, len(texts))
        if done % 200 < bs:
            log(f"      {tag} {done}/{len(texts)} {t} rss={mem_gb():.1f}GB",
                quiet=True)
    return np.concatenate(out, axis=0)


def layer_features(H, B, cats, layer):
    """Night-1 direction matrix + its degeneracy diagnostic at one layer."""
    bmean = B[:, layer, :].astype(np.float32).mean(axis=0)
    order = sorted(set(cats))
    raw = np.stack([H[[i for i, x in enumerate(cats) if x == c], layer, :]
                    .astype(np.float32).mean(axis=0) - bmean for c in order])
    norms = np.linalg.norm(raw, axis=1)
    R = (raw / (norms[:, None] + 1e-8)).astype(np.float32)
    n_zero = int((norms < 1e-6).sum())
    S = np.linalg.svd(R, compute_uv=False)
    p = S ** 2
    var = p / p.sum()
    U, Sv, Vt = np.linalg.svd(R, full_matrices=False)
    pc1 = Vt[0]
    if float(np.dot(R.mean(0), pc1)) < 0:
        pc1 = -pc1
    cos = R @ pc1
    nz = var[var > 1e-12]
    return {
        "layer": layer,
        "top1_var": float(var[0]),
        "top3_var": float(var[:3].sum()),
        "mean_cos_pc1": float(cos.mean()),
        "min_cos_pc1": float(cos.min()),
        "effective_rank": float(np.exp(-(nz * np.log(nz)).sum())),
        "degenerate": bool(n_zero > 0),
        "n_zero_rows": n_zero,
        "median_raw_norm": float(np.median(norms)),
    }


def analyse(H, B, cats):
    """Night-1 peak rule: search the SECOND HALF only, skipping degenerate
    layers. The layer-0 guard is applied to every model, no exceptions."""
    n_layers = H.shape[1]
    profile = [layer_features(H, B, cats, l) for l in range(n_layers)]
    lo = n_layers // 2
    cand = [s for s in profile[lo:] if not s["degenerate"]]
    if not cand:
        raise RuntimeError("no non-degenerate layer in the second half")
    peak = max(cand, key=lambda s: s["top1_var"])
    naive = max(profile, key=lambda s: s["top1_var"])
    return {
        "n_layers": n_layers,
        "d_model": int(H.shape[2]),
        "peak_layer": peak["layer"],
        "peak": peak,
        "naive_argmax_layer": naive["layer"],
        "degenerate_layers": [s["layer"] for s in profile if s["degenerate"]],
        "layer_profile": [{k: s[k] for k in
                           ("layer", "top1_var", "mean_cos_pc1", "degenerate")}
                          for s in profile],
    }


def main():
    log("=" * 70)
    log(f"N4 PHASE A — widen clean prior | disk free {disk_free_gb():.0f}GB")
    harmful, cats, benign = load_data()
    out = read_json(OUT, {"models": {}, "failures": {}})

    for repo, family in TARGETS:
        sl = slug(repo)
        if sl in out["models"] or sl in out["failures"]:
            log(f"  {sl}: cached result, skipping")
            continue
        free = disk_free_gb()
        log(f"  {sl} ({family}) | disk free {free:.0f}GB")
        if free < MIN_FREE_GB:
            log(f"    WARNING free disk {free:.0f}GB < {MIN_FREE_GB}GB — purging cache")
            for r, _ in TARGETS:
                if slug(r) not in CACHED:
                    purge(r)

        t = Timer()
        model = tok = None
        try:
            if sl in CACHED and (ACTS / sl / "harmful.npy").exists():
                log("    reusing night-1 cached activations")
                H = np.load(ACTS / sl / "harmful.npy")
                B = np.load(ACTS / sl / "benign.npy")
            else:
                model, tok = load_model(repo)
                if tok.chat_template is None:
                    raise RuntimeError("tokenizer has no chat template")
                log(f"    loaded, mps={mps_gb():.2f}GB")
                H = extract_all_layers(model, tok, harmful, tag="harm")
                B = extract_all_layers(model, tok, benign, tag="ben")

            res = analyse(H, B, cats)
            res.update({"repo": repo, "family": family,
                        "top1_var": res["peak"]["top1_var"],
                        "mean_cos_pc1": res["peak"]["mean_cos_pc1"],
                        "reused_cached_activations": sl in CACHED,
                        "seconds": round(t.elapsed, 1)})
            out["models"][sl] = res
            write_json(OUT, out)
            log(f"    peak L{res['peak_layer']}/{res['n_layers']-1} "
                f"top1={res['top1_var']:.4f} cos={res['mean_cos_pc1']:.4f} "
                f"(naive argmax L{res['naive_argmax_layer']}, "
                f"{len(res['degenerate_layers'])} degenerate layers) [{t}]")
            del H, B
        except Exception as e:
            log(f"    FAILED {type(e).__name__}: {str(e)[:200]}")
            out["failures"][sl] = {"repo": repo, "family": family,
                                   "error": f"{type(e).__name__}: {str(e)[:300]}"}
            write_json(OUT, out)
        finally:
            b = mps_gb()
            if model is not None:
                del model
            if tok is not None:
                del tok
            gc.collect()
            free_memory(sl, before=b)
            if sl not in CACHED:
                purge(repo)

    ok = out["models"]
    fams = sorted({v["family"] for v in ok.values()})
    log(f"PHASE A complete: {len(ok)} models, {len(fams)} families, "
        f"{len(out['failures'])} failures | disk free {disk_free_gb():.0f}GB")


if __name__ == "__main__":
    main()
