"""STAGE 2 + 3 — per-category direction matrix and its structure.

Stage 2: r_c^(l) = mean(activations of category c) - mean(benign pool),
         L2-normalized, stacked into R^(l) of shape (45, d_model).
Stage 3: SVD spectrum, effective rank, pairwise cosines, per-category
         alignment with PC1, and the layer profile that locates the peak band.

All linear algebra on CPU in numpy float32 -- MPS linalg is unreliable.
"""
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (ACTS, CATEGORY_NAMES, FIGS, MODELS, RESULTS, Timer,
                    load_data, log, read_json, slug, write_json)

STRUCT = RESULTS / "structure.json"


def direction_matrix(harm, ben, cats, layer, with_diag=False):
    """Stage 2. -> (45, d_model) float32, rows L2-normalized.

    Also reports how much raw signal the layer actually carries. At layer 0
    the last token is the *same* chat-template token for every prompt, so its
    embedding is identical and mean(category) - mean(benign) cancels to
    exactly zero for nearly every category. Normalizing that with an epsilon
    manufactures a spectrum out of nothing, so callers must check `degenerate`
    rather than trusting the variance numbers.
    """
    bmean = ben[:, layer, :].astype(np.float32).mean(axis=0)
    raw = []
    for c in sorted(set(cats)):
        idx = [i for i, x in enumerate(cats) if x == c]
        raw.append(harm[idx, layer, :].astype(np.float32).mean(axis=0) - bmean)
    raw = np.stack(raw)
    norms = np.linalg.norm(raw, axis=1)
    R = (raw / (norms[:, None] + 1e-8)).astype(np.float32)
    if not with_diag:
        return R
    act_norm = float(np.linalg.norm(
        harm[:, layer, :].astype(np.float32), axis=1).mean())
    n_zero = int((norms < 1e-6).sum())
    diag = {
        "median_raw_norm": float(np.median(norms)),
        "n_zero_rows": n_zero,
        # Separation relative to the size of the representation itself.
        "relative_separation": float(np.median(norms) / (act_norm + 1e-8)),
        "degenerate": bool(n_zero > 0),
    }
    return R, diag


def spectrum_stats(R):
    """SVD-based structure summary for one layer."""
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    p = S ** 2
    var = p / p.sum()
    # Participation ratio of the eigenvalue spectrum: (sum l)^2 / sum l^2.
    pr = (p.sum() ** 2) / (p ** 2).sum()
    # Effective rank: exp(Shannon entropy of the normalized spectrum).
    nz = var[var > 1e-12]
    eff = float(np.exp(-(nz * np.log(nz)).sum()))
    return {
        "top1": float(var[0]),
        "top3": float(var[:3].sum()),
        "top5": float(var[:5].sum()),
        "participation_ratio": float(pr),
        "effective_rank": eff,
        "singular_values": S[:10].tolist(),
    }, Vt[0].astype(np.float32)


def analyse_model(model_id, cats):
    sl = slug(model_id)
    d = ACTS / sl
    if not (d / "harmful.npy").exists() or not (d / "benign.npy").exists():
        log(f"  {sl}: activations missing, skipping")
        return None

    harm = np.load(d / "harmful.npy")
    ben = np.load(d / "benign.npy")
    n_layers = harm.shape[1]
    log(f"  {sl}: harmful{harm.shape} benign{ben.shape}")

    per_layer, pcs = [], []
    for l in range(n_layers):
        R, diag = direction_matrix(harm, ben, cats, l, with_diag=True)
        st, v1 = spectrum_stats(R)
        st["layer"] = l
        st.update(diag)
        per_layer.append(st)
        pcs.append(v1)

    top1 = [s["top1"] for s in per_layer]
    naive_peak = int(np.argmax(top1))

    # Peak band. A naive argmax lands on the degenerate early layers: layer 0
    # is undefined (see direction_matrix) and layers 1-3 sit on top of it,
    # where every prompt's representation is still ~98% cosine-identical and
    # the raw separation is an order of magnitude smaller than at mid-depth.
    # A single-layer spike adjacent to a degenerate layer is not a "band".
    # We therefore search the second half of the network, which is where
    # refusal is computed in prior work and where the profile shows a genuine
    # broad plateau. The early spike is kept and tested causally in stage 4.
    lo = n_layers // 2
    peak = lo + int(np.argmax(top1[lo:]))
    log(f"  {sl}: peak layer {peak}/{n_layers - 1} top1={top1[peak]:.3f} "
        f"(top3={per_layer[peak]['top3']:.3f}, "
        f"eff_rank={per_layer[peak]['effective_rank']:.1f})")
    if naive_peak != peak:
        log(f"  {sl}: naive argmax was L{naive_peak} "
            f"(top1={top1[naive_peak]:.3f}, "
            f"rel_sep={per_layer[naive_peak]['relative_separation']:.4f} vs "
            f"{per_layer[peak]['relative_separation']:.4f} at L{peak}) "
            f"-- rejected as early-layer degeneracy")
    # Save the early spike direction so stage 4 can show it is not causal.
    early = max(1, naive_peak)
    np.save(RESULTS / f"pc1early_{sl}.npy", pcs[early])

    # Detailed artefacts at the peak layer.
    Rp = direction_matrix(harm, ben, cats, peak)
    v1 = pcs[peak]
    # Orient PC1 so that it points along the refusal directions.
    if float(np.dot(Rp.mean(0), v1)) < 0:
        v1 = -v1
    np.save(RESULTS / f"pc1_{sl}.npy", v1)

    cos_pc1 = Rp @ v1                       # rows are unit norm, v1 is unit
    C = Rp @ Rp.T                           # 45x45 pairwise cosine
    np.save(RESULTS / f"cosine_matrix_{sl}.npy", C)

    order = sorted(set(cats))
    dist = {
        str(c): float(cos_pc1[i]) for i, c in enumerate(order)
    }
    off = np.triu_indices(len(order), k=1)
    entry = {
        "n_layers": n_layers,
        "d_model": int(harm.shape[2]),
        "peak_layer": peak,
        "naive_argmax_layer": naive_peak,
        "early_spike_layer": early,
        "peak_search_restricted_to_layers_from": lo,
        "degenerate_layers": [s["layer"] for s in per_layer if s["degenerate"]],
        "peak_stats": per_layer[peak],
        "layer_profile": per_layer,
        "cos_with_pc1": dist,                # full distribution, saved per spec
        "mean_pairwise_cosine": float(C[off].mean()),
        "median_pairwise_cosine": float(np.median(C[off])),
        "min_pairwise_cosine": float(C[off].min()),
        "categories_furthest_from_core": [
            {"category": int(c), "name": CATEGORY_NAMES[int(c)],
             "cos_pc1": float(cos_pc1[i])}
            for i, c in sorted(enumerate(order), key=lambda kv: cos_pc1[kv[0]])[:8]
        ],
        "categories_closest_to_core": [
            {"category": int(c), "name": CATEGORY_NAMES[int(c)],
             "cos_pc1": float(cos_pc1[i])}
            for i, c in sorted(enumerate(order),
                               key=lambda kv: -cos_pc1[kv[0]])[:8]
        ],
    }
    make_figures(sl, per_layer, C, cos_pc1, order, peak)
    return entry


def make_figures(sl, per_layer, C, cos_pc1, order, peak):
    # Layer profile.
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [s["layer"] for s in per_layer]
    ax.plot(xs, [s["top1"] for s in per_layer], label="top-1", lw=2)
    ax.plot(xs, [s["top3"] for s in per_layer], label="top-3", ls="--")
    ax.plot(xs, [s["top5"] for s in per_layer], label="top-5", ls=":")
    ax.axvline(peak, color="r", alpha=.4, label=f"peak (L{peak})")
    ax.set_xlabel("layer"); ax.set_ylabel("variance explained")
    ax.set_title(f"{sl}: shared-component strength vs depth")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(FIGS / f"layer_profile_{sl}.png", dpi=140); plt.close(fig)

    # 45x45 cosine heatmap.
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"{sl}: pairwise cosine of category refusal directions (L{peak})")
    ax.set_xlabel("category"); ax.set_ylabel("category")
    ax.set_xticks(range(0, len(order), 5))
    ax.set_xticklabels([order[i] for i in range(0, len(order), 5)])
    ax.set_yticks(range(0, len(order), 5))
    ax.set_yticklabels([order[i] for i in range(0, len(order), 5)])
    fig.colorbar(im, ax=ax, shrink=.8); fig.tight_layout()
    fig.savefig(FIGS / f"cosine_heatmap_{sl}.png", dpi=140); plt.close(fig)

    # Per-category alignment with PC1 -- the detection statistic.
    fig, ax = plt.subplots(figsize=(9, 5))
    idx = np.argsort(cos_pc1)
    ax.barh(range(len(order)), cos_pc1[idx], color="steelblue")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(
        [f"{order[i]} {CATEGORY_NAMES[int(order[i])][:34]}" for i in idx],
        fontsize=6.5)
    ax.set_xlabel("cosine with PC1")
    ax.set_title(f"{sl}: per-category alignment with the shared core (L{peak})")
    ax.grid(alpha=.3, axis="x"); fig.tight_layout()
    fig.savefig(FIGS / f"cos_pc1_{sl}.png", dpi=140); plt.close(fig)


def main():
    log("=" * 70)
    log("STAGE 2+3 — DIRECTION MATRIX & STRUCTURE")
    harmful, cats, benign = load_data()
    out = read_json(STRUCT, {})
    for model_id in MODELS:
        t = Timer()
        try:
            entry = analyse_model(model_id, cats)
            if entry:
                out[slug(model_id)] = entry
                write_json(STRUCT, out)
                log(f"  {slug(model_id)} done in {t}")
        except Exception as e:
            log(f"  ERROR on {model_id}: {type(e).__name__}: {e}")
            continue
    log("STAGE 2+3 complete")


if __name__ == "__main__":
    main()
