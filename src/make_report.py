"""Assemble the summary figure and the numbers FINDINGS.md quotes.

Reads results/structure.json + results/metrics.json; writes the gate verdict
and a per-condition comparison figure. No model loading.
"""
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (CATEGORY_NAMES, FIGS, MODELS, RESULTS, log, read_json,
                    slug, write_json)

STRUCT = RESULTS / "structure.json"
METRICS = RESULTS / "metrics.json"
CONDS = ["baseline", "ablate_pc1", "ablate_random", "ablate_early_spike"]


def gate_verdict(m):
    """The GATE, stated per spec."""
    b = m.get("baseline", {}).get("refusal_rate")
    p = m.get("ablate_pc1", {}).get("refusal_rate")
    r = m.get("ablate_random", {}).get("refusal_rate")
    if b is None or p is None or r is None:
        return "INCOMPLETE", {}
    drop_pc1 = b - p
    drop_rnd = b - r
    nums = {"baseline": b, "ablate_pc1": p, "ablate_random": r,
            "drop_pc1": drop_pc1, "drop_random": drop_rnd}

    # A refusal drop only counts as evidence if the model is still working.
    arc = m.get("arc_easy", {})
    a_b, a_p = arc.get("baseline"), arc.get("ablate_pc1")
    arc_drop = None
    if a_b is not None and a_p is not None:
        arc_drop = a_b - a_p
        nums["arc_baseline"], nums["arc_ablated"] = a_b, a_p
        nums["arc_drop"] = arc_drop
    # Relative loss matters more than absolute at different capability levels.
    broken = arc_drop is not None and a_b > 0 and (arc_drop / a_b) > 0.25

    # "collapses broadly" vs "barely moves", and the two-move-alike failure.
    if drop_pc1 >= 0.30 and drop_rnd <= 0.10:
        v = "CORE IS REAL AND CAUSAL — PROCEED"
    elif drop_pc1 >= 0.15 and drop_rnd < drop_pc1 / 2:
        v = "CORE IS CAUSAL BUT PARTIAL — PROCEED WITH CAVEAT"
    elif abs(drop_pc1 - drop_rnd) < 0.10 and drop_pc1 > 0.15:
        v = "BUG IN THE ABLATION HOOK — both conditions move alike"
    elif drop_pc1 < 0.15:
        v = "CORE IS NOT CAUSALLY LOAD-BEARING"
    else:
        v = "AMBIGUOUS"

    if broken:
        v += (f" [CONFOUNDED: ARC-easy fell {a_b:.2f} -> {a_p:.2f}, so the "
              "refusal drop cannot be cleanly attributed to refusal removal]")
    nums["capability_confound"] = bool(broken)
    return v, nums


def summary_figure(metrics):
    models = [slug(m) for m in MODELS if slug(m) in metrics]
    if not models:
        return
    present = [c for c in CONDS
               if any(c in metrics[m] for m in models)]
    fig, ax = plt.subplots(figsize=(1.9 * len(models) + 3, 4.2))
    w = 0.8 / len(present)
    x = np.arange(len(models))
    colors = {"baseline": "#4C6EF5", "ablate_pc1": "#E03131",
              "ablate_random": "#868E96", "ablate_early_spike": "#F59F00"}
    for k, c in enumerate(present):
        vals = [metrics[m].get(c, {}).get("refusal_rate", np.nan) for m in models]
        bars = ax.bar(x + k * w - 0.4 + w / 2, vals, w,
                      label=c, color=colors.get(c))
        for b_, v_ in zip(bars, vals):
            if np.isfinite(v_):
                ax.text(b_.get_x() + b_.get_width() / 2, v_ + .015,
                        f"{v_:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("refusal rate"); ax.set_ylim(0, 1.08)
    ax.set_title("Refusal rate by intervention (135 prompts, 3 per category)")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGS / "causal_summary.png", dpi=140)
    plt.close(fig)


def cross_model_pc1(struct):
    """Is the peak-layer top-1 variance consistent across models?"""
    return {sl: {"peak_layer": e["peak_layer"],
                 "n_layers": e["n_layers"],
                 "depth_frac": round(e["peak_layer"] / (e["n_layers"] - 1), 2),
                 "top1": e["peak_stats"]["top1"],
                 "top3": e["peak_stats"]["top3"],
                 "top5": e["peak_stats"]["top5"],
                 "effective_rank": e["peak_stats"]["effective_rank"],
                 "participation_ratio": e["peak_stats"]["participation_ratio"],
                 "mean_pairwise_cosine": e["mean_pairwise_cosine"]}
            for sl, e in struct.items()}


def main():
    struct = read_json(STRUCT, {})
    metrics = read_json(METRICS, {})
    summary = {"structure": cross_model_pc1(struct), "gate": {}}
    for sl, m in metrics.items():
        v, nums = gate_verdict(m)
        summary["gate"][sl] = {"verdict": v, **nums,
                               "arc_easy": m.get("arc_easy", {}),
                               "ablate_early_spike":
                                   m.get("ablate_early_spike", {})
                                   .get("refusal_rate")}
        log(f"  GATE {sl}: {v}")
        if nums:
            log(f"    baseline={nums['baseline']:.3f} "
                f"pc1={nums['ablate_pc1']:.3f} "
                f"random={nums['ablate_random']:.3f}")
    summary_figure(metrics)
    write_json(RESULTS / "summary.json", summary)
    log("report artefacts written")


if __name__ == "__main__":
    main()
