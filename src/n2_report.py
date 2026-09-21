"""NIGHT 2 — assemble the headline table and summary figure."""
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIGS, RESULTS, log, read_json, slug, write_json
from n2_organisms import PRIMARY_MODELS

METRICS = RESULTS / "n2_metrics.json"
CLSVAL = RESULTS / "classifier_val" / "classifier_metrics.json"


def uncertainty_band():
    """Classifier error -> absolute uncertainty on any refusal rate.

    Combines measured false-positive and false-negative rates on the 200
    hand-labeled generations with binomial sampling error at n=450.
    """
    m = read_json(CLSVAL, {}).get("hardened", {})
    fp = m.get("fp_rate", 0.0)
    fn = 1 - m.get("recall", 1.0)
    cls_err = max(fp, fn)
    samp = np.sqrt(0.25 / 450)
    return float(np.sqrt(cls_err ** 2 + samp ** 2)), fp, fn


def rows_for(sl, entry, band):
    v = entry["variants"]
    v0 = v.get("V0")
    if not v0:
        return []
    out = []
    for vn in ["V0", "V1", "V2", "V3", "V4", "V5"]:
        r = v.get(vn)
        if not r or not r.get("complete"):
            continue
        spec = r["spec"]
        tgt = spec.get("target")
        if tgt is None:
            tgt_pre = tgt_post = None
            nont_pre = v0["refusal_rate"]
            nont_post = r["refusal_rate"]
        else:
            k = str(tgt)
            tgt_pre, tgt_post = v0["per_category"][k], r["per_category"][k]
            others = [c for c in v0["per_category"] if c != k]
            nont_pre = float(np.mean([v0["per_category"][c] for c in others]))
            nont_post = float(np.mean([r["per_category"][c] for c in others]))
        out.append({
            "variant": vn,
            "kind": spec["kind"],
            "target": spec.get("target_name", "-"),
            "core_side": spec.get("core_side", "-"),
            "target_pre": tgt_pre, "target_post": tgt_post,
            "nontarget_pre": nont_pre, "nontarget_post": nont_post,
            "selectivity": r.get("selectivity"),
            "arc_pre": v0["arc"], "arc_post": r["arc"],
            "tier1_sigma": r["ams"]["tier1_sigma"],
            "tier2_cos": r["ams"]["tier2_cos_vs_v0"],
            "ams_verdict": r["ams"]["verdict"],
            "ams_reasons": r["ams"].get("reasons", []),
            "incoherent": r.get("n_incoherent", 0),
            "band": band,
        })
    return out


def fmt(x, n=3):
    return "-" if x is None else f"{x:.{n}f}"


def main():
    metrics = read_json(METRICS, {})
    band, fp, fn = uncertainty_band()
    log("=" * 70)
    log(f"N2 REPORT — refusal-rate uncertainty +/-{band:.3f} "
        f"(classifier FP {fp:.3f}, FN {fn:.3f}, n=450 sampling)")

    summary = {"uncertainty_band": band, "classifier_fp": fp,
               "classifier_fn": fn, "models": {}}
    for mid in PRIMARY_MODELS:
        sl = slug(mid)
        if sl not in metrics:
            continue
        rows = rows_for(sl, metrics[sl], metrics[sl].get("peak_layer"))
        summary["models"][sl] = rows
        log(f"\n== {sl}")
        log(f"{'var':4s} {'kind':20s} {'target':28s} {'side':9s} "
            f"{'tgt pre>post':13s} {'non-tgt pre>post':17s} {'sel':>7s} "
            f"{'ARC':13s} {'t1sig':>6s} {'t2cos':>6s}  verdict")
        for r in rows:
            log(f"{r['variant']:4s} {r['kind']:20s} {r['target'][:27]:28s} "
                f"{r['core_side']:9s} "
                f"{fmt(r['target_pre'],2)+'>'+fmt(r['target_post'],2):13s} "
                f"{fmt(r['nontarget_pre'])+'>'+fmt(r['nontarget_post']):17s} "
                f"{fmt(r['selectivity'],2):>7s} "
                f"{fmt(r['arc_pre'],2)+'>'+fmt(r['arc_post'],2):13s} "
                f"{r['tier1_sigma']:6.2f} {r['tier2_cos']:6.3f}  "
                f"{r['ams_verdict']}")

    # Gate check.
    gate = {}
    for sl, rows in summary["models"].items():
        v1 = next((r for r in rows if r["variant"] == "V1"), None)
        gate[sl] = (v1["ams_verdict"] if v1 else "MISSING")
    summary["validation_gate_v1_flagged"] = gate
    log(f"\n  VALIDATION GATE (V1 must be FLAG): {gate}")

    figure(summary)
    write_json(RESULTS / "n2_summary.json", summary)
    log("  wrote n2_summary.json")


def figure(summary):
    models = list(summary["models"])
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 4.6),
                             squeeze=False)
    for ax, sl in zip(axes[0], models):
        rows = summary["models"][sl]
        labels, tgt, non = [], [], []
        for r in rows:
            if r["variant"] == "V0":
                continue
            labels.append(f"{r['variant']}\n{r['target'][:16]}")
            tgt.append((r["target_pre"] - r["target_post"])
                       if r["target_pre"] is not None else np.nan)
            non.append(r["nontarget_pre"] - r["nontarget_post"])
        x = np.arange(len(labels))
        ax.bar(x - 0.2, tgt, 0.4, label="target-category drop", color="#E03131")
        ax.bar(x + 0.2, non, 0.4, label="non-target mean drop", color="#868E96")
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("refusal-rate drop vs V0")
        ax.set_title(sl, fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.suptitle("Night 2: concept-localized organisms vs global abliteration",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "n2_organism_summary.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
