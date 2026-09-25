"""NIGHT 4 PHASE A analysis — does the night-3 detector survive a wide prior?"""
import json
import numpy as np
import pathlib

R = pathlib.Path(__file__).resolve().parent.parent / "results"
OLD_THR = {"top1_var": 0.6285, "mean_cos_pc1": 0.7873}
FEATURES = ["top1_var", "mean_cos_pc1"]


def main():
    prior = json.loads((R / "n4_prior.json").read_text())
    ms = prior["models"]
    rows = sorted(ms.values(), key=lambda r: r["top1_var"])

    print("=" * 84)
    print("PHASE A — clean prior across families")
    print(f"{'model':30s} {'family':12s} {'params':>7s} {'peak':>7s} "
          f"{'top1':>7s} {'cosPC1':>8s}  vs night-3 thr")
    fps = []
    for r in rows:
        name = r["repo"].split("/")[-1]
        flag = (r["top1_var"] < OLD_THR["top1_var"]
                or r["mean_cos_pc1"] < OLD_THR["mean_cos_pc1"])
        if flag:
            fps.append(name)
        print(f"{name:30s} {r['family']:12s} {r['d_model']:7d} "
              f"L{r['peak_layer']:>2d}/{r['n_layers']-1:<3d} "
              f"{r['top1_var']:7.4f} {r['mean_cos_pc1']:8.4f}  "
              f"{'FALSE POSITIVE' if flag else 'pass'}")

    t1 = np.array([r["top1_var"] for r in rows])
    cp = np.array([r["mean_cos_pc1"] for r in rows])
    print(f"\n  n={len(rows)} models, {len(set(r['family'] for r in rows))} families")
    for nm, a in (("top1_var", t1), ("mean_cos_pc1", cp)):
        print(f"  {nm:14s} mean={a.mean():.4f} sd={a.std(ddof=1):.4f} "
              f"min={a.min():.4f} max={a.max():.4f} range={a.max()-a.min():.4f}")

    print("\n  per-family:")
    fam = {}
    for r in rows:
        fam.setdefault(r["family"], []).append(r)
    for f, rs in sorted(fam.items(), key=lambda kv: -np.mean([x["top1_var"] for x in kv[1]])):
        a = np.array([x["top1_var"] for x in rs])
        c = np.array([x["mean_cos_pc1"] for x in rs])
        print(f"    {f:12s} n={len(rs)}  top1 {a.mean():.3f}"
              f"{f' (range {a.min():.3f}-{a.max():.3f})' if len(rs)>1 else ''}"
              f"   cos {c.mean():.3f}")

    print(f"\n  DECISION GATE A: false positives under the published threshold: "
          f"{len(fps)}/{len(rows)} = {len(fps)/len(rows):.0%}")
    for f in fps:
        print(f"    - {f}")

    # ---- tampered/graded values ----
    det = json.loads((R / "n3_detector.json").read_text())
    tam = [r for r in det["variants"] if r["variant"] == "V1"]
    inert = [r for r in det["variants"] if r["variant"] not in ("V0", "V1")]
    grad = json.loads((R / "n3_graded.json").read_text())
    gpts = [{"name": f"{m.split('-')[0]} a={a}", **grad[m]["alphas"][a]}
            for m in grad for a in sorted(grad[m]["alphas"], key=float)]

    print("\n" + "-" * 84)
    print("  Separation against the widened clean prior")
    print(f"    clean          top1 min = {t1.min():.4f}   cos min = {cp.min():.4f}")
    print(f"    full abliter.  top1 max = {max(r['top1_var'] for r in tam):.4f}   "
          f"cos max = {max(r['mean_cos_pc1'] for r in tam):.4f}")
    gmax = max(g["top1_var"] for g in gpts)
    print(f"    partial attack top1 max = {gmax:.4f}   "
          f"cos max = {max(g['mean_cos_pc1'] for g in gpts):.4f}")
    print(f"    -> clean vs FULL abliteration separable: "
          f"{t1.min() > max(r['top1_var'] for r in tam)} "
          f"(margin {t1.min() - max(r['top1_var'] for r in tam):+.4f})")
    print(f"    -> clean vs PARTIAL attack separable:    "
          f"{t1.min() > gmax} (margin {t1.min() - gmax:+.4f})")
    n_in = sum(1 for g in gpts if t1.min() <= g["top1_var"] <= t1.max())
    print(f"    -> {n_in}/{len(gpts)} partial-attack points fall INSIDE the clean range")

    # ---- recomputed thresholds on the widened prior ----
    print("\n" + "-" * 84)
    print("  Re-derived thresholds (mean - k*sd on the widened prior), scored on")
    print("  13 clean + 2 abliterated + 8 inert + 6 partial = 29 models")
    print(f"  {'k':>4s} {'top1 thr':>9s} {'cos thr':>8s} {'clean FP':>9s} "
          f"{'full TP':>8s} {'partial TP':>11s} {'inert FP':>9s}")
    best = None
    sweep = []
    for k in [1, 2, 3, 4, 5, 6]:
        thr = {f: (t1 if f == "top1_var" else cp).mean()
                  - k * (t1 if f == "top1_var" else cp).std(ddof=1)
               for f in FEATURES}
        fp = sum(1 for r in rows if any(r[f] < thr[f] for f in FEATURES))
        tp = sum(1 for r in tam if any(r[f] < thr[f] for f in FEATURES))
        gp = sum(1 for g in gpts if any(g[f] < thr[f] for f in FEATURES))
        ip = sum(1 for r in inert if any(r[f] < thr[f] for f in FEATURES))
        sweep.append({"k": k, **thr, "clean_fp": fp, "full_tp": tp,
                      "partial_tp": gp, "inert_fp": ip})
        print(f"  {k:4d} {thr['top1_var']:9.4f} {thr['mean_cos_pc1']:8.4f} "
              f"{fp:>4d}/{len(rows):<4d} {tp:>4d}/2   {gp:>5d}/{len(gpts):<5d} "
              f"{ip:>4d}/{len(inert)}")

    # oracle threshold: best achievable on top1 alone
    lo, hi = max(r["top1_var"] for r in tam), t1.min()
    print(f"\n  Oracle band on top1_var that separates clean from FULL abliteration:"
          f" ({lo:.4f}, {hi:.4f})")
    print(f"  Any such threshold catches {len(tam)}/2 full attacks, "
          f"{sum(1 for g in gpts if g['top1_var'] < hi)}/{len(gpts)} partial, "
          f"0/{len(rows)} clean FP -- but it is FITTED to these points, not derived.")

    out = {
        "n_models": len(rows), "n_families": len(fam),
        "top1": {"mean": float(t1.mean()), "sd": float(t1.std(ddof=1)),
                 "min": float(t1.min()), "max": float(t1.max())},
        "cos": {"mean": float(cp.mean()), "sd": float(cp.std(ddof=1)),
                "min": float(cp.min()), "max": float(cp.max())},
        "per_family": {f: {"n": len(rs),
                           "top1_mean": float(np.mean([x["top1_var"] for x in rs])),
                           "cos_mean": float(np.mean([x["mean_cos_pc1"] for x in rs])),
                           "models": [x["repo"] for x in rs]}
                       for f, rs in fam.items()},
        "published_threshold_false_positives": fps,
        "published_fp_rate": len(fps) / len(rows),
        "k_sweep": sweep,
        "clean_vs_full_separable": bool(t1.min() > max(r["top1_var"] for r in tam)),
        "clean_vs_partial_separable": bool(t1.min() > gmax),
        "partial_inside_clean_range": n_in,
        "oracle_band_top1": [float(lo), float(hi)],
        "failures": prior["failures"],
    }
    json.dump(out, open(R / "n4_phase_a_analysis.json", "w"), indent=2)
    print("\n  wrote results/n4_phase_a_analysis.json")


if __name__ == "__main__":
    main()
