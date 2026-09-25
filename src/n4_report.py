"""NIGHT 4 — figures for phases A, B, C."""
import json, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parent.parent / "results"
F = R / "figures"
THR = {"top1_var": 0.6285, "mean_cos_pc1": 0.7873}


def fig_prior():
    pa = json.loads((R/"n4_prior.json").read_text())["models"]
    det = json.loads((R/"n3_detector.json").read_text())["variants"]
    grad = json.loads((R/"n3_graded.json").read_text())
    adv = json.loads((R/"n4_adversary.json").read_text())["betas"]

    rows = sorted(pa.values(), key=lambda r: r["top1_var"])
    fig, ax = plt.subplots(figsize=(9.5, 6))
    fams = sorted({r["family"] for r in rows})
    cmap = plt.get_cmap("tab10")
    fc = {f: cmap(i % 10) for i, f in enumerate(fams)}
    for r in rows:
        ax.scatter(r["top1_var"], r["mean_cos_pc1"], color=fc[r["family"]],
                   s=110, edgecolors="k", linewidths=.6, zorder=3)
        ax.annotate(r["repo"].split("/")[-1][:18], (r["top1_var"], r["mean_cos_pc1"]),
                    fontsize=6, xytext=(4, 4), textcoords="offset points")
    for r in det:
        if r["variant"] == "V1":
            ax.scatter(r["top1_var"], r["mean_cos_pc1"], marker="X", s=190,
                       color="#E03131", edgecolors="k", zorder=4,
                       label="full abliteration" if r["model"].startswith("Qwen") else None)
    gx = [grad[m]["alphas"][a]["top1_var"] for m in grad for a in grad[m]["alphas"]]
    gy = [grad[m]["alphas"][a]["mean_cos_pc1"] for m in grad for a in grad[m]["alphas"]]
    ax.scatter(gx, gy, marker="^", s=110, color="#F59F00", edgecolors="k",
               zorder=4, label="partial abliteration")
    ax_ = [v for v in adv.values() if "error" not in v]
    ax.scatter([v["top1_var"] for v in ax_], [v["mean_cos_pc1"] for v in ax_],
               marker="*", s=260, color="#7048E8", edgecolors="k", zorder=5,
               label="spectrum-preserving adversary")
    ax.axvline(THR["top1_var"], color="r", ls="--", alpha=.7)
    ax.axhline(THR["mean_cos_pc1"], color="r", ls="--", alpha=.7)
    ax.add_patch(plt.Rectangle((0.30, 0.40), THR["top1_var"]-0.30,
                               THR["mean_cos_pc1"]-0.40, color="r", alpha=.06, zorder=0))
    ax.text(0.33, 0.43, "night-3 FLAG region", color="r", fontsize=9)
    for f in fams:
        ax.scatter([], [], color=fc[f], s=80, edgecolors="k", label=f)
    ax.set_xlabel("top-1 variance explained"); ax.set_ylabel("mean cosine with PC1")
    ax.set_title("Night 4: the night-3 detector against 13 clean models,\n"
                 "graded attacks, and a spectrum-preserving adversary", fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc="lower right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(F/"n4_prior_and_attacks.png", dpi=140); plt.close(fig)


def fig_ridge():
    b = json.loads((R/"n4_logits.json").read_text())
    models = [m for m in b if b[m].get("ridge_sweep")]
    fig, axes = plt.subplots(1, len(models), figsize=(6.4*len(models), 4.6), squeeze=False)
    for ax, m in zip(axes[0], models):
        for vn, sw in b[m]["ridge_sweep"].items():
            p = sw["points"]
            x = [q["cos_isolated_vs_target"] for q in p]
            y = [q["target_delta"] for q in p]
            lo = [q["target_ci"][0] for q in p]; hi = [q["target_ci"][1] for q in p]
            ax.errorbar(x, y, yerr=[np.array(y)-np.array(lo), np.array(hi)-np.array(y)],
                        marker="o", capsize=3, label=f"{vn} {sw['target_name'][:20]}")
            ax.plot(x, [q["nontarget_delta"] for q in p], ls=":", marker=".",
                    alpha=.6, label=f"{vn} non-target")
        ax.axhline(0, color="k", lw=.8)
        ref = -b[m]["sanity_v0_vs_v1_allprompts"]["mean_delta"]
        ax.axhline(-0.1*ref, color="r", ls="--", alpha=.6,
                   label="10% of abliteration effect")
        ax.set_xlabel("cos(v_isolated, v_target)  — weaker isolation ->")
        ax.set_ylabel("Δ refusal log-odds (negative = suppressed)")
        ax.set_title(m, fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.suptitle("Night 4 Phase B: isolation buys selectivity at the cost of all force",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(F/"n4_ridge_tradeoff.png", dpi=140); plt.close(fig)


def fig_adversary():
    a = json.loads((R/"n4_adversary.json").read_text())
    pts = [v for v in a["betas"].values() if "error" not in v]
    pts.sort(key=lambda v: v["beta"])
    b = [v["beta"] for v in pts]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(b, [v["refusal_rate"] for v in pts], "o-", color="#E03131", lw=2,
            label="refusal rate")
    ax.plot(b, [v["top1_var"] for v in pts], "s--", color="#4C6EF5", lw=2,
            label="top-1 variance (detector feature)")
    ax.plot(b, [v["arc"] for v in pts], "^:", color="#2F9E44", label="ARC-easy")
    ax.axhline(THR["top1_var"], color="#4C6EF5", ls=":", alpha=.8)
    ax.text(b[0], THR["top1_var"]+.015, "detector threshold", color="#4C6EF5", fontsize=8)
    ax.axhline(0.742, color="#E03131", ls=":", alpha=.5)
    ax.text(b[0], 0.742+.015, "clean refusal 0.742", color="#E03131", fontsize=8)
    for v in pts:
        ax.annotate(v["verdict"], (v["beta"], v["top1_var"]), fontsize=8,
                    xytext=(0, 8), textcoords="offset points", ha="center",
                    color="#E03131" if v["verdict"]=="FLAG" else "#2F9E44",
                    weight="bold")
    ax.set_xlabel("beta  (rotation of a substitute direction, alpha=1.0 fixed)")
    ax.set_ylabel("rate / variance"); ax.set_xscale("log")
    ax.set_xticks(b); ax.set_xticklabels([str(x) for x in b])
    ax.set_title("Night 4 Phase C: an adversary that suppresses refusal\n"
                 "while restoring the spectrum the detector reads", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(F/"n4_adversary.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    fig_prior(); fig_ridge(); fig_adversary()
    print("wrote n4_prior_and_attacks.png, n4_ridge_tradeoff.png, n4_adversary.png")
