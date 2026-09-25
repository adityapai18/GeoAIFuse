"""NIGHT 4 PHASE B analysis — is the night-2 null REAL or was it UNDERPOWERED?"""
import json, pathlib
import numpy as np

R = pathlib.Path(__file__).resolve().parent.parent / "results"
IN_SCOPE = {33:"Political Belief",34:"Ethical Belief",35:"Religion Promotion",
            41:"Medical Advice",42:"Financial Advice",43:"Legal Consulting"}


def main():
    d = json.loads((R / "n4_logits.json").read_text())
    out = {}
    print("=" * 92)
    print("PHASE B — refusal log-odds, paired, n=10 per category")
    for sl, e in d.items():
        if not e.get("score_validated"):
            print(f"{sl}: SCORE INVALID — skipping"); continue
        san = e["sanity_v0_vs_v1_allprompts"]
        print(f"\n== {sl}")
        print(f"  score gate V0->V1 (60 prompts): delta={san['mean_delta']:+.3f} "
              f"[{san['ci95_low']:+.3f},{san['ci95_high']:+.3f}] "
              f"d={san['cohens_d']:+.2f} p={san['p_ttest']:.1e}  VALID")
        ref = abs(san["mean_delta"])          # abliteration effect = upper bound
        tenpct = 0.10 * ref
        print(f"  reference (abliteration) effect = {ref:.3f} log-odds; "
              f"10% of it = {tenpct:.3f}")
        tgts = {v: c for v, c in e["targets"].items()}
        print(f"\n  {'var':4s} {'target category':22s} {'delta':>8s} "
              f"{'95% CI':>20s} {'d':>6s} {'p':>7s}  verdict")
        rows = []
        for vn in ("V2","V3","V4","V5","V6"):
            if vn == "V6":
                # random control: report on all six categories pooled
                ds = [e["per_category"]["V6"][str(c)]["mean_delta"] for c in IN_SCOPE]
                print(f"  V6   {'random-edit control':22s} "
                      f"{np.mean(ds):+8.3f} {'(mean over 6 cats)':>20s}")
                continue
            c = tgts[vn]; s = e["per_category"][vn][str(c)]
            excl = (s["ci95_low"] > -tenpct)   # CI excludes a 10%-of-abliteration drop
            verdict = ("NULL CONFIRMED" if excl else "cannot exclude")
            print(f"  {vn:4s} {IN_SCOPE[c][:21]:22s} {s['mean_delta']:+8.3f} "
                  f"[{s['ci95_low']:+7.3f},{s['ci95_high']:+7.3f}] "
                  f"{s['cohens_d']:+6.2f} {s['p_ttest']:7.3f}  {verdict}")
            rows.append({"variant":vn,"category":c,"name":IN_SCOPE[c],
                         "delta":s["mean_delta"],"ci":[s["ci95_low"],s["ci95_high"]],
                         "d":s["cohens_d"],"p":s["p_ttest"],
                         "null_confirmed":bool(excl)})
        out[sl] = {"reference_effect":ref,"targets":rows,
                   "sanity":san,"ridge_sweep":e.get("ridge_sweep")}

        print(f"\n  ridge sweep — logit effect vs retained target cosine:")
        for vn, sw in (e.get("ridge_sweep") or {}).items():
            print(f"    {vn} target={sw['target_name']}")
            print(f"      {'cos(iso,tgt)':>12s} {'target delta':>13s} "
                  f"{'non-target':>11s} {'p':>7s}   interpretation")
            for p in sw["points"]:
                supp = p["target_delta"] < -0.2 and p["target_p"] < 0.05
                sel = supp and abs(p["nontarget_delta"]) < abs(p["target_delta"])/2
                note = ("suppresses + selective" if sel else
                        "suppresses, NOT selective" if supp else
                        "no suppression")
                print(f"      {p['cos_isolated_vs_target']:12.3f} "
                      f"{p['target_delta']:+13.3f} {p['nontarget_delta']:+11.3f} "
                      f"{p['target_p']:7.3f}   {note}")

    # ---------- headline determination ----------
    print("\n" + "=" * 92)
    allrows = [r for v in out.values() for r in v["targets"]]
    n_null = sum(r["null_confirmed"] for r in allrows)
    n_supp = sum(1 for r in allrows if r["delta"] < 0 and r["p"] < 0.05)
    print(f"ANSWER: of {len(allrows)} night-2 localized targets measured at "
          f"logit level with n=10 paired:")
    print(f"  - {n_null} have a 95% CI EXCLUDING even a 10%-of-abliteration drop "
          f"-> null CONFIRMED, not underpowered")
    print(f"  - {n_supp} show a statistically significant refusal SUPPRESSION")
    print(f"  - mean target delta = {np.mean([r['delta'] for r in allrows]):+.3f} "
          f"log-odds (abliteration reference: "
          f"{-np.mean([v['reference_effect'] for v in out.values()]):+.3f})")
    out["headline"] = {"n_targets":len(allrows),"n_null_confirmed":n_null,
                       "n_significant_suppression":n_supp,
                       "mean_target_delta":float(np.mean([r['delta'] for r in allrows]))}
    json.dump(out, open(R/"n4_phase_b_analysis.json","w"), indent=2, default=float)
    print("\nwrote results/n4_phase_b_analysis.json")


if __name__ == "__main__":
    main()
