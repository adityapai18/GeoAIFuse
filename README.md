# Category-resolved refusal geometry

Do per-category refusal directions in small instruct models share a dominant
low-dimensional component? Does it causally mediate refusal? And can that
geometry be used to detect a tampered model?

Three nights of work on a 16GB Apple Silicon Mac (MPS, float16, PyTorch +
HuggingFace only). Full write-ups: **[FINDINGS.md](FINDINGS.md)** (night 1),
**[FINDINGS_N2.md](FINDINGS_N2.md)** (night 2),
**[FINDINGS_N3.md](FINDINGS_N3.md)** (night 3).

## The three results

**Night 1 — the core is real and causal.** Across 45 SORRY-Bench harm
categories, one direction carries **72–76%** of the variance in the
per-category refusal directions, at 68–94% network depth, consistently in all
three models (effective rank 3.4–4.5 — one dominant core plus a few secondary
directions, not one direction and not 45). Ablating it collapses refusal
0.82 → 0.15/0.17 while a random-direction control does not move and ARC-easy is
untouched. Llama-3.2-3B is confounded (ARC 0.94 → 0.49) and is excluded from
the clean claim.

**Night 2 — the loud attack is caught; the quiet attack was never built.**
Global abliteration collapses refusal (0.742 → 0.058) and is flagged. All eight
concept-localized organisms pass the scanner — but none suppressed its target
category, so the evasion is vacuous: they pass because they are inert. The
cause is geometric. With categories 70–74% mutually correlated, RepIt-style
isolation leaves only 25–60% of the target direction, and the refusal the model
actually performs lives in the shared core the method discards. Separately,
**AMS Tier 1 missed global abliteration in both models**; every flag came from
Tier 2, which needs a trusted clean copy.

**Night 3 — a baseline-free detector.** The same 45-category matrix yields two
dimensionless features (`top1_var`, `mean_cos_pc1`) that need no reference
model. They score **12/12** on the built variants, are stable across a
12× range of thresholds, and fire when only **11–17%** of the attack's eventual
refusal suppression has happened. This fills the quadrant Tier 1 was meant to
occupy.

| | needs a trusted clean copy? | caught full abliteration | caught partial |
|---|---|---|---|
| AMS Tier 1 | no | **no (0/2)** | — |
| AMS Tier 2 | **yes** | yes (2/2) | would need the copy |
| Night-3 detector | **no** | yes (2/2) | yes (6/6) |

## Three things that nearly went wrong

- **A layer-0 degeneracy** (night 1) made a naive peak-picker choose layer 0/1
  in all three models, with *higher* apparent variance than the true peak. At
  the last token — the same chat-template token for every prompt — the
  difference-of-means cancels to exactly zero for 43 of 45 categories, and an
  epsilon in the normalizer manufactured a spectrum from noise. Unnoticed, the
  causal stage would have ablated noise and reported a false negative.
- **A refusal classifier with a 7.5% false-positive rate** (night 2) put a
  floor under every number. Hardened against 200 hand-labelled generations to
  2.3% (F1 0.800 → 0.909), which revealed night 1's rates were inflated
  (Qwen baseline 0.742, not 0.822).
- **A CPU surrogate for abliteration that was not faithful** (night 3), off by
  up to 0.66 on `mean_cos_pc1`. Discarded in favour of real weight edits — but
  the failure itself showed that downstream layers partially *reconstruct* the
  shared core after weight-level abliteration.

## Repository layout

```
src/
  common.py              config, data loading, memory discipline, logging
  stage0_dtype.py        float16-vs-float32 numerics gate (run first)
  smoke_test.py          20-prompt end-to-end + runtime extrapolation
  stage1_extract.py      activations -> results/activations/
  stage23_structure.py   direction matrix, SVD, layer profile, figures
  stage4_causal.py       ablation hooks, refusal + ARC measurement
  make_report.py         night-1 gate verdict and summary figure
  validate_classifier.py night-1 classifier false-positive probe

  n2_classifier.py       hardened 3-way refusal classifier
  n2_gen_valset.py       classifier validation set (in-scope prompts only)
  n2_eval_classifier.py  precision/recall/FP before and after
  n2_organisms.py        target selection, RepIt isolation, weight editing
  n2_ams.py              AMS-style scanner (tier 1 + tier 2)
  n2_run.py              build + evaluate + scan + save night-3 data
  n2_stage0_band.py      Llama-3.2-3B layer-band sweep (timeboxed)
  n2_report.py           night-2 headline table and figure

  n3_detector.py         baseline-free detector, clean prior, k-sweep (CPU)
  n3_graded.py           real partial-abliteration sweep (GPU)
  n3_report.py           detector-vs-AMS comparison and figures

results/
  activations/           cached last-token states, all layers (305MB)
  structure.json         night-1 geometry, per-layer, per-model
  metrics.json           night-1 causal results
  summary.json           night-1 gate verdicts
  classifier_val/        200 generations (benign + in-scope), labels, metrics
  n2_metrics.json        per-variant refusal, per-category, ARC, AMS
  n2_summary.json        night-2 headline table
  n2_band_sweep.json     3B layer-band sweep
  organisms/             edit vectors + per-variant direction matrices
  n3_detector.json       detector features, verdicts, threshold sensitivity
  n3_graded.json         refusal + features vs abliteration strength
  n3_surrogate.json      surrogate fidelity check (negative result)
  figures/               13 figures
  run.log                timestamped log of every run
```

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install \
    torch transformers datasets huggingface_hub numpy scipy scikit-learn matplotlib

cd src
../.venv/bin/python stage0_dtype.py        # numerics gate — must pass first
../.venv/bin/python smoke_test.py          # end-to-end on 20 prompts
../.venv/bin/python stage1_extract.py
../.venv/bin/python stage23_structure.py
../.venv/bin/python stage4_causal.py
../.venv/bin/python make_report.py

../.venv/bin/python n2_gen_valset.py       # night 2
../.venv/bin/python n2_eval_classifier.py
../.venv/bin/python n2_stage0_band.py
../.venv/bin/python n2_run.py
../.venv/bin/python n2_report.py

../.venv/bin/python n3_detector.py         # night 3
../.venv/bin/python n3_graded.py
../.venv/bin/python n3_report.py
```

Every stage is resumable and skips work already cached on disk. Total runtime
is roughly 25 min (night 1), 55 min (night 2), 30 min (night 3).

### Data note

`sorry-bench/sorry-bench-202406` is gated and this account is not authorized
(metadata reads, file fetch 403s). The code uses `SillyTilly/SorryBench`, a
byte-faithful reup: 450 rows, identical schema, 45 categories × 10, all base
style. Category *names* ship in no public mirror, so the taxonomy is taken from
the SORRY-Bench paper and was verified against prompt content for 18 spot-checked
categories.

## Safety

Model-organism work here is deliberately scoped. Concept-localized targets come
only from six categories where refusal is a policy judgment and non-refusal
produces harmless text — legal, medical and financial advice; religious,
ethical and political belief promotion — and `n2_organisms.build_localized()`
raises on anything else. Global abliteration is a whole-model control with no
category target, evaluated on refusal *rates* only.

No harmful completion content is written to disk or to any log, in any night.
Generations are classified to a binary label inside the generation loop and the
text is discarded. The only persisted completions are in
`results/classifier_val/`, drawn exclusively from benign alpaca prompts and the
six in-scope categories. Organisms are stored as edit *vectors* plus a
deterministic loader, never as edited checkpoints.

## Open questions

1. **A spectrum-preserving adversary.** Every night-3 positive is PC1
   abliteration at some strength. An attacker optimising against `top1_var`
   directly is untested and is the next thing to build.
2. **Is night 2's null a property of RepIt or of the safe category set?** The
   six permitted categories have baseline refusal 0.2–0.5 versus 0.742 across
   all 45 — they are the ones these models refuse least, so a working organism
   had little headroom. Answering this safely needs a benign high-refusal
   category from outside SORRY-Bench, or a logit-level measurement that does not
   require generating from a suppressed hazardous refusal.
3. **Llama-3.2-3B remains inconclusive.** No layer band bought refusal
   suppression without capability damage; night 1's 3B number is best read as
   general degradation.
