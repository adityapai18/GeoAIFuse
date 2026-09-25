# Category-resolved refusal geometry

Do per-category refusal directions in small instruct models share a dominant
low-dimensional component? Does it causally mediate refusal? And can that
geometry detect a tampered model?

**Answers: yes, yes (in small models), and no.**

Four nights of work on a 16GB Apple Silicon Mac (MPS, float16, PyTorch +
HuggingFace only, no cloud). Preprint draft: **[PAPER.md](PAPER.md)**.
Per-night detail: [FINDINGS.md](FINDINGS.md) · [FINDINGS_N2.md](FINDINGS_N2.md)
· [FINDINGS_N3.md](FINDINGS_N3.md) · [FINDINGS_N4.md](FINDINGS_N4.md).
Figure index: [FIGURES.md](FIGURES.md).

## What holds

**Fine-grained category geometry, causally validated.** Across 45 SORRY-Bench
harm categories and 13 instruct models from 9 families, one direction carries
**55–79%** of the variance in the per-category refusal directions (effective
rank 3.4–4.5 in the stronger models). In the two small models tested causally,
ablating it collapses refusal — Qwen2.5-1.5B **0.770 → 0.119**, Llama-3.2-1B
**0.704 → 0.170** — while a random-direction control and an early-layer
"topic" control do not move and ARC-easy is preserved.

**AMS Tier 1 does not detect abliteration, and there is a mechanistic reason.**
A reimplementation of the Vantablack-2026 scanner misses global abliteration in
both models (0/2). Tier 1 measures how well harmful and benign *prompts*
separate — harm-topic representation — which abliteration leaves intact. It
removes the refusal *response* direction, a different object.

**RepIt-style concept isolation does not work at this scale.** Eight localized
organisms produced no target suppression. Measured at logit level with paired
n=10 (far more powerful than the binary measure), 6 of 8 have CIs excluding even
a 10%-of-abliteration effect, and four *raise* refusal. Because the 45 category
directions are 70–74% mutually collinear, isolating one leaves 25–60% of it, and
a ridge sweep shows suppression only appears once isolation is abandoned:
**isolation buys selectivity by giving up all of the force.**

## What does not hold — read this before using the detector

Night 3 proposed a reference-free tamper detector from PC1 statistics of the
category matrix. **Night 4 broke it in both directions, and it should not be
used.**

- **Its features are PC1 statistics and the attack it was tested on was PC1
  ablation.** That is the whole of its validation.
- **38% false positives.** Widening the clean prior from 3 models to 13 (9
  families) flags 5 clean models: StableLM-2, TinyLlama, SmolLM2, Qwen3-0.6B,
  Qwen3-1.7B. The night-3 prior understated clean spread by 4×.
- **Defeated by the first adversary tried.** `W' = W − v(vᵀW) + β·u(vᵀW)` at
  β=2.0 drops refusal 0.742 → **0.042** with ARC 0.870 → 0.860, and presents
  top1 = 0.706 — mid-clean-range. The detector **passes** it.
- **It is still caught by a reference-based scan.** AMS Tier 2 flags the same
  model at cos 0.226 (threshold 0.8). The adversary preserves the PC1
  *statistics* while rotating the *direction* away.

The generalizable conclusion: **reference-free tamper detection built on
spectrum statistics is defeatable by construction; the reference comparison is
doing the real work.** Both reference-free tiers tested in this project fail
against attacks aimed at them.

## Corrections we made to our own earlier nights

This repo's value is that negative results got written up as negative results,
including against itself.

- **Llama-3.2-3B's causal effect was withdrawn.** Night 1 reported 0.652 →
  0.385 and called it "confounded." Re-measured with the hardened classifier
  it is 0.348 → 0.356 — no effect at all. The apparent drop was the loose
  classifier's false positives disappearing as outputs changed.
- **The shared core is not as uniform as night 1 claimed.** 0.716–0.755 across
  three models became **0.550–0.785** across nine families. A dominant component
  exists everywhere; its strength is model- and scale-dependent.
- **A layer-0 degeneracy** nearly inverted night 1. At the last token — the same
  chat-template token for every prompt — the difference-of-means cancels to
  exactly zero for 43 of 45 categories, and an epsilon manufactured a spectrum
  from noise. The naive peak-picker chose layer 0/1 for **all 13** models tested.
- **A refusal classifier with a 7.5% FP rate** was hardened to 2.3% (F1 0.800 →
  0.909) against 200 hand-labelled generations, and every night-1 number was
  re-measured so the paper quotes one classifier throughout.
- **The selectivity metric was retired** — undefined when both its terms are ~0.

## Repository layout

```
src/
  common.py                config, data, memory discipline, logging
  stage0_dtype.py          float16-vs-float32 numerics gate (run first)
  smoke_test.py            20-prompt end-to-end + runtime extrapolation
  stage1_extract.py        activations -> results/activations/
  stage23_structure.py     direction matrix, SVD, layer profile
  stage4_causal.py         ablation hooks, refusal + ARC measurement
  make_report.py           night-1 gate verdict and figure
  validate_classifier.py   night-1 classifier FP probe

  n2_classifier.py         hardened 3-way refusal classifier
  n2_gen_valset.py         classifier validation set (in-scope prompts only)
  n2_eval_classifier.py    precision/recall/FP before and after
  n2_organisms.py          target selection, RepIt isolation, weight editing
  n2_ams.py                AMS-style scanner (tier 1 + tier 2)
  n2_run.py                build + evaluate + scan + save night-3 data
  n2_stage0_band.py        Llama-3.2-3B layer-band sweep (timeboxed)
  n2_report.py             night-2 headline table

  n3_detector.py           reference-free detector + clean prior + k-sweep
  n3_graded.py             real partial-abliteration sweep
  n3_report.py             detector-vs-AMS comparison

  n4_phase_a.py            13-model clean prior (one model at a time, purged)
  n4_phase_a_analysis.py   FP rate, per-family breakdown, threshold re-derivation
  n4_phase_b.py            refusal log-odds, paired tests, ridge sweep
  n4_phase_b_analysis.py   null-confirmed-vs-underpowered determination
  n4_phase_c.py            spectrum-preserving adversary (timeboxed)
  n4_phase_d1.py           night-1 causal re-measured with one classifier
  n4_report.py             night-4 figures
```

`results/` holds cached activations (305MB), per-night metrics JSON, the
organism edit vectors and direction matrices, 16 figures, and `run.log`.

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install \
    torch transformers datasets huggingface_hub numpy scipy scikit-learn matplotlib

cd src
../.venv/bin/python stage0_dtype.py     # numerics gate — must pass first
../.venv/bin/python smoke_test.py
../.venv/bin/python stage1_extract.py && ../.venv/bin/python stage23_structure.py
../.venv/bin/python stage4_causal.py    && ../.venv/bin/python make_report.py
../.venv/bin/python n2_gen_valset.py    && ../.venv/bin/python n2_eval_classifier.py
../.venv/bin/python n2_stage0_band.py   && ../.venv/bin/python n2_run.py
../.venv/bin/python n2_report.py
../.venv/bin/python n3_detector.py      && ../.venv/bin/python n3_graded.py
../.venv/bin/python n3_report.py
../.venv/bin/python n4_phase_a.py       && ../.venv/bin/python n4_phase_a_analysis.py
../.venv/bin/python n4_phase_b.py       && ../.venv/bin/python n4_phase_b_analysis.py
../.venv/bin/python n4_phase_d1.py      && ../.venv/bin/python n4_phase_c.py
../.venv/bin/python n4_report.py
```

Every stage is resumable and skips work already on disk. Roughly 25 min
(night 1), 55 (night 2), 30 (night 3), 70 (night 4).

### Data note

`sorry-bench/sorry-bench-202406` is gated and this account is not authorized
(metadata reads, file fetch 403s). The code uses `SillyTilly/SorryBench`, a
byte-faithful reup: 450 rows, identical schema, 45 categories × 10, all base
style. Category names ship in no public mirror, so the taxonomy comes from the
SORRY-Bench paper; all 12 categories named in any claim were verified against
prompt content (night 4, D2). Two Phase-A targets were unavailable:
`google/gemma-2-2b-it` (license-gated) and `internlm/internlm2_5-1_8b-chat`
(tokenizer requires `trust_remote_code`, declined).

## Safety

Model-organism work is deliberately scoped. Concept-localized targets come only
from six categories where refusal is a policy judgment and non-refusal produces
harmless text — legal, medical and financial advice; religious, ethical and
political belief promotion — and `n2_organisms.build_localized()` raises on
anything else. Global abliteration and the Phase-C adversary are whole-model
controls with no category target, evaluated on refusal *rates* only.

Night 4's logit measurement was adopted for statistical power on the
already-permitted categories. It does **not** widen the scope: measurement
safety and edit-construction safety are different questions, and the guard is
about the latter.

No harmful completion content is written to disk or any log, in any night.
Generations are classified to a binary label inside the generation loop and the
text discarded. The only persisted completions are in `results/classifier_val/`,
drawn exclusively from benign alpaca prompts and the six in-scope categories.
Organisms are stored as edit *vectors* plus a deterministic loader, never as
edited checkpoints.

## Open questions

1. **Can an adversary hold the PC1 direction *and* its statistics?** The Phase-C
   attack beats the reference-free detector but is caught by Tier 2 at cos
   0.226. Sweeping the substitute direction `u` jointly with β is the test.
2. **Is there any reference-free invariant that survives a targeted attack?**
   Two independent reference-free tiers failed here for two different reasons.
3. **Why do Qwen3 and the older small models have such weak cores** (0.55–0.63)
   while Phi-3.5 and Falcon3 reach 0.77? n=1–3 per family — currently an
   observation, not a result.
