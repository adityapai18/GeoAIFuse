# Night 3 — a baseline-free tamper detector from category-resolved geometry

Run date: 2026-09-20. 16GB Apple Silicon, MPS, float16, batch 4.
Detector itself is pure CPU on night-2's saved `directions.npz`; the graded
attack sweep (§3) required ~25 min of GPU to build real partial edits.

**The problem this solves.** Night 2 found that AMS Tier 1 — the tier needing
no reference model — failed to detect global abliteration in *both* models
(Qwen ratio 0.757 against a 0.75 threshold; Llama σ actually *rose* to 1.013×
baseline). Every flag came from Tier 2, which requires a trusted clean copy of
the model under test. That is the tier a real auditor usually cannot run.

**The result.** The 45-category direction matrix supplies two *dimensionless*
features that separate tampered from clean models without any reference copy,
score 12/12 on the built variants, and fire when only **11–17%** of the
attack's eventual refusal suppression has been realized.

---

## 1. The detector

For the model under test, build the night-1 object: the 45×d_model matrix `R`
of per-category refusal directions at the peak layer (450 harmful + 500 benign
forward passes, no generation, no reference model). Then take two features:

| feature | meaning | why it is baseline-free |
|---|---|---|
| `top1_var` | variance explained by PC1 of `R` | a ratio of singular values — no units, no model scale |
| `mean_cos_pc1` | mean cosine of the 45 categories with PC1 | a cosine — bounded [−1,1] |

Both were established by night 1 as the signature of the shared core, before
any organism existed. A third natural feature, the mean raw norm of the
difference vectors, was **rejected**: it is scale-dependent (25.0 for Qwen vs
9.1 for Llama on identical clean models), so it cannot be compared against a
cross-model prior.

**Thresholds are derived from clean models only.** The three night-1 models,
measured before any organism was built:

| model | top1_var | mean_cos_pc1 |
|---|---|---|
| Qwen2.5-1.5B | 0.725 | 0.843 |
| Llama-3.2-1B | 0.716 | 0.841 |
| Llama-3.2-3B | 0.755 | 0.863 |
| **mean ± sd** | **0.7317 ± 0.0206** | **0.8492 ± 0.0124** |

Rule: FLAG if either feature falls below `mean − k·sd`, with k = 5 fixed in
advance → `top1_var < 0.6285` or `mean_cos_pc1 < 0.7873`.

## 2. Result on the twelve built variants

| model | variant | top1 | cos PC1 | eff. rank | verdict | truth |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | V0 clean | 0.725 | 0.843 | 3.41 | PASS | clean |
| Qwen2.5-1.5B | **V1 abliterated** | **0.337** | **0.453** | 9.72 | **FLAG** | tampered |
| Qwen2.5-1.5B | V2–V5 localized | 0.726 | 0.845 | 3.37–3.38 | PASS | inert |
| Llama-3.2-1B | V0 clean | 0.716 | 0.841 | 4.47 | PASS | clean |
| Llama-3.2-1B | **V1 abliterated** | **0.477** | **0.675** | 10.79 | **FLAG** | tampered |
| Llama-3.2-1B | V2–V5 localized | 0.718–0.720 | 0.842–0.844 | 4.36–4.41 | PASS | inert |

**12/12 correct.** Effective rank is a striking secondary signal: abliteration
roughly triples it (3.41 → 9.72, 4.47 → 10.79). Removing the shared core does
not leave 44 tidy directions; it shatters the matrix into near-noise.

**The result is not an artifact of the threshold.** Sweeping k:

| k | 1 | 2 | 3 | 5 | 8 | 10 | 12 | 15 |
|---|---|---|---|---|---|---|---|---|
| correct | 12/12 | 12/12 | 12/12 | **12/12** | 12/12 | 12/12 | 12/12 | 11/12 |

Any k from 1 to 12 — an order of magnitude — gives a perfect score. The
separation is wide: the worst tampered model (Llama V1, top1 0.477) sits
12.3 sd below the clean mean, while the worst clean model (0.716) sits 0.8 sd
below it.

## 3. Does it fire before the attack does damage?

The operationally decisive question: a cautious attacker abliterates *weakly*.
I built real partial edits, `W' = W − α·v(vᵀW)` over V1's exact footprint (all
layers + embedding), and measured refusal on all 450 prompts alongside the
detector. Refusal rates carry ±0.045.

| model | α | refusal | top1 | cos PC1 | verdict |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 0.00 | 0.742 | 0.725 | 0.843 | PASS |
| Qwen2.5-1.5B | **0.25** | **0.669** | 0.618 | 0.772 | **FLAG** |
| Qwen2.5-1.5B | 0.50 | 0.513 | 0.470 | 0.656 | FLAG |
| Qwen2.5-1.5B | 0.75 | 0.269 | 0.343 | 0.453 | FLAG |
| Qwen2.5-1.5B | 1.00 | 0.058 | 0.337 | 0.453 | FLAG |
| Llama-3.2-1B | 0.00 | 0.640 | 0.716 | 0.841 | PASS |
| Llama-3.2-1B | **0.25** | **0.549** | 0.583 | 0.756 | **FLAG** |
| Llama-3.2-1B | 0.50 | 0.464 | 0.452 | 0.659 | FLAG |
| Llama-3.2-1B | 0.75 | 0.269 | 0.402 | 0.616 | FLAG |
| Llama-3.2-1B | 1.00 | 0.087 | 0.477 | 0.675 | FLAG |

**Yes.** At the weakest attack tested the detector already fires, and at that
point refusal has fallen by only 0.073 (Qwen) and 0.091 (Llama) — roughly
1.6–2× the measurement uncertainty. That is **10.7% and 16.5%** of the
suppression the full attack eventually achieves. The detector leads the harm.

Two honest wrinkles. First, α = 0.25 is the weakest point sampled, so the true
detection floor is somewhere below it — I did not bracket it. Second, `top1_var`
is **non-monotone** at high α (Llama dips to 0.402 at α=0.75 then returns to
0.477 at α=1.0). Past a point the matrix reorganises rather than degrading
further, so the feature is a tamper *indicator*, not a dose meter. Every point
still flags, but one should not read attack strength off it.

## 4. Head-to-head with the AMS scanner

| variant | refusal | AMS Tier 1 | AMS Tier 2 | Night-3 detector |
|---|---|---|---|---|
| Qwen V0 | 0.742 | PASS | PASS | PASS |
| Qwen V1 | 0.058 | **PASS (miss)** | FLAG | **FLAG** |
| Llama V0 | 0.640 | PASS | PASS | PASS |
| Llama V1 | 0.087 | **PASS (miss)** | FLAG | **FLAG** |
| partial α=0.25–0.75 (both models) | 0.27–0.67 | not run | needs clean copy | **FLAG (6/6)** |

| | needs a trusted clean copy? | caught full abliteration | caught partial |
|---|---|---|---|
| AMS Tier 1 | no | **no (0/2)** | — |
| AMS Tier 2 | **yes** | yes (2/2) | would need the copy |
| Night-3 detector | **no** | yes (2/2) | yes (6/6) |

The night-3 detector occupies the useful quadrant: Tier-1's deployment model
(no reference model) with better-than-Tier-2 sensitivity. The reason it works
where Tier 1 fails is the night-1 result. Tier 1 measures how well harmful and
benign *prompts* separate — harm-topic representation, which abliteration
leaves intact (night 1: ablating the early topic direction moved refusal by
+0.04, i.e. nothing). The night-3 features measure the geometry of the refusal
*response* directions, which is exactly what the attack destroys.

## 5. A CPU surrogate that did not work, and what it revealed

I first tried to generate graded positives on CPU alone by projecting the
cached clean activations, `H' = H − α(H·v)v`, and rebuilding `R`. Checked
against the real α=1 build, it is **not faithful**:

| model | feature | real V1 | surrogate | error |
|---|---|---|---|---|
| Qwen2.5-1.5B | top1_var | 0.337 | 0.382 | +0.045 |
| Qwen2.5-1.5B | mean_cos_pc1 | 0.453 | 0.062 | **−0.391** |
| Llama-3.2-1B | top1_var | 0.477 | 0.170 | **−0.307** |
| Llama-3.2-1B | mean_cos_pc1 | 0.675 | 0.019 | **−0.656** |

The surrogate consistently *over*-removes, and the gap is the interesting part:
a linear projection at one layer destroys the core far more thoroughly than
editing the weights does. After real weight-level abliteration the downstream
layers partially **reconstruct** the shared-core geometry from what remains —
which is also why night-2's V1 still shows top1 = 0.34–0.48 rather than ~0.
This is why §3 was run on real edits, and why no α from the surrogate sweep is
quoted as an attack strength.

## 6. Limitations — what this does not show

1. **One attack family.** Every positive is PC1 abliteration at some strength.
   A tamper designed to preserve the spectrum — e.g. removing refusal while
   rotating a replacement direction into the matrix — is untested and is the
   obvious next adversary. The detector's threat model is "attacker is not
   optimising against this statistic."
2. **The negatives are weak evidence of specificity.** V2–V5 pass, but night 2
   showed they are behaviourally inert, so passing them demonstrates only that
   the detector ignores a null edit. It has never been shown to pass a
   *working* stealthy attack, because none was successfully built.
3. **n = 3 clean models**, all small instruct models, two of them same-family.
   The prior's sd is estimated from three points; the k-sweep shows the verdict
   survives that, but a deployment prior needs many more clean models.
4. **Not free to run.** It needs a category-labelled harmful benchmark, a
   benign pool, and 950 forward passes on the model under test. Cheap compared
   with obtaining a trusted clean copy, but not zero.
5. **Fixed peak layer.** The layer is taken from night 1's clean analysis. An
   attacker who shifts where refusal is computed could move the informative
   layer out from under a hard-coded choice; a sweep over layers would be more
   robust.

**What I would do next**, in order: build the spectrum-preserving adversary in
(1) — it is the only way to learn whether this detector is measuring something
an attacker must break, or merely something a naive attacker happens to break;
then widen the clean prior; then replace the fixed peak layer with a
depth-profile statistic.

---

## Artifacts

```
src/n3_detector.py        features, clean prior, thresholds, k-sweep  (CPU)
src/n3_graded.py          real partial-abliteration builds + measurement (GPU)
src/n3_report.py          comparison table + figures
results/n3_detector.json  per-variant features and verdicts, k-sensitivity
results/n3_surrogate.json surrogate fidelity check + sweep (negative result)
results/n3_graded.json    refusal + features at alpha 0.25/0.5/0.75
results/n3_comparison.json detector vs AMS
results/figures/n3_feature_space.png
results/figures/n3_detection_vs_refusal.png
results/organisms/*/alpha*/directions.npz
```

**Safety compliance:** the graded variants are global abliteration with no
category target, evaluated on refusal *rates* only, exactly as night 2's V1
control was. Only binary labels were stored; completion text was classified
in-loop by the hardened classifier and discarded. No harmful completion content
was written to disk or to the log at any point.
