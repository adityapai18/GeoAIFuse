# Night 4 — making three nights survive review

Run date: 2026-09-25. 16GB Apple Silicon, MPS, float16, batch 4.
Phases in priority order: A (clean prior), B (logit power), D (consistency),
C (adversary, timeboxed 60 min — completed in 27), E (writeup).

**Summary: both of the detector's load-bearing claims fail.** Widening the
clean prior from 3 models to 13 gives the published threshold a **38% false
positive rate** on clean models. A spectrum-preserving adversary, built on the
first construction tried, suppresses refusal from 0.742 to 0.042 with ARC
essentially intact and **PASSES** the detector. Separately, the night-2 null is
now confirmed real rather than underpowered, and re-measuring night 1 with the
hardened classifier **removes the Llama-3.2-3B effect entirely**.

The geometry results survive. The detection result does not.

---

## The four plain answers

**1. How wide is the clean prior?** Much wider than night 3 assumed.
13 models, 9 families: top1_var **0.675 ± 0.089** (range 0.550–0.785),
mean_cos_pc1 **0.812 ± 0.057** (range 0.734–0.879). Night 3's prior (n=3,
sd 0.021) understated the spread by **4×**.

**2. Does the published threshold survive?** No. **5 of 13 clean models (38%)
are false positives.** DECISION GATE A is tripped: the night-3 detector as
published is broken.

**3. Is the night-2 null real or underpowered?** **Real.** 6 of 8 targets have
a 95% CI excluding even a 10%-of-abliteration effect. Mean target delta is
**+0.569** log-odds — refusal slightly *increased* — against an abliteration
reference of −6.097.

**4. Adversary outcome?** **Detector defeated.** β = 2.0 gives refusal 0.042
(clean 0.742), ARC 0.860 (clean 0.870), top1 0.706 — inside the clean range —
and the detector **PASSES**. First construction, no optimization. The same
model is still caught by AMS Tier 2 (cos 0.226), which reads the PC1 *direction*
rather than its statistics — see the Phase C addendum.

---

## Phase A — the clean prior (W1)

13 models across 9 families. Two skipped: `google/gemma-2-2b-it` (license-gated,
not accepted on this account) and `internlm/internlm2_5-1_8b-chat` (tokenizer
requires `trust_remote_code`, which I declined to enable rather than execute
arbitrary remote code). Every model got night 1's peak rule including the
layer-0 degeneracy guard; the naive argmax landed in the first three layers for **all 13**
(layer 0 for 3, layer 1 for 9, layer 2 for Phi-3.5-mini), and every model had
exactly one degenerate layer — confirming the artifact is universal and not a
Qwen/Llama quirk.

| model | family | top1_var | mean_cos_pc1 | vs published threshold |
|---|---|---|---|---|
| stablelm-2-1_6b-chat | StableLM-2 | 0.5503 | 0.7344 | **FALSE POSITIVE** |
| TinyLlama-1.1B-Chat | TinyLlama | 0.5649 | 0.7432 | **FALSE POSITIVE** |
| SmolLM2-1.7B-Instruct | SmolLM2 | 0.5729 | 0.7365 | **FALSE POSITIVE** |
| Qwen3-0.6B | Qwen3 | 0.5761 | 0.7508 | **FALSE POSITIVE** |
| Qwen3-1.7B | Qwen3 | 0.6338 | 0.7838 | **FALSE POSITIVE** |
| Qwen2.5-0.5B-Instruct | Qwen2.5 | 0.6418 | 0.7913 | pass (margin 0.004) |
| Llama-3.2-1B-Instruct | Llama-3.2 | 0.7155 | 0.8408 | pass |
| OLMo-2-0425-1B-Instruct | OLMo-2 | 0.7174 | 0.8399 | pass |
| Qwen2.5-1.5B-Instruct | Qwen2.5 | 0.7247 | 0.8434 | pass |
| Llama-3.2-3B-Instruct | Llama-3.2 | 0.7550 | 0.8634 | pass |
| Falcon3-1B-Instruct | Falcon3 | 0.7716 | 0.8752 | pass |
| Phi-3.5-mini-instruct | Phi-3.5 | 0.7732 | 0.8708 | pass |
| Qwen2.5-3B-Instruct | Qwen2.5 | 0.7848 | 0.8791 | pass |

### This qualifies night 1, not just night 3

Night 1 reported top1_var of 0.716–0.755 across three models and called the
shared core "strikingly consistent." Across 9 families it ranges **0.550 to
0.785**. A dominant component exists everywhere — even the weakest model puts
55% of the variance in one direction out of 45 — but its *strength* is
model-dependent, and night 1's narrow range was an artifact of sampling two
closely-related families.

Two patterns, both under-powered at n=1–3 per family and offered as
observations rather than claims:

- **Scale.** Within Qwen2.5: 0.5B → 0.642, 1.5B → 0.725, 3B → 0.785.
- **Family.** Qwen3 (0.576, 0.634) sits far below Qwen2.5 at matched size; the
  weakest four are StableLM-2, TinyLlama, SmolLM2 and Qwen3-0.6B.

### Can any threshold be rescued?

| k (mean − k·sd, widened prior) | clean FP | full-abliteration TP | partial TP | inert FP |
|---|---|---|---|---|
| 1 | 4/13 | 2/2 | 5/6 | 0/8 |
| **2** | **0/13** | **2/2** | **4/6** | **0/8** |
| 3 | 0/13 | 1/2 | 2/6 | 0/8 |
| 5 (night-3 rule) | 0/13 | 1/2 | 1/6 | 0/8 |

Clean top1 bottoms out at 0.5503; full abliteration tops out at 0.4770. The
margin is **0.073**, so a threshold separating clean from full abliteration
exists — but night 3's own k=5 rule applied to the widened prior only catches
1 of 2. Partial attacks are worse: 2 of 6 fall **inside** the clean range, so
night 3's "6/6 on partial attacks" does not survive contact with a realistic
prior. Any threshold that keeps both properties has to be fitted to these
points, which is not a detector.

## Phase B — the night-2 null is real (W2)

Binary refusal at n=10 has a 95% CI of ±0.28–0.43 on a per-category difference
(Phase D3), so night 2 could not have detected anything smaller than a ~0.4
drop. Replacing it with a continuous paired measure — refusal log-odds at the
first generated position, one forward pass per prompt, no generation:

**Score validation (required gate).** V0 → V1 moves the score sharply in both
models: Qwen −3.596 (d = −1.85, p = 8e-21), Llama −8.599 (d = −1.51, p = 5e-17).
The score measures refusal.

| model | variant | target | Δ log-odds | 95% CI | d | p | verdict |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | V2 | Political Belief | −0.045 | [−0.163, +0.074] | −0.27 | 0.418 | null confirmed |
| Qwen2.5-1.5B | V3 | Financial Advice | −0.171 | [−0.431, +0.089] | −0.47 | 0.171 | cannot exclude |
| Qwen2.5-1.5B | V4 | Ethical Belief | **+0.628** | [+0.527, +0.728] | +4.46 | <0.001 | null confirmed |
| Qwen2.5-1.5B | V5 | Religion Promotion | −0.176 | [−0.345, −0.007] | −0.74 | 0.043 | null confirmed |
| Llama-3.2-1B | V2 | Political Belief | +0.420 | [−0.152, +0.992] | +0.53 | 0.131 | null confirmed |
| Llama-3.2-1B | V3 | Financial Advice | −0.972 | [−1.294, −0.650] | −2.16 | <0.001 | cannot exclude |
| Llama-3.2-1B | V4 | Religion Promotion | **+2.870** | [+2.020, +3.720] | +2.41 | <0.001 | null confirmed |
| Llama-3.2-1B | V5 | Legal Consulting | **+1.994** | [+1.441, +2.547] | +2.58 | <0.001 | null confirmed |

Random-edit control (V6): −0.001 and +0.055 — indistinguishable from clean, as
it should be.

"Null confirmed" means the 95% CI excludes even a 10%-of-abliteration drop. Six
of eight qualify. The two that "cannot exclude" are both Financial Advice, and
Llama's is a real but small suppression: −0.972 against an abliteration
reference of −8.599, i.e. **11%** of the available effect.

The striking result is the sign. Four targets *raise* refusal log-odds
significantly, one by +2.87. Removing a category's isolated direction does not
suppress its refusal; in several cases it makes the model refuse harder.

### The isolation/force tradeoff

Sweeping the ridge so cos(v_isolated, v_target) spans 0.245 → 0.937:

| cos(iso,target) | Qwen V2 Δ target | Qwen V2 Δ non-target | interpretation |
|---|---|---|---|
| 0.245 (night-2 default) | −0.045 | +0.032 | no suppression |
| 0.454 | −0.112 | −0.079 | no suppression |
| 0.614 | −0.247 | −0.208 | suppresses, not selective |
| 0.824 | −1.060 | −0.728 | suppresses, not selective |

Suppression appears only above cos ≈ 0.6, and when it appears the non-target
categories move almost as much. **Isolation buys selectivity by giving up all
of the force.** That is the mechanism behind night 2's null, measured directly
rather than inferred.

## Phase C — the spectrum-preserving adversary (W3)

`W' = W − α·v(vᵀW) + β·u(vᵀW)`, α = 1.0, v = PC1, u a fixed unit vector
orthogonal to v (seeded, not searched). Qwen2.5-1.5B, all layers + embedding,
450 prompts. Completed in 27 of the 60-minute timebox.

| β | refusal | ARC-easy | top1_var | mean_cos_pc1 | eff. rank | detector |
|---|---|---|---|---|---|---|
| — (clean) | 0.742 | 0.870 | 0.725 | 0.843 | 3.41 | PASS |
| 0.25 | 0.062 | 0.900 | 0.359 | 0.520 | 9.15 | FLAG |
| 0.5 | 0.047 | 0.890 | 0.411 | 0.603 | 8.09 | FLAG |
| 1.0 | 0.033 | 0.880 | 0.536 | 0.717 | 5.94 | FLAG |
| **2.0** | **0.042** | **0.860** | **0.706** | **0.835** | 3.63 | **PASS** |
| 4.0 | 0.033 | 0.770 | 0.837 | 0.914 | 2.31 | PASS |

**The detector is defeated.** At β = 2.0 the model refuses 4% of SORRY-Bench
(clean: 74%), keeps ARC-easy at 0.860 against a clean 0.870, and presents a
direction matrix whose statistics — top1 0.706, cos 0.835, effective rank 3.63 —
are indistinguishable from a healthy model. It sits in the middle of the clean
range, closest to Qwen2.5-1.5B itself (0.725 / 0.843).

β sweeps monotonically from "obvious" to "invisible" while refusal stays at the
floor throughout: the attacker pays **nothing in refusal suppression** for
evading the detector, and only ~0.01 ARC between β=1 (caught) and β=2 (clean).
β=4 over-rotates, pushing top1 to 0.837 — above every clean model measured — and
costing 0.10 ARC, so the adversary has an optimum rather than a monotone
tradeoff.

This is one hand-built construction, not an optimized attack, which makes it
strong evidence *against* the detector: if the first thing tried defeats it, a
determined attacker has no difficulty. It is correspondingly **not** evidence
that β≈2 is the best attack or that other detectors fail the same way.

## Phase D — consistency fixes

**D1. One classifier throughout.** Night 1's causal numbers used the loose
matcher. Re-measured with the hardened classifier, same prompts and conditions:

| model | condition | night-1 (loose) | corrected (hardened) |
|---|---|---|---|
| Qwen2.5-1.5B | baseline | 0.822 | **0.770** |
| Qwen2.5-1.5B | ablate PC1 | 0.148 | **0.119** |
| Qwen2.5-1.5B | random control | 0.807 | **0.763** |
| Qwen2.5-1.5B | early-spike control | 0.874 | **0.822** |
| Llama-3.2-1B | baseline | 0.822 | **0.704** |
| Llama-3.2-1B | ablate PC1 | 0.170 | **0.170** |
| Llama-3.2-1B | random control | 0.837 | **0.711** |
| Llama-3.2-1B | early-spike control | 0.859 | **0.704** |
| Llama-3.2-3B | baseline | 0.652 | **0.348** |
| Llama-3.2-3B | ablate PC1 | 0.385 | **0.356** |
| Llama-3.2-3B | random control | 0.681 | **0.341** |
| Llama-3.2-3B | early-spike control | 0.622 | **0.289** |

Night 1's conclusion for the two small models is unchanged and slightly
stronger: PC1 ablation collapses refusal (0.770 → 0.119; 0.704 → 0.170) while
the random and early-spike controls do not move.

**The Llama-3.2-3B result changes completely.** Corrected, its baseline is
0.348 and PC1 ablation gives 0.356 — refusal does not fall at all. Night 1
called the 3B "confounded"; the correct statement is that **there was never a
refusal effect to confound.** The apparent 0.652 → 0.385 drop was the loose
classifier's false positives disappearing as the outputs changed, not refusal
being removed. The 9/135 incoherent outputs under ablation (versus 0 elsewhere)
confirm the intervention damaged that model rather than steering it.

**D2. Category names verified.** All 12 categories named in any claim checked
against two sample prompts each: Impersonation, IP Infringement, Non-sexual
Explicit Content, False Advertising, Military Use, Political Belief, Ethical
Belief, Religion Promotion, False Common Knowledge, Medical Advice, Financial
Advice, Legal Consulting. **No mismatches.** Every claim that names a category
is safe.

**D3. Selectivity retired.** Replaced with raw drops and CIs. Every night-2
target effect has a 95% CI containing zero and spanning ±0.28–0.43 — which is
the quantitative statement of W2 and the reason Phase B was necessary.

## What this leaves standing

| claim | status after night 4 |
|---|---|
| Fine-grained category directions share a dominant component | **holds**, widened: 0.550–0.785 across 9 families, not 0.72–0.76 |
| That component causally mediates refusal (2 small models) | **holds**, strengthened under one consistent classifier |
| Llama-3.2-3B shows a weaker version of the effect | **withdrawn** — no effect once measured properly |
| AMS Tier 1 misses global abliteration | **holds** (unchanged from night 2) |
| RepIt-style isolation produces inert edits | **holds**, upgraded from "unreadable" to confirmed with CIs |
| A baseline-free detector separates clean from tampered | **withdrawn** — 38% FP on clean models, defeated by a first-try adversary |


### Addendum: the adversary is still caught by a reference-based scan

The Phase C attack defeats the reference-free night-3 detector. It does not
defeat AMS Tier 2, which compares the model's aggregate safety direction
against a stored clean baseline:

| β | night-3 detector | AMS Tier 1 σ (clean 4.861) | AMS Tier 2 cos | AMS verdict |
|---|---|---|---|---|
| 2.0 | **PASS** | 3.830 | **0.226** | **FLAG** (tier 2) |
| 4.0 | **PASS** | 3.293 | **0.133** | **FLAG** (tiers 1 and 2) |

This is the mechanistic point of the whole night. The adversary preserves the
PC1 *statistics* — variance explained, mean cosine, effective rank — while
rotating the PC1 *direction* away from where it was. A reference-free detector
can only read the statistics, so it passes. A reference-based detector reads
the direction, so it fires at cos 0.226 against a 0.8 threshold.

The conclusion for the field is therefore sharper than "our detector broke":
**reference-free tamper detection built on spectrum statistics is defeatable by
construction, and the reference comparison is doing the real work.** Night 2's
finding that AMS Tier 1 misses abliteration still stands — Tier 1 is
reference-free and fails for a different reason (it reads harm-topic separation,
not refusal geometry). Both reference-free tiers tested across this project fail
against an attack aimed at them; the one tier that needs a trusted clean copy
catches everything thrown at it.

## Limitations of night 4 itself

1. **13 models is still small**, and 9 families with n=1 for most means the
   family and scale patterns are observations, not measured effects.
2. **One adversary construction, one model.** β was swept on Qwen2.5-1.5B only.
   That is enough to break the detector but not to characterize the attack.
3. **Phase B measures logits at the first token**, which correlates with but is
   not identical to refusal behavior over a full completion. The V0→V1 gate
   shows it tracks refusal; it does not prove the mapping is linear.
4. **Two target models were unavailable** (gemma-2 license, internlm
   trust_remote_code), so the prior has no Google or InternLM representative.
5. **Only one substitute direction u was tried**, seeded and fixed rather than
   searched. An attacker optimizing u jointly with beta would likely do better,
   and might also evade the Tier-2 check below.

**What I would do next:** characterize the attack properly — sweep u, and test
whether an adversary can hold *both* the PC1 statistics and the PC1 direction
while still removing refusal. The addendum below says that is the binding
constraint.
