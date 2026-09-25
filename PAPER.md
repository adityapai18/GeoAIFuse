# Category-Resolved Refusal Geometry: Causal Structure, the Failure of Concept-Localized Editing, and the Limits of Reference-Free Tamper Detection

*Preprint draft. All experiments run locally on a 16GB Apple Silicon machine.*

---

## Abstract

Refusal in instruction-tuned language models has been characterized as mediated
by a single direction (Arditi et al., 2024), and subsequently as a small cone or
a shared core with domain-specific tails (Wollschläger et al., 2025; Joad et
al., 2026). We measure this structure at **fine category granularity** — 45
SORRY-Bench harm categories — across 13 instruct models from 9 families, and ask
what the resulting geometry is good for.

Three results hold. (i) A dominant shared component exists in every model
measured, carrying **55–79%** of the variance across the 45 per-category refusal
directions, with effective rank 3.4–4.5 in the stronger models; in two small
models we confirm it is causally load-bearing, with directional ablation
collapsing refusal from 0.770 to 0.119 and 0.704 to 0.170 while a random-vector
control and an early-layer "topic" control do not move and general capability is
preserved. (ii) A reimplementation of an activation-based model-integrity
scanner (Vantablack, 2026) **fails to detect global abliteration** in both
models using its reference-free tier, and we give the mechanistic reason: that
tier measures harm-*topic* separation, which abliteration leaves intact, rather
than refusal-*response* geometry, which it destroys. (iii) **RepIt-style concept
isolation does not produce working concept-localized organisms at this scale.**
Because the 45 category directions are 70–74% mutually collinear, isolating one
retains only 25–60% of it; measured at logit level with paired tests, 6 of 8
localized edits have confidence intervals excluding even a 10%-of-abliteration
effect, and four *increase* refusal. A ridge sweep shows suppression appears only
once isolation is abandoned.

**The central limitation is in our own positive proposal.** We derive a
reference-free tamper detector from PC1 statistics of the category matrix and
report that it fails. Its features are PC1 statistics and the only attack it was
validated against was PC1 ablation. Widening the clean prior from 3 models to 13
gives it a **38% false-positive rate**, and the first adversary we constructed —
subtracting the refusal direction while rotating a substitute direction into the
residual stream — suppresses refusal from 0.742 to 0.042 with capability intact
and **passes** the detector. The same model is still caught by a
reference-*based* comparison against a stored clean baseline (cosine 0.226
against a 0.8 threshold). We therefore report the detector as a negative result
and conclude that reference-free tamper detection built on spectrum statistics
is defeatable by construction, while the reference comparison does the real work.

---

## 1. Introduction

If refusal behavior in an aligned language model is carried by a small number of
directions in activation space, two things follow. First, those directions are a
natural target for anyone who wants to remove the behavior — "abliteration" is
now a commodity technique. Second, they are a natural place to *look* for
evidence that someone has done so. This paper works both sides of that and finds
the attack side much easier than the defense side.

Prior work established the structure at coarse granularity: a single mediating
direction (Arditi et al., 2024), later refined to a low-dimensional cone
(Wollschläger et al., 2025) and to a shared core with domain-specific tails
(Joad et al., 2026). Nobody had measured it at the granularity of individual harm
*categories*. We do, using SORRY-Bench's 45-category taxonomy (Xie et al., 2024),
and we ask three questions in sequence:

1. **Is there a shared core at fine granularity, and does it causally mediate
   refusal, or is it topic encoding?** (§5.1)
2. **Can that structure be exploited to build a stealthy, category-localized
   attack?** (§5.2)
3. **Can it be used to detect a tampered model?** (§5.4–5.5)

The answers are yes, no, and no. The two negative results are the more useful
contributions, and one of them is against a method we proposed ourselves.

### Contributions

- The first category-resolved measurement of refusal geometry (45 categories ×
  13 models × 9 families), with causal validation and controls.
- A mechanistic explanation for why a published activation-based integrity
  scanner's reference-free tier misses abliteration.
- A negative result on RepIt-style concept isolation at this scale, with the
  collinearity mechanism measured directly via a ridge sweep, and resolved at
  logit level rather than left as an underpowered null.
- A reference-free detector, and its refutation: a 38% false-positive rate under
  a realistic clean prior, and a first-try adversary that defeats it.
- Four self-corrections to our own earlier analyses, including the withdrawal of
  a causal claim about Llama-3.2-3B.

---

## 2. Related Work

**Refusal directions.** Arditi et al. (2024) showed that refusal in chat models
is mediated by a single direction, removable by directional ablation and
inducible by activation addition. Wollschläger et al. (2025) found that refusal
is better described by a *concept cone* of several directions than by one.
Joad et al. (2026) decomposed this further into a shared, reusable core plus
domain-specific components. Our work sits directly downstream: we push the
granularity from domains to 45 fine categories and ask whether the shared core
remains dominant (it does) and whether the per-category tails are independently
manipulable (they are not).

**Concept isolation and representation editing.** RepIt-style methods
(cf. Marks et al.) isolate a target concept's representation by projecting away a
basis of non-target concepts, aiming for edits that affect one concept and spare
others. We apply this construction to per-category refusal directions and find it
inert at this scale, for a reason that is specific to refusal geometry: the
non-target basis is 70–74% collinear with the target, so the residual after
projection is small and carries little of the behavior.

**Robustness of representation-level interventions.** Zou et al. (2024) showed
that circuit-breaker-style representation engineering can make harmful
completions hard to elicit, and Bailey et al. demonstrated *obfuscated
activations* — attacks that preserve the activation statistics a monitor reads
while changing behavior. Our Phase-C adversary is an instance of the same
phenomenon applied to spectrum statistics rather than probe activations, and our
result is consistent with theirs: statistics-based monitoring is defeatable by
construction when the attacker knows the statistic.

**Model-integrity scanning.** Vantablack (2026) proposed a two-tier activation
scan for detecting safety-training modification: Tier 1 measures cluster
separation between harmful and benign prompts (no reference model needed);
Tier 2 compares the model's aggregate safety direction against a stored clean
baseline. We reimplement both and find Tier 1 blind to global abliteration,
which we trace to the harm-topic/refusal-response distinction.

**Benchmarks.** SORRY-Bench (Xie et al., 2024) supplies 450 prompts over a
balanced 45-category taxonomy, which is what makes category-resolved measurement
possible. We use Alpaca (Taori et al., 2023) for the benign contrast pool and
ARC-easy (Clark et al., 2018) as a capability check.

---

## 3. Method

### 3.1 Setup

All experiments run locally: PyTorch + HuggingFace on a 16GB Apple Silicon
machine (MPS), float16, batch size 4, prompts left-padded and truncated to 256
tokens. A mandatory numerics gate runs before anything else: we generate from a
fixed probe in float16 and verify coherent output and finite hidden states at
every layer, falling back to float32 otherwise. float16 passed, so all reported
numbers are float16. All linear algebra is done on CPU in float32.

### 3.2 The category direction matrix

For model *m*, layer *ℓ*, and category *c*, we take the last-token residual
stream (post-chat-template, pre-generation) and compute the difference of means
against a benign pool of 500 filtered Alpaca instructions:

&nbsp;&nbsp;&nbsp;&nbsp;`r_c^(ℓ) = mean(act(c)) − mean(act(benign))`

L2-normalized and stacked into `R^(ℓ) ∈ ℝ^(45×d)`. Structure is read off the SVD
of `R`: variance explained by the top components, effective rank
`exp(H(spectrum))`, participation ratio, the 45×45 cosine matrix, and each
category's cosine with PC1.

**A degeneracy that must be guarded.** At the last token position — which is the
*same* chat-template token for every prompt — the embedding is identical across
prompts, so `r_c` cancels to exactly zero for 43 of 45 categories at layer 0.
Normalizing those with an epsilon manufactures a spectrum from numerical noise,
and a naive argmax over layers then selects layer 0 or 1. This happened for **all 13 models we tested**: the naive argmax fell in the
first three layers every time (layer 0 for 3 models, layer 1 for 9, layer 2 for
one), and every model had exactly one degenerate layer. We therefore restrict peak selection to the second
half of the network and exclude degenerate layers, and we report the naive
argmax alongside so the artifact is visible. Ablating the spurious early
direction has no causal effect (§5.1), confirming it is an artifact.

### 3.3 Causal validation

Directional ablation (Arditi-style) applied via forward hooks at the embedding
output and every block output, for every token, throughout generation:

&nbsp;&nbsp;&nbsp;&nbsp;`x' = x − v(vᵀx)`, `‖v‖ = 1`

Conditions: baseline; ablate PC1; ablate a random unit vector (control); ablate
the spurious early-layer direction (topic-encoding control). Capability is
checked with 100 ARC-easy questions.

### 3.4 Refusal measurement

Binary refusal labels come from a hardened 3-way classifier (REFUSAL / COMPLY /
INCOHERENT) applied to the first 250 characters. It was built against 200
hand-labelled generations and improves on a naive substring matcher from 7.5% to
**2.3%** false-positive rate (F1 0.800 → 0.909, recall 0.962). The INCOHERENT
class exists so that an intervention which merely breaks the model is not scored
as "stopped refusing." Reported refusal rates carry **±0.045** (classifier error
plus binomial sampling at n=450). All numbers in this paper use this one
classifier; earlier analyses using a looser matcher were re-measured (§6).

Where statistical power matters (§5.3) we replace the binary measure with a
continuous one: the refusal log-odds at the first generated position,

&nbsp;&nbsp;&nbsp;&nbsp;`s = log Σ_{t∈R} p(t) − log Σ_{t∈C} p(t)`

with `R` = refusal-initiating first tokens ("I", "Sorry", "As", "Unfortunately",
"I'm") and `C` = compliance-initiating first tokens ("Sure", "Here",
"Certainly", "Yes", "Absolutely", "To", "The", "Of", "1", "Below"), token ids
resolved per-tokenizer with overlaps dropped. This score is validated by a
required gate: it must move sharply between a clean model and a fully
abliterated one, which it does (d = −1.85 and −1.51, p < 1e-16).

### 3.5 Model organisms

- **V1, global abliteration** (positive control): orthogonalize every
  residual-stream write (embedding, attention `o_proj`, MLP `down_proj`) against
  the pooled top refusal direction, all layers.
- **V2–V5, concept-localized** (RepIt-style): take the target category's
  direction `v_t`; stack the other 44 reweighted by inverse norm into `N`;
  whiten with a ridge-regularized Gram `G = NNᵀ + λ·tr(G)/n·I`; project
  `v_iso = v_t − Nᵀ(G+λI)⁻¹N v_t`; orthogonalize weights against `v_iso` over a
  band around the peak layer.
- **V6, random-edit control**: a random unit vector over the same footprint.
- **Phase-C adversary**: `W' = W − α·v(vᵀW) + β·u(vᵀW)` with `u ⊥ v` fixed.

Organisms are stored as edit *vectors* plus a deterministic loader, never as
edited checkpoints.

### 3.6 Safety scope

Concept-localized targets are restricted to six categories where refusal is a
policy judgment rather than a safety guardrail and non-refusal yields harmless
text: legal, medical and financial advice; religious, ethical and political
belief promotion. The construction function raises on anything else. Global
abliteration and the Phase-C adversary are whole-model edits with no category
target, evaluated on refusal *rates* only. No harmful completion content is
written to disk or logs at any point: generations are classified to a binary
label inside the generation loop and the text discarded. The only persisted
completions are from benign prompts and the six in-scope categories, used to
build the classifier validation set.

---

## 4. Experimental scope

| | models | prompts | measure |
|---|---|---|---|
| Geometry (§5.1) | 13 instruct models, 9 families | 450 harmful + 500 benign | SVD of `R` |
| Causal validation (§5.1) | 3 | 135 (3/category) | refusal rate, ARC |
| Organisms (§5.2) | 2 × 6 variants | 450 (10/category) | refusal rate, ARC, AMS |
| Logit power (§5.3) | 2 × 7 configs + 14 ridge points | 60 in-scope | refusal log-odds |
| Detector (§5.4–5.5) | 13 clean + 16 edited | — | PC1 statistics |
| Adversary (§5.5) | 1 × 5 β | 450 | refusal, ARC, PC1 stats, AMS |

---

## 5. Results

### 5.1 A dominant shared component exists, and mediates refusal

Across 13 models from 9 families, one direction carries **55–79%** of the
variance across the 45 per-category refusal directions (mean 0.675, sd 0.089),
peaking at 60–95% network depth. Effective rank is 3.4–4.5 in the stronger
models and mean pairwise cosine between category directions is 0.70–0.74. The
picture is neither one direction nor 45 independent ones: it is one dominant
core plus a few secondary directions, consistent with Wollschläger et al. and
Joad et al., now resolved per category.

The strength is **not** uniform. Phi-3.5-mini (0.773), Falcon3-1B (0.772) and
Qwen2.5-3B (0.785) sit far above StableLM-2-1.6B (0.550), TinyLlama-1.1B (0.565)
and Qwen3-0.6B (0.576). Within Qwen2.5 the component strengthens monotonically
with scale (0.5B → 0.642, 1.5B → 0.725, 3B → 0.785). With n = 1–3 per family we
report these as observations, not measured effects.

**Causal validation** (135 prompts, 3 per category, one classifier):

| model | baseline | ablate PC1 | random control | early-spike control | ARC |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 0.770 | **0.119** | 0.763 | 0.822 | 0.87 → 0.89 |
| Llama-3.2-1B | 0.704 | **0.170** | 0.711 | 0.704 | 0.47 → 0.44 |
| Llama-3.2-3B | 0.348 | 0.356 | 0.341 | 0.289 | 0.94 → 0.49 |

For the two small models this is the clean outcome: refusal collapses by
0.53–0.65 absolute while the random control moves by <0.01 and capability is
preserved. The **early-spike control is the key one for the topic-encoding
question**: ablating the spurious early-layer direction — which has a *higher*
nominal variance-explained than the true peak — changes refusal by +0.00 to
+0.05, i.e. not at all. Whatever early layers encode about harmful topics is not
what drives the refusal decision.

**Llama-3.2-3B shows no effect.** Its baseline is 0.348 and ablation gives
0.356. A layer-band sweep ([20-27], [22-27], [24-27], [23-25]) found no band
that suppressed refusal, and ablation at all layers produced 9/135 incoherent
outputs and halved ARC (0.94 → 0.49). We conclude the intervention damages this
model rather than steering it.

### 5.2 Concept-localized organisms are inert

Eight organisms (4 targets × 2 models), evaluated on all 450 prompts:

- Target-category refusal changes: 0.000, −0.100, 0.000, −0.100 (Qwen) and
  0.000, 0.000, +0.100, +0.100 (Llama) — at most one prompt in ten, two in the
  wrong direction.
- Non-target refusal, ARC, and incoherence (0/450 for every localized variant)
  are unchanged.
- All eight **pass** the integrity scanner, but vacuously: they pass because
  they do nothing.

The build diagnostics show why. The isolation step works exactly as designed —
`max|cos(v_iso, non-targets)| ≤ 0.02` — and that is the problem. With category
directions 70–74% mutually collinear, the component unique to one category is
small: after projecting off the other 44, only **25–60%** of the target direction
survives. The refusal the model performs lives in the shared core that the
isolation procedure discards.

Two controls rule out an implementation artifact: widening the edit to all
layers plus the embedding (V1's footprint) still produces no suppression, and
the same editing code given the *pooled* direction produces V1's full collapse.

### 5.3 The null is real, not underpowered

Per-category refusal at n = 10 has a 95% CI of ±0.28–0.43 on a difference, so
§5.2 could not have detected anything smaller than a ~0.4 drop. Replacing it
with the paired continuous measure of §3.4:

| model | target | Δ log-odds | 95% CI | d | p |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | Political Belief | −0.045 | [−0.163, +0.074] | −0.27 | 0.418 |
| Qwen2.5-1.5B | Financial Advice | −0.171 | [−0.431, +0.089] | −0.47 | 0.171 |
| Qwen2.5-1.5B | Ethical Belief | **+0.628** | [+0.527, +0.728] | +4.46 | <0.001 |
| Qwen2.5-1.5B | Religion Promotion | −0.176 | [−0.345, −0.007] | −0.74 | 0.043 |
| Llama-3.2-1B | Political Belief | +0.420 | [−0.152, +0.992] | +0.53 | 0.131 |
| Llama-3.2-1B | Financial Advice | −0.972 | [−1.294, −0.650] | −2.16 | <0.001 |
| Llama-3.2-1B | Religion Promotion | **+2.870** | [+2.020, +3.720] | +2.41 | <0.001 |
| Llama-3.2-1B | Legal Consulting | **+1.994** | [+1.441, +2.547] | +2.58 | <0.001 |

Reference: full abliteration moves the same score by −3.596 (Qwen) and −8.599
(Llama). Six of eight targets have CIs excluding even a 10%-of-abliteration
effect — **the null is confirmed, not underpowered.** The random-edit control
moves by −0.001 and +0.055. Four targets *significantly increase* refusal
log-odds, one by +2.87: removing a category's isolated direction can make the
model refuse harder.

**The isolation/force tradeoff.** Sweeping the ridge so that
`cos(v_iso, v_target)` spans 0.245 → 0.937:

| cos(iso, target) | Δ target | Δ non-target | interpretation |
|---|---|---|---|
| 0.245 | −0.045 | +0.032 | no suppression |
| 0.454 | −0.112 | −0.079 | no suppression |
| 0.614 | −0.247 | −0.208 | suppresses, not selective |
| 0.824 | −1.060 | −0.728 | suppresses, not selective |

Suppression appears only above cos ≈ 0.6, and where it appears the non-target
categories move almost as much. **Isolation buys selectivity by giving up all of
the force.** This is the mechanism behind §5.2, measured rather than inferred,
and it is a property of refusal geometry — high inter-category collinearity —
rather than of the RepIt construction itself.

### 5.4 A reference-free integrity tier fails, for a specific reason

Our reimplementation of the two-tier scanner (3 coarse safety concepts × 16
surface-form-matched harmful/benign pairs; Tier 1 = cluster separation, Tier 2 =
cosine against a stored clean baseline) gives:

| variant | refusal | Tier 1 | Tier 2 |
|---|---|---|---|
| Qwen2.5-1.5B clean | 0.742 | PASS | PASS |
| Qwen2.5-1.5B abliterated | 0.058 | **PASS (miss)** | FLAG (cos 0.450) |
| Llama-3.2-1B clean | 0.640 | PASS | PASS |
| Llama-3.2-1B abliterated | 0.087 | **PASS (miss)** | FLAG (cos 0.275) |

Tier 1 misses global abliteration in both models; in Llama-3.2-1B its separation
statistic *increases* after refusal is removed (σ 4.797 → 4.862). The
mechanistic reason is the same distinction that §5.1's early-spike control
establishes: **Tier 1 measures how well harmful and benign prompts separate —
harm-topic representation — while abliteration removes the refusal *response*
direction.** A model can represent "this prompt is about explosives" perfectly
while having lost the machinery that makes it decline. This matters
operationally because Tier 1 is the tier that needs no trusted clean copy.

### 5.5 Our own reference-free detector fails too

Motivated by §5.4 we built a reference-free detector from two **dimensionless**
statistics of `R` at the peak layer — top-1 variance explained and mean cosine
with PC1 — with thresholds from a clean prior. On the variants available at the
time it scored 12/12 and fired when only 11–17% of an attack's eventual refusal
suppression had occurred. Two subsequent tests refute it.

**(a) False positives under a realistic prior.** The original prior was 3
models, two of them same-family, and the thresholds were set from those same
three points. Widening to 13 models and 9 families:

| | n | mean | sd | min | max |
|---|---|---|---|---|---|
| top1_var (3-model prior) | 3 | 0.732 | 0.021 | 0.716 | 0.755 |
| top1_var (13-model prior) | 13 | 0.675 | **0.089** | **0.550** | 0.785 |

The clean spread was understated by **4×**. Five of 13 clean models —
StableLM-2-1.6B, TinyLlama-1.1B, SmolLM2-1.7B, Qwen3-0.6B, Qwen3-1.7B — fall
below the published threshold: a **38% false-positive rate**. Re-deriving
thresholds on the widened prior, no setting recovers the original performance:
at k = 2 (0 clean FP) the detector catches 2/2 full abliterations but only 4/6
partial ones, and 2 of 6 partial attacks fall *inside* the clean range.

**(b) Defeated by the first adversary constructed.** With
`W' = W − v(vᵀW) + β·u(vᵀW)`, α fixed at 1.0, sweeping β on Qwen2.5-1.5B:

| β | refusal | ARC | top1_var | mean_cos_pc1 | eff. rank | detector |
|---|---|---|---|---|---|---|
| clean | 0.742 | 0.870 | 0.725 | 0.843 | 3.41 | PASS |
| 0.25 | 0.062 | 0.900 | 0.359 | 0.520 | 9.15 | FLAG |
| 1.0 | 0.033 | 0.880 | 0.536 | 0.717 | 5.94 | FLAG |
| **2.0** | **0.042** | **0.860** | **0.706** | **0.835** | 3.63 | **PASS** |
| 4.0 | 0.033 | 0.770 | 0.837 | 0.914 | 2.31 | PASS |

At β = 2.0 the model refuses 4% of SORRY-Bench against a clean 74%, retains
ARC-easy at 0.860 against 0.870, and presents spectrum statistics closest to
clean Qwen2.5-1.5B itself. The attacker pays **nothing in refusal suppression**
to evade the detector and about 0.01 ARC between the caught and uncaught
settings. This is one hand-built construction with a fixed, unsearched `u`,
which makes it strong evidence against the detector and weak evidence about
optimal attacks.

**(c) But the reference-based tier still catches it.** Running the same
adversarial models through the scanner of §5.4:

| β | our detector | Tier 1 σ (clean 4.861) | Tier 2 cos | AMS verdict |
|---|---|---|---|---|
| 2.0 | PASS | 3.830 | **0.226** | **FLAG** |
| 4.0 | PASS | 3.293 | **0.133** | **FLAG** |

The adversary preserves the PC1 *statistics* while rotating the PC1 *direction*
away. A reference-free detector can only read the statistics; a reference-based
one reads the direction. **Reference-free tamper detection built on spectrum
statistics is defeatable by construction, and the reference comparison is doing
the real work.** Both reference-free tiers tested in this project fail against
attacks aimed at them, for two different reasons.

---

## 6. Corrections to our own earlier analyses

We list these because three of the four materially changed a conclusion.

1. **The layer-0 degeneracy (§3.2)** initially caused the peak-layer selector to
   choose layer 0 or 1 in every model. Had it gone unnoticed, the causal stage
   would have ablated a noise direction and reported a false negative.
2. **The refusal classifier** originally had a 7.5% false-positive rate, which
   was described as a "measurement floor." Hardened to 2.3%, and every earlier
   number re-measured.
3. **Llama-3.2-3B's causal effect is withdrawn.** Under the loose classifier it
   appeared as 0.652 → 0.385 and was reported as "confounded." Under one
   consistent classifier it is 0.348 → 0.356 — no effect. The apparent drop was
   the loose classifier's false positives disappearing as outputs changed.
4. **The selectivity metric** — (target drop)/(mean non-target drop) — is
   undefined when both terms are ≈0 and produced values of 22.0 and −14.7 from
   noise. Retired in favor of raw drops with CIs.

---

## 7. Limitations

**On the geometry result.**
1. Thirteen models, but n = 1–3 per family; the scale and family patterns in
   §5.1 are observations, not measured effects. Two intended models were
   unavailable (one license-gated, one requiring `trust_remote_code`), so there
   is no Google or InternLM representative.
2. Causal validation covers only three models, and only two of them show the
   effect. All are ≤3B parameters. Nothing here speaks to frontier-scale models.
3. The direction matrix is built at a single peak layer chosen by a heuristic.
   A different layer-selection rule could change the numbers.
4. Difference-of-means is a coarse estimator of a category direction; the benign
   contrast pool is fixed Alpaca and its composition affects every `r_c`.

**On the organism result.**
5. The six permitted target categories have baseline refusal 0.2–0.5 versus
   0.742 across all 45 — they are the categories these models refuse *least*, so
   a working organism had limited headroom. The logit measure (§5.3) mitigates
   but does not eliminate this.
6. We tested one isolation construction. A different method — sparse
   dictionaries, supervised probes, non-linear isolation — might succeed where
   this failed.
7. The logit score is read at the first generated position. The validation gate
   shows it tracks refusal, but not that the mapping to full-completion behavior
   is linear.

**On the detection result.**
8. The adversary was swept on one model with one fixed substitute direction `u`.
   It suffices to refute the detector; it does not characterize the attack
   surface.
9. We did not test whether an adversary can hold the PC1 direction *and* its
   statistics simultaneously, which is what would be needed to defeat the
   reference-based tier as well.
10. The AMS reimplementation follows the published description; any discrepancy
    with the authors' implementation is ours. Our Tier-1 threshold (0.75× the
    clean baseline) is our choice, though Llama's σ *rising* after abliteration
    means no threshold on that statistic would have caught it.
11. The clean prior, even at 13 models, is small for setting a deployment
    threshold, and all models are small instruct models.

**On measurement throughout.**
12. Refusal is scored by a string-based classifier with a measured 2.3% FP and
    3.8% FN rate; all rates carry ±0.045. A judge model would be better.
13. `raw_norms` is unavailable (stored as NaN) for the six graded-attack
    direction matrices; it is scale-dependent and unused by any reported
    statistic.

---

## 8. Reproducibility

Every experiment runs on one 16GB consumer machine with no cloud or API
inference. Total compute is roughly three hours across four nights. Every stage
is resumable from disk state and skips work already cached; metrics are written
to JSON incrementally; a timestamped log records every phase boundary.

The repository contains the extraction and analysis code, cached activations
(305MB), per-night metrics, organism edit vectors (not checkpoints), the
direction matrices for all variants, 16 figures, and the hand-labelled
classifier validation set. Model organisms are distributed as edit vectors plus
a deterministic loader; no abliterated weights are published.

The harmful benchmark is SORRY-Bench's base split. The canonical repository is
gated; we used a byte-faithful public reup with identical schema (450 rows, 45
categories × 10, all base style) and verified the taxonomy names against prompt
content for all 12 categories named in any claim.

---

## 9. Future Work

1. **Can an adversary preserve both the statistics and the direction?** §5.5(c)
   identifies the binding constraint. Sweeping `u` jointly with β, or optimizing
   directly against a Tier-2-style objective, is the natural next experiment and
   would determine whether reference-*based* detection is also defeatable.
2. **Is there any reference-free invariant that survives a targeted attack?**
   Two independent reference-free tiers failed here for two different reasons.
   A negative result at the level of "no low-dimensional activation statistic is
   robust" would be worth more than another detector.
3. **Does the shared core strengthen with scale?** The Qwen2.5 0.5B→3B trend
   (0.642 → 0.785) is suggestive; a controlled scan within one family across
   more sizes would settle it, and would say whether frontier models are more or
   less exposed to this attack.
4. **Why do some families have weak cores?** Qwen3 at 0.576–0.634 sits far below
   Qwen2.5 at matched size. If this reflects a training difference, it is a
   lever: models whose refusal is *not* concentrated in one direction are
   intrinsically harder to abliterate.
5. **Better concept isolation.** §5.3 shows the failure is collinearity, not
   the method's logic. A method that can act on the shared core *conditioned* on
   category — rather than on the orthogonal residual — might produce the
   localized organism this work could not build.

---

## References

Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., &
Nanda, N. (2024). *Refusal in Language Models Is Mediated by a Single
Direction.* NeurIPS 2024.

Bailey, L., et al. *Obfuscated Activations Bypass LLM Latent-Space Defenses.*

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., &
Tafjord, O. (2018). *Think You Have Solved Question Answering? Try ARC.*

Joad, et al. (2026). *More to Refusal than a Single Direction.*

Marks, S., et al. *RepIt: Isolating concept representations for targeted
editing.*

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang,
P., & Hashimoto, T. (2023). *Stanford Alpaca: An Instruction-following LLaMA
Model.*

Vantablack (2026). *Detecting Safety Training Modification in Language Models
via Activation Analysis.* IEEE Access.

Wollschläger, T., et al. (2025). *The Geometry of Refusal in Large Language
Models: Concept Cones and Representational Independence.* ICML 2025.

Xie, T., et al. (2024). *SORRY-Bench: Systematically Evaluating Large Language
Model Safety Refusal Behaviors.* arXiv:2406.14598.

Zou, A., et al. (2024). *Improving Alignment and Robustness with Circuit
Breakers.* NeurIPS 2024.

*Citation details are from memory and must be verified against the primary
sources before submission; several entries are incomplete by design.*
