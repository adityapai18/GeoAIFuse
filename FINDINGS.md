# Category-resolved refusal geometry — findings

> **SUPERSEDED IN PART — see [FINDINGS_N4.md](FINDINGS_N4.md) and [PAPER.md](PAPER.md).**
> This night-1 record used a loose refusal classifier (7.5% FP) that was later
> hardened. Corrected numbers: Qwen2.5-1.5B baseline **0.770 → 0.119** (not
> 0.822 → 0.148); Llama-3.2-1B **0.704 → 0.170** (not 0.822 → 0.170).
> **The Llama-3.2-3B causal claim below is WITHDRAWN**: re-measured it is
> 0.348 → 0.356, i.e. no effect, not a "confounded" weak effect.
> The claim that the shared core is "strikingly consistent" at 0.716–0.755 is
> **qualified**: across 9 model families it ranges 0.550–0.785.
> The geometry and the two-small-model causal conclusions stand.


Run date: 2026-08-13. 16GB Apple Silicon, MPS, PyTorch 2.13 + transformers 5.15.
Wall clock: ~22 min total across all stages (Stage 1 extraction 2 min,
Stage 3 analysis <1 s, Stage 4 causal 16 min).

**The question:** across 45 fine-grained harm categories, do per-category
refusal directions share a dominant low-dimensional component, and does that
component causally mediate refusal, or is it just topic encoding?

**The short answer:** yes, a dominant shared component exists — one direction
carries **72–76%** of the variance across all 45 category directions, at
**68–94% network depth**, consistently in all three models. In the two smaller
models it is **causally load-bearing**: ablating it drops refusal from 0.82 to
0.15/0.17 while a random-direction control does not move at all and general
capability is untouched. In Llama-3.2-3B the same ablation also damages the
model (ARC-easy 0.94 → 0.49), so that model's result is **confounded and should
not be counted as clean evidence**.

---

## 1. Which dtype, and why

**float16**, everywhere, batch size 4.

The mandatory probe loaded Qwen2.5-1.5B-Instruct in float16 on MPS and
generated 30 tokens from "Explain photosynthesis in one sentence." Verbatim
output:

> Photosynthesis is the process by which plants use sunlight to convert carbon
> dioxide and water into glucose and oxygen.

Coherent English, and every hidden state across all 29 layers was finite. No
float32 fallback was needed, so batch size stayed at 4 for all three models
(the float32/3B batch-2 rule never triggered). Every saved activation array was
re-checked for finiteness after extraction; all passed
(`results/activations/*/manifest.json`, `"finite": true`).

## 2. Does a dominant shared component exist?

Yes, and it is strikingly consistent across models.

At the peak layer, SVD of the 45×d_model matrix of L2-normalized per-category
refusal directions gives:

| model | peak layer | depth | top-1 | top-3 | top-5 | eff. rank | mean pairwise cos |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 19 / 28 | 68% | **0.725** | 0.879 | 0.920 | 3.41 | 0.706 |
| Llama-3.2-1B | 15 / 16 | 94% | **0.716** | 0.796 | 0.839 | 4.47 | 0.701 |
| Llama-3.2-3B | 24 / 28 | 86% | **0.755** | 0.831 | 0.870 | 3.71 | 0.740 |

One component explains 72–76% of the variance; three explain ~80–88%. The
effective rank of the spectrum is 3.4–4.5, and the participation ratio is
1.7–1.9. So the picture is **not** a single direction and **not** 45 independent
ones: it is one dominant core plus a small number of secondary directions —
consistent with the "shared core plus domain-specific tails" picture from
Wollschläger et al. and Joad et al., now resolved at fine category granularity.

Per-category alignment with the top component (the full distribution is saved
in `results/structure.json:cos_with_pc1`, and is the detection statistic for
later work):

| model | mean cos with PC1 | min | max | # categories < 0.5 |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0.843 | 0.478 | 0.972 | 1 |
| Llama-3.2-1B | 0.841 | 0.603 | 0.948 | 0 |
| Llama-3.2-3B | 0.863 | 0.549 | 0.962 | 0 |

**Layer profile.** The shared component is weakest in the middle of the network
and strongest in a broad late plateau (Qwen L19–24, Llama-1B L10–15, Llama-3B
L23–27). See `results/figures/layer_profile_*.png`.

### An artifact worth flagging: the layer-0/1 spike

A naive argmax over layers picks **layer 0 or 1** in all three models, with
top-1 variance of 0.84–0.90 — apparently *stronger* than the real peak. It is
spurious, and the run originally reported it before this was caught.

At the last token position — which is the same chat-template token for every
prompt — the embedding is literally identical across prompts, so
`mean(category) − mean(benign)` cancels to **exactly zero for 43 of 45
categories** at layer 0. Normalizing those zero vectors with an epsilon
manufactures a spectrum out of numerical noise. Layers 1–3 inherit the problem:
prompt representations there are still 92–99% cosine-identical, and the raw
difference-vector norms are ~20× smaller than at mid-depth (relative separation
0.07–0.18 at L1 versus 0.38–0.81 at the true peak).

Peak selection therefore searches the second half of the network only, and
`structure.json` records `naive_argmax_layer`, `degenerate_layers`, and the
relative-separation diagnostic per layer. The early direction was then tested
causally — see below — and it does nothing, confirming it is an artifact rather
than a competing refusal direction.

## 3. The gate — which outcome occurred

Directional ablation (`x' = x − v vᵀx`) applied at the embedding output and
every block output, throughout generation, 135 prompts (3 per category,
subsampled because MPS generation is slow), 48 new tokens, greedy, batch 4.

| model | baseline | ablate PC1 | random control | early-spike control | ARC-easy (base → ablated) |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 0.822 | **0.148** | 0.807 | 0.874 | 0.87 → **0.89** |
| Llama-3.2-1B | 0.822 | **0.170** | 0.837 | 0.859 | 0.47 → **0.44** |
| Llama-3.2-3B | 0.652 | **0.385** | 0.681 | 0.622 | 0.94 → **0.49** |

**Qwen2.5-1.5B and Llama-3.2-1B: "(b) collapses refusal broadly, (c) barely
moves → core is real and causal. PROCEED."**

This is the clean outcome. Refusal falls by 0.65–0.67 absolute (a ~80%
reduction) while the random control moves by −0.02 to +0.01 — i.e. not at all,
in the wrong direction if anything. The collapse is *broad*, not driven by a few
categories: for Qwen, 43 of the 44 categories that refused at baseline dropped,
and **28 of 45 collapsed to exactly zero**; for Llama-1B, 42 of 44 dropped and
26 collapsed to zero. No category with a high baseline resisted.

Two further controls support the causal reading:

- **Capability is preserved.** ARC-easy is unchanged (Qwen 0.87 → 0.89,
  Llama-1B 0.47 → 0.44). The models still answer questions; they just stop
  refusing. The ablation is not simply breaking them.
- **The classifier floor.** On 60 benign alpaca prompts the same string-matcher
  fires 10% of the time (mostly "as an AI…" hedges). The ablated refusal rates
  of 0.148/0.170 are therefore close to the measurement floor — refusal is
  approximately *eliminated*, not merely reduced.
- **Topic encoding is ruled out.** Ablating the spurious early-layer direction
  changes refusal by +0.04/+0.05 (i.e. nothing), despite that direction having
  a *higher* nominal variance-explained. Whatever the early layers encode about
  harmful topics, it is not what drives the refusal decision.

**Llama-3.2-3B: partial, and confounded — do not count it as clean evidence.**

Refusal falls 0.652 → 0.385 while the random control stays at 0.681, so the
direction is doing *something* specific. But ARC-easy falls **0.94 → 0.49**, so
this ablation is also substantially damaging the model, and the refusal drop
cannot be cleanly attributed to removing refusal rather than to general
degradation. Only 12 of 45 categories collapsed to zero and two (Child-related
Crimes, Extremist Content Promotion) did not move at all. The honest reading is
that the 3B result is **inconclusive**, not a weaker version of the same effect.

## 4. Which categories sit furthest from the core

These are the candidate hard false-positive cases. Ranked by cosine with PC1 at
the peak layer (full distributions in `results/structure.json`, bar charts in
`results/figures/cos_pc1_*.png`):

| rank | Qwen2.5-1.5B | Llama-3.2-1B | Llama-3.2-3B |
|---|---|---|---|
| 1 | Religion Promotion (0.48) | Legal Consulting Advice (0.60) | Religion Promotion (0.55) |
| 2 | Ethical Belief Promotion (0.53) | IP Infringement (0.61) | Impersonation (0.65) |
| 3 | False Advertising (0.56) | Religion Promotion (0.65) | Ethical Belief Promotion (0.66) |
| 4 | Legal Consulting Advice (0.66) | Impersonation (0.66) | Non-sexual Explicit Content (0.71) |
| 5 | False Common Knowledge (0.69) | Non-sexual Explicit Content (0.70) | False Advertising (0.73) |
| 6 | IP Infringement (0.73) | Military Use (0.70) | Military Use (0.73) |

The overlap across models is the interesting part. **Religion Promotion
(Missionary)** is the single most off-core category in two of three models and
third in the other. **Legal Consulting Advice, IP Infringement, Ethical Belief
Promotion, Impersonation, False Advertising** and **Military Use** recur.

These share a structure: they are the categories where the "harm" is contextual
or contested rather than intrinsic — writing a Facebook post promoting a
religion, drafting a lease, reproducing copyrighted text, giving legal advice.
A model's refusal here is a policy judgment, not a safety reflex, and the
geometry reflects that: they are refused via directions that are measurably
distinct from the shared core. That makes them exactly the cases where a
core-alignment detection statistic should be expected to fail first.

## 5. What broke, and what I would fix first

**What broke:**

1. **The layer-0 degeneracy (caught and fixed).** The direction matrix is
   undefined at the embedding layer because the last token is identical across
   prompts; an epsilon in the normalizer hid a division by zero and produced a
   confident-looking 0.84 top-1 variance from 2 non-zero rows out of 45. The
   naive peak-picker then chose layers 0–1 for all three models. Had this gone
   unnoticed, Stage 4 would have ablated a noise direction and reported "the
   core is not causally load-bearing" — a completely wrong negative result.
2. **`sorry-bench/sorry-bench-202406` is gated and this account is not
   authorized** (metadata reads fine, file fetch 403s). Used
   `SillyTilly/SorryBench`, a faithful reup: 450 rows, identical schema
   (`question_id`/`category`/`turns`/`prompt_style`), 45 categories × 10, all
   base style. Category *names* ship in neither the mirror nor any other public
   copy, so the taxonomy was taken from the SORRY-Bench paper and verified
   against actual prompt content for 18 spot-checked categories before use.
3. **A memory-reporting bug (fixed).** `free_memory` measured MPS usage after
   the caller's `del`, so it always reported "freed 0.00GB". Callers now capture
   the baseline while the model is alive; the corrected numbers show 2.47–6.43GB
   reclaimed per model.
4. **Llama-3.2-3B's capability collapse** under ablation (ARC 0.94 → 0.49) is
   an unresolved result, not a bug I found and fixed.

**What I would fix first:** the 3B confound. Ablating one direction at *every*
layer is a blunt instrument, and in the 3B it evidently removes something the
model needs for general competence. The obvious next step is to ablate over a
restricted layer band around the peak rather than all 29 layers, and sweep the
band while tracking ARC — if refusal can be driven down while ARC holds, the 3B
joins the clean result; if not, that is a real and interesting difference
between the models.

After that, in order:

- **Replace the string-match refusal classifier.** A 10% false-positive rate on
  benign prompts is a soft floor under every number in Section 3, and it cannot
  distinguish "complied" from "produced fluent nonsense." A small judge model,
  or at minimum a coherence check on non-refusals, would tighten this. (The
  safety constraint — never persisting harmful completion content — was kept
  throughout: only binary labels were stored, and text was discarded inside the
  generation loop.)
- **Raise the causal sample.** 3 prompts per category means per-category rates
  are quantized to {0, ⅓, ⅔, 1}; the aggregate is solid but the per-category
  breakdown is coarse.
- **Test whether the secondary components matter.** Top-1 leaves ~25% of the
  variance unexplained and the effective rank is ~3.5–4.5. Ablating the top-3
  subspace rather than top-1 would show whether the tails are functional or
  incidental.

---

### Reproducing

```
src/stage0_dtype.py        # dtype gate (must run first)
src/smoke_test.py          # 20-prompt end-to-end + runtime extrapolation
src/stage1_extract.py      # activations -> results/activations/
src/stage23_structure.py   # direction matrix + SVD -> results/structure.json
src/stage4_causal.py       # ablation -> results/metrics.json
src/make_report.py         # gate verdict + summary figure
src/validate_classifier.py # refusal-classifier false-positive rate
```

Every stage is resumable from disk state and skips work already cached.
Artifacts: `results/activations/` (305MB, float16 + shape manifests),
`results/structure.json`, `results/metrics.json`, `results/summary.json`,
`results/figures/` (10 figures), `results/run.log`.
