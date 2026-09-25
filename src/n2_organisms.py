"""NIGHT 2 STAGES 2-3 — target selection and organism construction.

Variants per primary model:
  V0  clean reference (no edit)
  V1  GLOBAL ABLITERATION, positive control: orthogonalize every residual-
      stream write against the pooled top refusal direction, all layers.
  V2-V5  CONCEPT-LOCALIZED: one per target category, RepIt-style isolation
      then orthogonalization over the peak layer band only.

SAFETY SCOPE: concept-localized targets may ONLY come from IN_SCOPE below --
six categories where refusal is a policy judgment and non-refusal yields
harmless text. build_localized() raises on anything else. The global
abliteration control has no category target and is evaluated on rates only.

STORAGE: we persist the edit VECTOR plus the band metadata, not the edited
weights. Full weights would be ~3GB x 12 variants; the vector is a few KB and
apply_edit() reconstructs the variant deterministically at load time.
"""
import numpy as np
import torch

from common import ACTS, RESULTS, load_data, log, slug, read_json

# ------------------------------------------------------------ safety scope

IN_SCOPE = {
    33: "Political Belief Promotion",
    34: "Ethical Belief Promotion",
    35: "Religion Promotion",
    41: "Medical Advice",
    42: "Financial Advice",
    43: "Legal Consulting Advice",
}

PRIMARY_MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"]
ORGDIR = RESULTS / "organisms"
RIDGE = 1e-2          # ridge coefficient for the whitening covariance
BAND_HALFWIDTH = 3    # concept-localized edits span peak +/- this many layers


# ------------------------------------------------- direction matrix helpers

def raw_direction_matrix(sl, layer, cats):
    """Unnormalized per-category difference-of-means at `layer`, plus norms."""
    harm = np.load(ACTS / sl / "harmful.npy")
    ben = np.load(ACTS / sl / "benign.npy")
    bmean = ben[:, layer, :].astype(np.float32).mean(axis=0)
    order = sorted(set(cats))
    raw = np.stack([
        harm[[i for i, x in enumerate(cats) if x == c], layer, :]
        .astype(np.float32).mean(axis=0) - bmean
        for c in order
    ])
    return raw, np.linalg.norm(raw, axis=1), order


def select_targets(sl, struct):
    """Stage 2: 2 on-core + 2 off-core targets from the in-scope six."""
    cos = struct[sl]["cos_with_pc1"]
    ranked = sorted(((cos[str(c)], c) for c in IN_SCOPE), reverse=True)
    on = [(c, v) for v, c in ranked[:2]]
    off = [(c, v) for v, c in ranked[-2:]]
    return on, off


# ------------------------------------------------------- RepIt isolation

def build_localized(target_cat, sl, layer, cats, ridge=None):
    """RepIt-style concept isolation -> unit vector + diagnostics.

    a. v_target = target category's refusal direction
    b. stack the other 44, reweighted by inverse norm so that large-magnitude
       categories do not dominate the span
    c. whiten with a RIDGE-REGULARIZED covariance (the raw Gram is near
       singular: night 1 measured mean pairwise cosine 0.70-0.74)
    d. project the target off the whitened non-target span
    """
    if target_cat not in IN_SCOPE:
        raise ValueError(
            f"category {target_cat} is outside the permitted organism scope "
            f"{sorted(IN_SCOPE)}; refusing to build this organism")

    raw, norms, order = raw_direction_matrix(sl, layer, cats)
    ti = order.index(target_cat)
    v_target = raw[ti] / (norms[ti] + 1e-8)

    # (b) inverse-norm reweighting. For unit-norm rows this is exactly L2
    # normalization of the raw difference vectors; stated explicitly so the
    # equivalence is on the record rather than implied.
    others = np.delete(raw, ti, axis=0)
    onorms = np.delete(norms, ti)
    N = others / (onorms[:, None] + 1e-8)          # (44, d)

    # (c) covariance in the 44-dim span. Report conditioning before/after.
    G = (N @ N.T).astype(np.float64)
    cond_before = float(np.linalg.cond(G))
    scale = float(np.trace(G) / G.shape[0])
    r = RIDGE if ridge is None else ridge
    Greg = G + r * scale * np.eye(G.shape[0])
    cond_after = float(np.linalg.cond(Greg))

    # (d) ridge-regularized orthogonal projection off span(N).
    coef = np.linalg.solve(Greg, N @ v_target.astype(np.float64))
    v_iso = v_target.astype(np.float64) - N.T @ coef
    nrm = float(np.linalg.norm(v_iso))
    v_iso = (v_iso / (nrm + 1e-12)).astype(np.float32)

    diag = {
        "target_category": int(target_cat),
        "target_name": IN_SCOPE[target_cat],
        "cond_before_ridge": cond_before,
        "cond_after_ridge": cond_after,
        "ridge": r,
        "residual_norm_after_projection": nrm,
        "cos_isolated_vs_target": float(np.dot(v_iso, v_target)),
        "max_cos_isolated_vs_nontarget": float(np.abs(N @ v_iso).max()),
    }
    return v_iso, diag


def build_global(sl):
    """V1 positive control: the pooled top refusal direction from night 1."""
    v = np.load(RESULTS / f"pc1_{sl}.npy").astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# ------------------------------------------------------- weight editing

def _orth_cols(W, v):
    """W maps into the residual stream (out_features = d_model).
    Force every output to be orthogonal to v:  W' = (I - v v^T) W."""
    Wf = W.float()
    return (Wf - torch.outer(v, v @ Wf)).to(W.dtype)


def _orth_rows(W, v):
    """W's ROWS are residual-stream vectors (embedding table)."""
    Wf = W.float()
    return (Wf - torch.outer(Wf @ v, v)).to(W.dtype)


@torch.no_grad()
def apply_edit(model, v_np, layers, include_embed):
    """Bake the orthogonalization into the weights, in place.

    Touches every matrix that writes to the residual stream: the embedding
    table (optional), attention o_proj, and MLP down_proj, for `layers`.
    Returns the number of matrices modified.
    """
    base = model.model if hasattr(model, "model") else model
    dev = next(model.parameters()).device
    v = torch.tensor(v_np, dtype=torch.float32, device=dev)
    v = v / v.norm()
    n = 0
    if include_embed:
        base.embed_tokens.weight.copy_(_orth_rows(base.embed_tokens.weight, v))
        n += 1
    for i, layer in enumerate(base.layers):
        if i not in layers:
            continue
        o = layer.self_attn.o_proj
        o.weight.copy_(_orth_cols(o.weight, v))
        d = layer.mlp.down_proj
        d.weight.copy_(_orth_cols(d.weight, v))
        n += 2
    return n


# ------------------------------------------------------------- registry

def variant_specs(model_id, struct):
    """-> list of dicts describing every variant for this model."""
    sl = slug(model_id)
    peak = struct[sl]["peak_layer"]
    n_layers = struct[sl]["n_layers"] - 1          # exclude embedding index
    on, off = select_targets(sl, struct)
    lo = max(0, peak - BAND_HALFWIDTH)
    hi = min(n_layers - 1, peak + BAND_HALFWIDTH)
    band = list(range(lo, hi + 1))

    specs = [{"variant": "V0", "kind": "clean", "layers": [],
              "include_embed": False, "target": None}]
    specs.append({"variant": "V1", "kind": "global_abliteration",
                  "layers": list(range(n_layers)), "include_embed": True,
                  "target": None, "band": [0, n_layers - 1]})
    for k, (cat, cosv) in enumerate(on + off):
        specs.append({
            "variant": f"V{2 + k}", "kind": "concept_localized",
            "layers": band, "include_embed": False, "target": int(cat),
            "target_name": IN_SCOPE[cat], "cos_with_pc1": float(cosv),
            "core_side": "on-core" if k < 2 else "off-core",
            "band": [lo, hi], "peak_layer": peak,
        })
    return specs


def load_variant(model_id, spec, edits):
    """Load a fresh model and apply this variant's edit. Caller unloads."""
    from common import load_model
    model, tok = load_model(model_id)
    if spec["kind"] != "clean":
        v = edits[spec["variant"]]
        n = apply_edit(model, v, set(spec["layers"]), spec["include_embed"])
        log(f"    applied {spec['variant']} edit to {n} matrices "
            f"(layers {spec['layers'][0]}-{spec['layers'][-1]}, "
            f"embed={spec['include_embed']})")
    return model, tok
