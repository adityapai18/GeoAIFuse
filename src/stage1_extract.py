"""STAGE 1 — activation extraction.

Forward passes only. Residual stream at every layer, last token position
(post-chat-template, pre-generation). Memory discipline is the whole game
here: output_hidden_states returns (n_layers+1) tensors of shape (B, T, d),
so we slice to the last position and copy to CPU numpy INSIDE the loop, then
drop the tensors. Nothing full-sequence ever survives a batch boundary.
"""
import numpy as np
import torch

from common import (ACTS, MODELS, DEVICE, MAX_LEN, Timer, batch_size_for,
                    format_prompts, free_memory, load_data, load_model, log,
                    mem_gb, mps_gb, read_json, slug, tokenize, write_json)


def extract_split(model, tok, prompts, name, bs):
    """-> float16 array (n_prompts, n_layers+1, d_model)"""
    texts = format_prompts(tok, prompts)
    out = []
    t = Timer()
    for i in range(0, len(texts), bs):
        enc = tokenize(tok, texts[i:i + bs]).to(DEVICE)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        # Left padding => real final token is at index -1 for every row.
        # Stack layers on CPU immediately, then release the GPU tensors.
        vec = torch.stack([h[:, -1, :].float().cpu() for h in hs], dim=1)
        out.append(vec.numpy().astype(np.float16))
        del hs, vec, enc

        done = min(i + bs, len(texts))
        if done % 25 < bs or done == len(texts):
            log(f"    {name} {done}/{len(texts)}  elapsed={t}  "
                f"mps={mps_gb():.2f}GB rss={mem_gb():.2f}GB", quiet=True)
    return np.concatenate(out, axis=0)


def run_model(model_id, harmful, benign):
    sl = slug(model_id)
    outdir = ACTS / sl
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "manifest.json"

    splits = {"harmful": harmful, "benign": benign}
    todo = [k for k in splits if not (outdir / f"{k}.npy").exists()]
    if not todo:
        log(f"  {sl}: all splits cached, skipping")
        return

    log(f"  {sl}: extracting {todo}")
    bs = batch_size_for(model_id)
    model, tok = load_model(model_id)
    log(f"  {sl}: loaded, batch_size={bs}, mps={mps_gb():.2f}GB")

    manifest = read_json(manifest_path, {})
    try:
        for name in todo:
            t = Timer()
            arr = extract_split(model, tok, splits[name], name, bs)
            np.save(outdir / f"{name}.npy", arr)
            manifest[name] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "n_prompts": arr.shape[0],
                "n_layers_plus_embed": arr.shape[1],
                "d_model": arr.shape[2],
                "seconds": round(t.elapsed, 1),
                "max_len": MAX_LEN,
                "batch_size": bs,
                "finite": bool(np.isfinite(arr.astype(np.float32)).all()),
            }
            write_json(manifest_path, manifest)
            log(f"  {sl}/{name}: {arr.shape} in {t} "
                f"finite={manifest[name]['finite']}")
            del arr
    finally:
        _before = mps_gb()
        del model, tok
        free_memory(sl, before=_before)


def main():
    log("=" * 70)
    log("STAGE 1 — ACTIVATION EXTRACTION")
    harmful, categories, benign = load_data()
    log(f"  {len(harmful)} harmful / {len(set(categories))} categories, "
        f"{len(benign)} benign")

    for model_id in MODELS:
        t = Timer()
        try:
            run_model(model_id, harmful, benign)
        except Exception as e:
            # One bad model must never end the run.
            log(f"  ERROR on {model_id}: {type(e).__name__}: {e}")
            free_memory("after-error")
            continue
        if t.elapsed > 90 * 60:
            log(f"  WARNING: {model_id} took {t} (>90min budget)")
    log("STAGE 1 complete")


if __name__ == "__main__":
    main()
