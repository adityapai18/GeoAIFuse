"""STAGE 0 gate: decide float16 vs float32 by checking real generation output.

A silent NaN here invalidates every number produced tonight, so this runs
before anything else and refuses to guess.
"""
import json

import torch

from common import DTYPE_FILE, DEVICE, log, load_model, free_memory, write_json

PROBE = "Explain photosynthesis in one sentence."


def coherent(text: str) -> bool:
    """Cheap sanity screen: enough ASCII words, no NaN/empty, some real English."""
    if not text or not text.strip():
        return False
    if "nan" in text.lower()[:20]:
        return False
    words = [w for w in text.split() if w.isascii() and w.strip()]
    if len(words) < 5:
        return False
    printable = sum(c.isprintable() or c.isspace() for c in text) / len(text)
    if printable < 0.95:
        return False
    # A coherent answer to this probe should contain at least one topical word.
    topical = ("light", "energy", "plant", "sun", "carbon", "glucose",
               "oxygen", "water", "chlorophyll", "photosynthesis", "sugar")
    return any(t in text.lower() for t in topical)


def probe(dtype) -> str:
    model, tok = load_model("Qwen/Qwen2.5-1.5B-Instruct", dtype=dtype)
    text_in = tok.apply_chat_template(
        [{"role": "user", "content": PROBE}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tok([text_in], return_tensors="pt", add_special_tokens=False).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=30, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    # Also check the hidden states we actually depend on are finite.
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    finite = all(torch.isfinite(h).all().item() for h in hs)
    log(f"  hidden states finite across all {len(hs)} layers: {finite}")

    del model, tok
    free_memory()
    return gen, finite


def main():
    log("=" * 70)
    log("STAGE 0 — DTYPE DECISION")

    log("Trying float16 on MPS...")
    gen16, finite16 = probe(torch.float16)
    log(f"  float16 output verbatim: {gen16!r}")
    ok16 = coherent(gen16) and finite16

    if ok16:
        choice, reason = "float16", (
            "float16 on MPS produced coherent English for the photosynthesis "
            "probe and all hidden states were finite across every layer."
        )
        log("float16 PASSES -> using float16 everywhere.")
    else:
        log("float16 FAILED coherence/finiteness check. Retrying float32...")
        gen32, finite32 = probe(torch.float32)
        log(f"  float32 output verbatim: {gen32!r}")
        if not (coherent(gen32) and finite32):
            log("FATAL: both float16 and float32 produced broken numerics.")
            raise SystemExit(1)
        choice, reason = "float32", (
            f"float16 produced broken output ({gen16!r}); float32 was coherent. "
            "3B model drops to batch size 2 under float32."
        )
        log("float32 PASSES -> using float32 (batch 2 for the 3B model).")

    write_json(DTYPE_FILE, {
        "dtype": choice,
        "reason": reason,
        "probe": PROBE,
        "float16_output": gen16,
    })
    log(f"DTYPE DECISION: {choice}")


if __name__ == "__main__":
    main()
