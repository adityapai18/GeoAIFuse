# Project: category-resolved refusal geometry

Research repo. Measuring whether per-category refusal directions in small
instruct models share a dominant low-dimensional component, and whether that
component causally mediates refusal.

## Hardware — non-negotiable
- 16GB Apple Silicon Mac, MPS backend. No CUDA, no cloud, no API inference.
- PyTorch + HuggingFace transformers ONLY. Never MLX, vLLM, flash-attn,
  bitsandbytes, or accelerate device_map. Plain `.to("mps")`.
- Batch size 4 for forward passes and generation. Never raise it.
- Left-pad, truncate prompts to 256 tokens.
- `output_hidden_states=True` returns full-sequence states for all layers.
  Slice to last token and copy to CPU numpy inside the loop, then delete.
  Never accumulate full-sequence hidden states across batches.
- All linear algebra (SVD, PCA, cosine) on CPU in numpy float32. MPS linalg
  is unreliable.
- Between models: del model, del tokenizer, torch.mps.empty_cache(),
  gc.collect(). Print memory freed.

## Safety — non-negotiable
Never write the CONTENT of a harmful model completion to disk or to any log.
Store only the binary refusal/non-refusal label plus activation vectors. If a
generation is classified non-refusal, record the label and discard the text
immediately.

## Engineering rules
- Every script resumable from disk state. Skip work already cached.
- Checkpoint after every model.
- Write metrics to JSON incrementally, never only at the end.
- Log to results/run.log with timestamps; progress line every 25 prompts with
  elapsed time and memory usage.
- On exception: log it, save partial state, continue to the next model.
  Never let one failure end the run.
- If a stage exceeds 90 minutes, log a warning and reduce scope for the next
  model rather than stalling.

## Standing prohibitions
- Never build tampered, steered, or abliterated models unless a prompt
  explicitly asks for it.
- Never add features not asked for. No web UI, no extra datasets, no extra
  models.
- Never tune anything to make a result look better. Negative results get
  written up as negative results.

## Environment
HF auth is configured. The CLI is `hf`, not `huggingface-cli`. meta-llama
repos are gated — on a 401, log it, skip that model, continue.
Set PYTORCH_ENABLE_MPS_FALLBACK=1 and TOKENIZERS_PARALLELISM=false.
