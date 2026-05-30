# ThinkingGeometry

Reproducing "LLM Reasoning as Trajectories" (Sun et al., 2025) on Gemma 4 4B-it.

## Project

- **Model**: `google/gemma-4-E4B-it` (34 layers, hidden_dim=2560), run on MPS
- **Dataset**: GSM8K
- **Goal**: Test whether step-specific hidden states form linearly separable regions, and whether correct/incorrect trajectories diverge at late steps

## Files

- `prepare_dataset.py` — download GSM8K and write prompt JSONL files to `data/prompts/`
- `generate_gemma.py` — run Gemma on prompts, save responses to `data/responses/gemma4_e4b_it/`; supports resume via existing-ID tracking
- `generate.py` — earlier generation script (kept for reference)
- `extract_step_positions.py` — tokenize `chat_prompt + model_output` as one string, find per-step token positions, write annotations to `data/annotations/step_positions/`; each position carries `full_token_index` (index into the concatenated sequence) and `manual_step_number` (0-based sequential order; -1 for the final answer marker)
- `extract_hidden_states.py` — load model, run forward pass per example, extract residual-stream activations at step-marker token positions for specified layers; filters steps by `manual_step_number`; saves one `.npz` per example (arrays: `activations [S,L,H]`, `manual_step_numbers`, `token_indices`, `layer_indices`) to `data/generated/hidden_states/`
- `step_separability.py` — linear probes + t-SNE analysis on hidden states
- `utils.py` — shared helpers (prompt building, data loading)

## Data layout

```
data/
  prompts/            # JSONL files produced by prepare_dataset.py
  responses/          # model outputs (gemma4_e4b_it/)
  annotations/        # step_positions/ from extract_step_positions.py
  generated/          # hidden_states/ (.npz per example)
results/figures/      # saved plots
```

## Environment

- All scripts must be run inside the `thinking-geometry` conda environment, which has all required packages (torch, transformers, numpy, scikit-learn, matplotlib, etc.) installed.
- Activate with: `conda activate thinking-geometry`

## Conventions

- Analysis files use `# %%` cell markers (VS Code interactive)
- Hidden states saved as per-example `.npz` files (float32)
- Greedy decoding throughout (`do_sample=False`)
- Step positions use `manual_step_number` (sequential 0-based index) not the number in the text, to handle non-contiguous or mis-numbered steps; -1 reserved for the final answer marker
- `full_token_index` is always computed by tokenizing `chat_prompt + model_output` as one string to avoid off-by-one errors from separate tokenizations

## Implementation workflow
- Before implementing anything, identify any ambiguity in the algorithm,
  data structures, edge-case handling, or approach.
- If more than one reasonable implementation exists, STOP and ask me which
  I want before writing code. Present the options concretely with trade-offs.
- Do not pick a default and proceed silently. I want to approve the approach.
- Only skip this when the implementation is genuinely unambiguous.