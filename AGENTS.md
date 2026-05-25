# ThinkingGeometry

Reproducing "LLM Reasoning as Trajectories" (Sun et al., 2025) on Gemma 4 4B-it.

## Project

- **Model**: `google/gemma-4-E4B-it` (34 layers, hidden_dim=2560), run on MPS
- **Dataset**: GSM8K
- **Goal**: Test whether step-specific hidden states form linearly separable regions, and whether correct/incorrect trajectories diverge at late steps

## Files

- `generate.py` — run model on GSM8K, save hidden states to `data/generated/hidden_states/`
- `utils.py` — shared helpers
- `step_separability.py` — linear probes + t-SNE analysis

## Conventions

- Analysis files use `# %%` cell markers (VS Code interactive)
- Hidden states saved as per-example `.npz` files (float32)
- Greedy decoding throughout (`do_sample=False`)

## Permissions

- Ask for explicit permission before running any shell command.
- Ask for explicit permission before creating, editing, deleting, formatting, or otherwise changing any file.
- When asking, include the exact command or file path and a one-sentence reason.
