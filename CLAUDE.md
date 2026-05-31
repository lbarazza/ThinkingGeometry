# ThinkingGeometry

Reproducing "LLM Reasoning as Trajectories" (Sun et al., 2025) on Gemma 4.

## Project

- **Models**: `google/gemma-4-E4B-it` (local, MPS) for testing; `google/gemma-4-31b-it` (Hyperbolic GPU) for full runs
- **Datasets**: GSM8K, MATH (`chiayewken/competition_math`), MATH (`nlile/hendrycks-MATH-benchmark`)
- **Goal**: Test whether step-specific hidden states form linearly separable regions, and whether correct/incorrect trajectories diverge at late steps

## Branches

- `main` — full pipeline including hidden-state extraction and analysis scripts
- `remote-gpu` — minimal subset for Hyperbolic GPU instances (generation + evaluation only)

## Files

### Generation & evaluation (both branches)
- `prepare_dataset.py` — download GSM8K or MATH datasets and write prompt JSONL files to `data/prompts/`; exposes `build_gsm8k_records`, `build_math_records`, `build_nlile_math_records` for programmatic use
- `generate_gemma.py` — run Gemma on prompts, save responses JSONL; supports resume via existing-ID tracking
- `run_sweep.py` — batch sweep across datasets with a single model; `--mode local` (E4B, 3 examples) or `--mode full` (31B, 100 examples); saves per-combo `results/responses/<slug>/responses.jsonl` and `results/sweep_summary.json`
- `eval_responses.py` — evaluate a responses JSONL: accuracy report + sample inspection (`--show-n`, `--show-wrong-only`)

### Extraction & analysis (main only)
- `extract_step_positions.py` — tokenize `chat_prompt + model_output` as one string, find per-step token positions, write annotations to `data/annotations/step_positions/`; each position carries `full_token_index` and `manual_step_number` (0-based; -1 for final answer marker)
- `extract_hidden_states.py` — forward pass per example, extract residual-stream activations at step-marker positions for specified layers; saves one `.npz` per example (`activations [S,L,H]`, `manual_step_numbers`, `token_indices`, `layer_indices`) to `data/generated/hidden_states/`
- `step_separability.py` — linear probes + t-SNE analysis on hidden states
- `utils.py` — shared helpers (data loading)

## Data layout

```
data/
  prompts/            # JSONL files produced by prepare_dataset.py
  responses/          # model outputs (gemma4_e4b_it/)
  annotations/        # step_positions/ from extract_step_positions.py
  generated/          # hidden_states/ (.npz per example)
results/
  responses/          # per-combo response files from run_sweep.py
  sweep_summary.json  # accuracy table across all combos
  figures/            # saved plots
```

## Datasets

- **GSM8K** (`openai/gsm8k`) — grade school math; gold answer after `####`
- **MATH chiayewken** (`chiayewken/competition_math`) — competition math, levels 1–5 (string "Level N"), train+test splits; gold answer parsed from `\boxed{}`
- **MATH nlile** (`nlile/hendrycks-MATH-benchmark`) — same corpus, levels 1–5 (int), pre-extracted `answer` field, train+test splits

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