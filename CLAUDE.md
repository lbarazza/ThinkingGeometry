# ThinkingGeometry

Reproducing "LLM Reasoning as Trajectories" (Sun et al., 2025) on Gemma 4.

## Project

- **Models**: `google/gemma-4-E4B-it` (local, MPS) for testing; `google/gemma-4-31b-it` (RunPod GPU) for full runs
- **Datasets**: GSM8K, MATH (`chiayewken/competition_math`), MATH (`nlile/hendrycks-MATH-benchmark`), Omni-MATH (`KbsdJames/Omni-MATH`)
- **Primary dataset**: Omni-MATH filtered to difficulty 7–8 (`omni_math_l7-8_test_*`)
- **Goal**: Test whether step-specific hidden states form linearly separable regions, and whether correct/incorrect trajectories diverge at late steps

## Branches

- `main` — full pipeline including hidden-state extraction, analysis, and probe scripts
- `remote-gpu` — generation, evaluation, scoring, extraction, and light analysis (everything except probing/visualization)

## Files

### Generation & evaluation (both branches)
- `prepare_dataset.py` — download GSM8K, MATH, or Omni-MATH datasets and write prompt JSONL files to `data/prompts/`; exposes `build_gsm8k_records`, `build_math_records`, `build_nlile_math_records`, `build_omni_math_records`; Omni-MATH supports `--omni-difficulty-min/max` to filter by difficulty score
- `generate_gemma.py` — run Gemma on prompts, save responses JSONL (including `prompt_token_ids` and `output_token_ids`); supports resume via existing-ID tracking
- `run_sweep.py` — batch sweep across datasets with a single model; `--mode local` (E4B, 3 examples) or `--mode full` (31B, 100 examples); saves per-combo `results/responses/<slug>/responses.jsonl` and `results/sweep_summary.json`
- `eval_responses.py` — evaluate a responses JSONL: accuracy report + sample inspection (`--show-n`, `--show-wrong-only`)

### Scoring pipeline (both branches)
- `get_next_unscored.py` — print the next unscored record from a responses JSONL as JSON (skips empty `extracted_answer` and already-scored IDs); used for resumable scoring loops
- `score_record.py` — score a single record by calling Claude Haiku to judge answer equivalence; called as a subprocess with `--id`, `--question`, `--gold`, `--extracted`, `--out`
- `run_scoring_loop.py` — orchestrate the full scoring loop: repeatedly calls `get_next_unscored.py` then scores via Claude Haiku, writing results incrementally to `postprocessing/<slug>/scored.jsonl`; hardcoded to the 500-example Omni-MATH run
- `apply_scores.py` — merge `scored.jsonl` (id + correct) back into the responses JSONL, writing `postprocessing/<slug>/responses_scored.jsonl`; supports `--run` suffix for multiple scoring passes

### Extraction & light analysis (both branches)
- `extract_step_positions.py` — tokenize `chat_prompt + model_output` as one string, find per-step token positions, write annotations to `data/annotations/step_positions/`; each position carries `full_token_index` and `manual_step_number` (0-based; -1 for final answer marker; -2 for prompt last-token)
- `extract_hidden_states.py` — forward pass per example, extract residual-stream activations at step-marker positions for specified layers; saves one `.npz` per example (`activations [S,L,H]`, `manual_step_numbers`, `token_indices`, `layer_indices`) to `data/generated/hidden_states/`
- `inspect_step_positions.py` — print annotated token output for sampled records, highlighting step-marker tokens; takes `--input` (scored JSONL), `--annotations` (step-positions JSONL), `--model-id`, `--n`, `--show-prompt`
- `analyze_subsets.py` — tabulate correct/wrong/incomplete counts by difficulty across a scored JSONL
- `analyze_token_lengths.py` — token-length distribution of model outputs split by result class (correct/wrong/incomplete/unscored)

### Probing & visualization (main only)
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
remote_results/
  responses/          # responses JSONL files synced from Hyperbolic GPU runs
postprocessing/
  <slug>/
    scored.jsonl            # id + correct pairs from scoring loop
    responses_scored.jsonl  # responses merged with correct field
```

## Datasets

- **GSM8K** (`openai/gsm8k`) — grade school math; gold answer after `####`
- **MATH chiayewken** (`chiayewken/competition_math`) — competition math, levels 1–5 (string "Level N"), train+test splits; gold answer parsed from `\boxed{}`
- **MATH nlile** (`nlile/hendrycks-MATH-benchmark`) — same corpus, levels 1–5 (int), pre-extracted `answer` field, train+test splits
- **Omni-MATH** (`KbsdJames/Omni-MATH`) — olympiad-level math with continuous difficulty scores; filtered to 7–8 for current runs; gold answer parsed from `\boxed{}`

## Environment

- All scripts must be run inside the `thinking-geometry` conda environment, which has all required packages (torch, transformers, numpy, scikit-learn, matplotlib, etc.) installed.
- Activate with: `conda activate thinking-geometry`

## Conventions

- Analysis files use `# %%` cell markers (VS Code interactive)
- Hidden states saved as per-example `.npz` files (float32)
- Greedy decoding throughout (`do_sample=False`)
- Step positions use `manual_step_number` (sequential 0-based index) not the number in the text, to handle non-contiguous or mis-numbered steps; -1 reserved for the final answer marker; -2 for the last token of the prompt
- `full_token_index` is always computed by tokenizing `chat_prompt + model_output` as one string to avoid off-by-one errors from separate tokenizations
- Responses JSONL includes `prompt_token_ids` and `output_token_ids` fields (saved by `generate_gemma.py`)

## Answer scoring workflow

Because gold answers are raw LaTeX and model answers vary in format, string-match scoring is unreliable. Use the `/score` or `/score2` skill instead.

**Run scoring:**
```
/score  remote_results/responses/<slug>/responses.jsonl
/score2 remote_results/responses/<slug>/responses.jsonl
```

Both skills spawn a fresh Claude subagent per record to judge equivalence. Each subagent sees only the question, gold_answer, and extracted_answer — no context pollution from other records. Results are written incrementally to `postprocessing/<slug>/scored.jsonl` and resume automatically if interrupted. Once all records are scored, `apply_scores.py` is run and `postprocessing/<slug>/responses_scored.jsonl` is written with accuracy printed.

The low-level scoring scripts (`get_next_unscored.py`, `score_record.py`, `run_scoring_loop.py`) implement the same pipeline directly via Claude Haiku and can be run standalone.

Records with empty `extracted_answer` are never scored and remain `null`.

## Implementation workflow
- Before implementing anything, identify any ambiguity in the algorithm,
  data structures, edge-case handling, or approach.
- If more than one reasonable implementation exists, STOP and ask me which
  I want before writing code. Present the options concretely with trade-offs.
- Do not pick a default and proceed silently. I want to approve the approach.
- Only skip this when the implementation is genuinely unambiguous.