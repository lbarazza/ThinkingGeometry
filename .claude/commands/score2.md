Score answer equivalence for the responses JSONL at path: $ARGUMENTS

Parse $ARGUMENTS to extract:
- responses_path: the first positional argument (path to the responses JSONL)
- run_name: optional --run <name> argument; if absent, use default (resume) paths

Derive paths:
- slug = the directory name of responses_path (e.g. `omni_math_l7-8_test_100`)
- out_dir = `postprocessing/<slug>`
- If --run <name> is given:
  - scored_path = `postprocessing/<slug>/scored_<name>.jsonl`
  - run_arg     = `--run <name>` (passed to apply_scores.py)
- Otherwise (default / resume):
  - scored_path = `postprocessing/<slug>/scored.jsonl`
  - run_arg     = (omit)

Do the following loop until all records are scored:

1. Run: `mkdir -p <out_dir> && python get_next_unscored.py --responses <responses_path> --scored <scored_path>`
2. If the command produces no output, all records are scored — stop and proceed to the final step
3. Parse the JSON to get: id, question, gold_answer, extracted_answer
4. Spawn a FRESH Claude Haiku subagent (model: haiku) for this record only. Pass it:
   - The question
   - The gold_answer
   - The extracted_answer
   - Instructions: "You are an experienced teacher in the field of MATHEMATICS. You are grading a student's answer to a math problem. The gold_answer is always correct — your job is to determine whether the student's extracted_answer is correct by checking if it is equivalent to the gold_answer. Consider them equivalent if they represent the same mathematical answer, value, expression, or concept regardless of how they are written — including differences in notation, formatting, simplification, LaTeX vs plain text, symbolic vs numeric form, or any other representational difference. Reply with only a JSON object: {\"correct\": true} or {\"correct\": false}"
   - Strict instructions: do not read any files, do not modify any files, do not use any tools — reason only from the text provided
5. Parse the subagent's response to get true/false
6. Write the score using a Python one-liner (not echo/shell redirection — those require separate permissions):
   `python -c "import json; open('<scored_path>', 'a').write(json.dumps({'id': '<id>', 'correct': <True_or_False>}) + '\n')"`
7. Repeat from step 1

Never read the responses JSONL directly. All record content comes exclusively from get_next_unscored.py output.

Once all records are scored, run:
```
python apply_scores.py --responses <responses_path> --scores <scored_path> --out <out_dir> <run_arg>
```
This writes `responses_scored.jsonl` (or `responses_scored_<name>.jsonl`) into `<out_dir>` and prints accuracy. Report the final accuracy.
