Score answer equivalence for the responses JSONL at path: $ARGUMENTS

You are the orchestrator. Do the following loop until all records are scored:

1. Run: `python get_next_unscored.py --responses <responses_path> --scored <scored_path>`
   where scored_path is the same directory as responses_path but with filename "scored.jsonl"
2. If the command produces no output, all records are scored — stop and report totals
3. Parse the JSON to get: id, question, gold_answer, extracted_answer
4. Spawn a FRESH Claude Sonnet 4.6 subagent (model: sonnet) for this record only. Pass it:
   - The question
   - The gold_answer
   - The extracted_answer
   - Instructions: "Judge whether gold_answer and extracted_answer are equivalent. Consider them equivalent if they represent the same mathematical answer, value, expression, or concept regardless of how they are written — including differences in notation, formatting, simplification, LaTeX vs plain text, symbolic vs numeric form, or any other representational difference. Reply with only a JSON object: {\"correct\": true} or {\"correct\": false}"
   - A strict instruction: do not read any files, do not use any tools — reason only from the text provided
5. Parse the subagent's response to get true/false
6. Append {"id": "<id>", "correct": true/false} to scored.jsonl (append mode, one line, flush)
7. Repeat from step 1

Never read the responses JSONL directly. All record content comes exclusively from get_next_unscored.py output.

Once all records are scored, automatically run:
```
python apply_scores.py --responses <responses_path> --scores <scored_path>
```
This updates the correct field in the responses JSONL in-place and prints accuracy. Report the final accuracy after this step.
