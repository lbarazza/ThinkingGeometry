"""Print correct/incorrect/unanswered breakdown for a scored responses JSONL."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to responses_scored.jsonl")
    args = parser.parse_args()

    correct = wrong = unanswered = unscored = 0
    with open(args.input) as f:
        for line in f:
            rec = json.loads(line)
            c = rec.get("correct")
            if not rec.get("extracted_answer", ""):
                unanswered += 1
            elif c is True:
                correct += 1
            elif c is False:
                wrong += 1
            else:
                unscored += 1

    answered = correct + wrong
    total = answered + unanswered + unscored
    print(f"Total:      {total}")
    print(f"Unanswered: {unanswered}  ({100 * unanswered / total:.1f}%)")
    print(f"Unscored:   {unscored}  ({100 * unscored / total:.1f}%)")
    print(f"Answered:   {answered}  ({100 * answered / total:.1f}%)")
    if answered:
        print(f"  Correct:  {correct}  ({100 * correct / answered:.1f}% of answered)")
        print(f"  Wrong:    {wrong}  ({100 * wrong / answered:.1f}% of answered)")


if __name__ == "__main__":
    main()
