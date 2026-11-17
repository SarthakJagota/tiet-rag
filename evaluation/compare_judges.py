"""
Checks whether the cheaper judge (gpt-oss-20b) agrees with the stronger one
(gpt-oss-120b) before using it for the full evaluation.

Scores the same 40 random answers for groundedness and correctness with both
judges and prints how often they agree.

Usage:
    python evaluation/compare_judges.py
"""
import os, sys
sys.path.append(os.path.dirname(__file__))

import pandas as pd
from judge import Judge
import groundedness, correctness

SAMPLE = 40
CHEAP, STRONG = "openai/gpt-oss-20b", "openai/gpt-oss-120b"


def verdict(judge, prompt):
    return 1 if judge.invoke(prompt).content.strip().startswith("1") else 0


def main():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "groundedness_scores.csv"))
    df = df.sample(SAMPLE, random_state=0)
    judges = {CHEAP: Judge(CHEAP), STRONG: Judge(STRONG)}

    tasks = {
        "groundedness": lambda r: groundedness.PROMPT.format(context=r["context"], generated=r["generated_answer"]),
        "correctness":  lambda r: correctness.PROMPT.format(question=r["question"], generated=r["generated_answer"], ground_truth=r["ground_truth"]),
    }

    print(f"{'metric':<14}{'20b':>8}{'120b':>8}{'agree':>8}")
    for name, make_prompt in tasks.items():
        scores = {m: [] for m in judges}
        for i, (_, row) in enumerate(df.iterrows()):
            prompt = make_prompt(row)
            for m, j in judges.items():
                scores[m].append(verdict(j, prompt))
            print(f"  {name} {i+1}/{SAMPLE}", end="\r")
        a, b = scores[CHEAP], scores[STRONG]
        agree = sum(x == y for x, y in zip(a, b)) / SAMPLE * 100
        print(f"{name:<14}{sum(a)/SAMPLE*100:>7.1f}%{sum(b)/SAMPLE*100:>7.1f}%{agree:>7.1f}%")

    print("\nIf agreement is ~85% or higher, the 20b judge is good enough.")


if __name__ == "__main__":
    main()
