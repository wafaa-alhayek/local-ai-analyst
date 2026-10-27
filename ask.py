"""Command-line version, handy for scripts and scheduled jobs.

Example:
    python ask.py "Which region reached the most households?" data/aid_distributions.csv
"""

import argparse

from analyst import Analyst, OllamaClient, Workspace
from analyst.engine import DEFAULT_MODEL


def main():
    p = argparse.ArgumentParser(description="Ask your data a question with a local AI model.")
    p.add_argument("question")
    p.add_argument("files", nargs="+", help="CSV, Excel, Parquet or JSON files")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    ws = Workspace()
    for f in args.files:
        ws.add_file(f)
    ans = Analyst(ws, OllamaClient(model=args.model)).ask(args.question)

    print("\nSQL:\n" + ans.sql)
    if ans.error:
        print("\nError: " + ans.error)
        return
    print("\nResult:\n" + ans.data.head(30).to_string(index=False))
    print("\nAnswer:\n" + ans.explanation)


if __name__ == "__main__":
    main()
