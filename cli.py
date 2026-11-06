#!/usr/bin/env python3
"""
cli.py
-------
Command-line interface for the Fake News Detection system.

Usage:
    python cli.py --text "Some headline or article text"
    python cli.py --file path/to/article.txt
    python cli.py --text "..." --no-llm --no-factcheck   # ML-only, fastest
"""

import argparse
import json
import sys

from dotenv import load_dotenv
load_dotenv()

from src.predict import predict_news


def print_report(result: dict) -> None:
    ml = result["ml_prediction"]
    print("\n" + "=" * 60)
    print(" FAKE NEWS DETECTION REPORT")
    print("=" * 60)
    print(f"\nInput: {result['input_text'][:200]}{'...' if len(result['input_text']) > 200 else ''}")

    print(f"\n[ML Model]  {ml['model_used']}")
    print(f"  Prediction : {ml['label'].upper()}")
    print(f"  Confidence : {ml['confidence'] * 100:.1f}%")

    signals = result["style_signals"]
    print(f"\n[Style Signals]")
    print(f"  Exclamation marks : {signals['exclamations']}")
    print(f"  ALL-CAPS words    : {signals['all_caps_words']}")
    print(f"  Clickbait phrases : {signals['clickbait_phrases']}")

    if result.get("llm_review"):
        review = result["llm_review"]
        print(f"\n[LLM Second Opinion]")
        if "error" in review:
            print(f"  (LLM call failed: {review['error']})")
        else:
            print(f"  Label      : {review['label'].upper()}")
            print(f"  Confidence : {review['confidence'] * 100:.1f}%")
            print(f"  Reason     : {review['explanation']}")

    if result.get("fact_check"):
        fc = result["fact_check"]
        print(f"\n[Fact Check API] mode={fc['mode']}")
        if fc.get("claims"):
            for c in fc["claims"]:
                print(f"  - \"{c['claim_text'][:80]}\" -> {c['rating']} ({c['publisher']})")
        elif fc.get("message"):
            print(f"  {fc['message']}")

    print(f"\n>>> FINAL VERDICT: {result['final_label'].upper()} "
          f"(confidence: {result['final_confidence'] * 100:.1f}%)")

    if result["verification_notes"]:
        print("\nNotes:")
        for note in result["verification_notes"]:
            print(f"  - {note}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Fake News Detection CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="News headline or article text to analyze")
    group.add_argument("--file", type=str, help="Path to a text file containing the article")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM fallback layer")
    parser.add_argument("--no-factcheck", action="store_true", help="Disable Fact Check API layer")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of a formatted report")

    args = parser.parse_args()

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            print(f"Could not read file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        text = args.text

    result = predict_news(text, use_llm_fallback=not args.no_llm, use_factcheck=not args.no_factcheck)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
