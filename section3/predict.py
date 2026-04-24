from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


VALID_LABELS = {"billing", "technical_issue", "feature_request", "complaint", "other"}

SAMPLE_TICKETS = [
    "I was charged twice for the same subscription in March.",
    "The export to CSV button does nothing when I click it.",
    "Please add SSO support for our enterprise account.",
    "This support experience has been extremely frustrating.",
    "What is the contact email for procurement paperwork?",
    "My refund still has not appeared on my bank statement.",
    "The Android app crashes when I open notifications.",
    "Can you add dark mode for the dashboard?",
    "Nobody on your team is taking ownership of this issue.",
    "Where can I find your uptime history?",
    "The invoice amount is higher than what sales promised.",
    "Uploads are stuck at 99 percent for large PDFs.",
    "We need a way to bulk archive completed items.",
    "Your support team keeps closing the ticket without fixing it.",
    "Do you offer implementation consulting for new customers?",
    "My card was billed even though I canceled last week.",
    "Login fails after I enter the one-time passcode.",
    "It would help if reports could be scheduled weekly.",
    "I have contacted support four times and still have no resolution.",
    "Can you share your standard DPA template?",
]


def load_classifier(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=128,
        device=-1,
    )


def predict(texts: List[str], model_dir: Path) -> dict:
    clf = load_classifier(model_dir)
    start = time.perf_counter()
    outputs = clf(texts, batch_size=16)
    elapsed_ms = (time.perf_counter() - start) * 1000
    labels = [item["label"] for item in outputs]
    return {
        "predictions": labels,
        "total_latency_ms": round(elapsed_ms, 2),
        "avg_latency_ms_per_ticket": round(elapsed_ms / max(1, len(texts)), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("section3/model"))
    parser.add_argument("--max-latency-ms", type=float, default=500.0)
    args = parser.parse_args()

    result = predict(SAMPLE_TICKETS, args.model_dir)

    invalid = [label for label in result["predictions"] if label not in VALID_LABELS]
    assert not invalid, f"Invalid labels predicted: {invalid}"
    assert (
        result["avg_latency_ms_per_ticket"] <= args.max_latency_ms
    ), f"Average latency exceeded {args.max_latency_ms} ms per ticket: {result['avg_latency_ms_per_ticket']}"

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
