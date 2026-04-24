from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


LABELS = ["billing", "technical_issue", "feature_request", "complaint", "other"]


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("section3/data"))
    parser.add_argument("--model-dir", type=Path, default=Path("section3/model"))
    args = parser.parse_args()

    rows = load_jsonl(args.data_dir / "eval.jsonl")
    texts = [row["text"] for row in rows]
    y_true = [row["label"] for row in rows]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=128,
        device=-1,
    )

    predictions = clf(texts, batch_size=16)
    y_pred = [pred["label"] for pred in predictions]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "per_class_f1": {
            label: f1_score(
                [1 if item == label else 0 for item in y_true],
                [1 if item == label else 0 for item in y_pred],
            )
            for label in LABELS
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=LABELS, output_dict=True),
    }

    with open(args.model_dir / "eval_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
