from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "distilbert-base-uncased"
LABELS = ["billing", "technical_issue", "feature_request", "complaint", "other"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            rows.append(json.loads(line))
    return rows


def build_dataset(rows: List[dict]) -> Dataset:
    return Dataset.from_list(
        [{"text": row["text"], "label": LABEL_TO_ID[row["label"]]} for row in rows]
    )


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("section3/data"))
    parser.add_argument("--model-dir", type=Path, default=Path("section3/model"))
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    train_rows = load_jsonl(args.data_dir / "train.jsonl")
    eval_rows = load_jsonl(args.data_dir / "eval.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = build_dataset(train_rows)
    eval_ds = build_dataset(eval_rows)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    training_args = TrainingArguments(
        output_dir=str(args.model_dir / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        seed=7,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.model_dir))
    tokenizer.save_pretrained(str(args.model_dir))

    with open(args.model_dir / "train_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
