from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from pipeline import LegalRAGPipeline


DEFAULT_QA_PAIRS: List[dict] = [
    {
        "question": "Which document is the annual return for 2026 Q2?",
        "expected_document": "Annual Return 2026 Q2.pdf",
        "expected_page": 1,
    },
    {
        "question": "Which PDF contains the quarterly annual return filing for 2026 Q2?",
        "expected_document": "Annual Return 2026 Q2.pdf",
        "expected_page": 1,
    },
    {
        "question": "Which document is the annual reports file?",
        "expected_document": "Annual Reports.pdf",
        "expected_page": 1,
    },
    {
        "question": "Which PDF should be searched for annual report disclosures and statements?",
        "expected_document": "Annual Reports.pdf",
        "expected_page": 1,
    },
    {
        "question": "Which document covers SEBI transaction rules from 2004?",
        "expected_document": "SEBI transaction rules 2004.pdf",
        "expected_page": 1,
    },
    {
        "question": "If I need the 2004 SEBI transaction rules, which PDF should I open?",
        "expected_document": "SEBI transaction rules 2004.pdf",
        "expected_page": 1,
    },
    {
        "question": "Which file contains the Securities Laws Amendment Act of 2014?",
        "expected_document": "Securities  Laws ,Amendment ACT 2014.pdf",
        "expected_page": 1,
    },
    {
        "question": "What document should I use for the 2014 securities laws amendment act?",
        "expected_document": "Securities  Laws ,Amendment ACT 2014.pdf",
        "expected_page": 1,
    },
    {
        "question": "Which PDF is related to ROC Bangalore?",
        "expected_document": "ROC Bangalore.pdf",
        "expected_page": 1,
    },
    {
        "question": "If I need the ROC Bangalore document, which file should be retrieved?",
        "expected_document": "ROC Bangalore.pdf",
        "expected_page": 1,
    },
]


def precision_at_3(index_dir: Path, qa_pairs: List[dict]) -> dict:
    pipeline = LegalRAGPipeline(index_dir)
    hits = 0
    details = []

    for item in qa_pairs:
        result = pipeline.query(item["question"])
        matched = any(
            source["document"] == item["expected_document"]
            and int(source["page"]) == int(item["expected_page"])
            for source in result["sources"][:3]
        )
        hits += int(matched)
        details.append(
            {
                "question": item["question"],
                "matched": matched,
                "expected_document": item["expected_document"],
                "expected_page": item["expected_page"],
                "retrieved": [
                    {
                        "document": source["document"],
                        "page": source["page"],
                    }
                    for source in result["sources"][:3]
                ],
            }
        )

    score = hits / max(1, len(qa_pairs))
    return {"precision_at_3": round(score, 3), "hits": hits, "total": len(qa_pairs), "details": details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, default=Path("section2/data/index"))
    args = parser.parse_args()

    result = precision_at_3(args.index_dir, DEFAULT_QA_PAIRS)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
