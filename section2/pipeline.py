from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import faiss
import numpy as np
from google import genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from ingest import EMBEDDING_MODEL_NAME, tokenize_for_bm25

# Load environment variables from the root .env file
load_dotenv(Path(__file__).parent.parent / ".env")


@dataclass
class RetrievedChunk:
    idx: int
    dense_score: float
    sparse_score: float
    hybrid_score: float
    metadata: Dict[str, object]


class LegalRAGPipeline:
    def __init__(self, index_dir: Path, use_gemini: bool = True) -> None:
        self.index_dir = index_dir
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index = faiss.read_index(str(index_dir / "chunks.faiss"))
        self.embeddings = np.load(index_dir / "embeddings.npy")
        with open(index_dir / "chunks.json", "r", encoding="utf-8") as fp:
            self.chunks: List[Dict[str, object]] = json.load(fp)
        with open(index_dir / "bm25.pkl", "rb") as fp:
            self.bm25 = pickle.load(fp)
        self.gemini_model_name = os.getenv("GEMINI_MODEL", "gemma-4-26b-a4b-it")
        self.gemini_enabled = use_gemini and bool(os.getenv("GEMINI_API_KEY"))
        self.gemini_client = genai.Client() if self.gemini_enabled else None

    def _dense_search(self, question: str, top_k: int = 12) -> Dict[int, float]:
        query = self.model.encode([question], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, top_k)
        return {int(idx): float(score) for idx, score in zip(indices[0], scores[0]) if idx != -1}

    def _sparse_search(self, question: str, top_k: int = 12) -> Dict[int, float]:
        scores = self.bm25.get_scores(tokenize_for_bm25(question))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {int(idx): float(scores[idx]) for idx in top_indices}

    @staticmethod
    def _normalize_scores(score_map: Dict[int, float]) -> Dict[int, float]:
        if not score_map:
            return {}
        values = np.array(list(score_map.values()), dtype=float)
        min_v, max_v = float(values.min()), float(values.max())
        if math.isclose(min_v, max_v):
            return {idx: 1.0 for idx in score_map}
        return {idx: (score - min_v) / (max_v - min_v) for idx, score in score_map.items()}

    def _hybrid_candidates(self, question: str, top_k: int = 8) -> List[RetrievedChunk]:
        dense = self._dense_search(question)
        sparse = self._sparse_search(question)
        dense_n = self._normalize_scores(dense)
        sparse_n = self._normalize_scores(sparse)

        candidate_ids = set(dense_n) | set(sparse_n)
        candidates: List[RetrievedChunk] = []
        for idx in candidate_ids:
            dense_score = dense_n.get(idx, 0.0)
            sparse_score = sparse_n.get(idx, 0.0)
            hybrid_score = 0.6 * dense_score + 0.4 * sparse_score
            candidates.append(
                RetrievedChunk(
                    idx=idx,
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    hybrid_score=hybrid_score,
                    metadata=self.chunks[idx],
                )
            )

        ranked = sorted(candidates, key=lambda item: item.hybrid_score, reverse=True)
        return self._mmr(question, ranked, top_k=top_k)

    def _mmr(
        self,
        question: str,
        ranked: Sequence[RetrievedChunk],
        top_k: int = 8,
        lambda_param: float = 0.75,
    ) -> List[RetrievedChunk]:
        if not ranked:
            return []
        query_vec = self.model.encode([question], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)
        selected: List[RetrievedChunk] = []
        candidates = list(ranked)

        while candidates and len(selected) < top_k:
            if not selected:
                selected.append(candidates.pop(0))
                continue

            best_idx = None
            best_score = float("-inf")
            for i, candidate in enumerate(candidates):
                candidate_vec = self.embeddings[candidate.idx : candidate.idx + 1].copy()
                faiss.normalize_L2(candidate_vec)
                query_sim = float(np.dot(query_vec[0], candidate_vec[0]))
                max_sim_to_selected = max(
                    float(np.dot(candidate_vec[0], self._normalized_embedding(sel.idx)))
                    for sel in selected
                )
                mmr_score = lambda_param * query_sim - (1 - lambda_param) * max_sim_to_selected
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(candidates.pop(best_idx))
        return selected

    def _normalized_embedding(self, idx: int) -> np.ndarray:
        vec = self.embeddings[idx].astype("float32").copy()
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    @staticmethod
    def _sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _answer_from_context(self, question: str, retrieved: Sequence[RetrievedChunk]) -> str:
        question_terms = set(tokenize_for_bm25(question))
        best_sentence = None
        best_score = -1.0
        for item in retrieved:
            for sentence in self._sentences(str(item.metadata["text"])):
                sentence_terms = set(tokenize_for_bm25(sentence))
                lexical_overlap = len(question_terms & sentence_terms)
                score = lexical_overlap + item.hybrid_score
                if score > best_score:
                    best_score = score
                    best_sentence = sentence

        if not best_sentence:
            return "I cannot answer confidently from the retrieved documents."
        return best_sentence

    def _gemini_answer(self, question: str, retrieved: Sequence[RetrievedChunk]) -> str:
        if not self.gemini_enabled or not retrieved:
            return self._answer_from_context(question, retrieved)

        context_blocks = []
        for item in retrieved:
            context_blocks.append(
                (
                    f"Document: {item.metadata['document']}\n"
                    f"Page: {item.metadata['page']}\n"
                    f"Chunk:\n{item.metadata['text']}"
                )
            )

        prompt = (
            "You are a legal document question-answering assistant.\n"
            "Answer only from the provided context.\n"
            "If the context is insufficient, reply exactly with: "
            "'I cannot answer confidently from the retrieved documents.'\n"
            "Keep the answer concise and factual.\n\n"
            f"Question: {question}\n\n"
            "Context:\n"
            + "\n\n---\n\n".join(context_blocks)
        )

        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=prompt,
            )
            text = (response.text or "").strip()
            return text or self._answer_from_context(question, retrieved)
        except Exception:
            return self._answer_from_context(question, retrieved)

    def _confidence(self, retrieved: Sequence[RetrievedChunk], answer: str, question: str) -> float:
        if not retrieved or answer.startswith("I cannot answer confidently"):
            return 0.0
        top = retrieved[0]
        answer_terms = set(tokenize_for_bm25(answer))
        question_terms = set(tokenize_for_bm25(question))
        overlap = len(answer_terms & question_terms) / max(1, len(question_terms))
        confidence = 0.5 * top.hybrid_score + 0.3 * top.dense_score + 0.2 * overlap
        return round(float(max(0.0, min(1.0, confidence))), 3)

    def query(self, question: str) -> Dict[str, object]:
        retrieved = self._hybrid_candidates(question, top_k=3)
        answer = self._gemini_answer(question, retrieved)
        confidence = self._confidence(retrieved, answer, question)

        if confidence < 0.35:
            answer = "I cannot answer confidently from the retrieved documents."

        sources = [
            {
                "document": item.metadata["document"],
                "page": item.metadata["page"],
                "chunk": item.metadata["text"],
            }
            for item in retrieved
        ]
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, default=Path("section2/data/index"))
    parser.add_argument("--question", required=True)
    parser.add_argument("--no-gemini", action="store_true")
    args = parser.parse_args()

    pipeline = LegalRAGPipeline(args.index_dir, use_gemini=not args.no_gemini)
    result = pipeline.query(args.question)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
