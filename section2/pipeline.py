from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from google import genai
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from ingest import EMBEDDING_MODEL_NAME

# Load environment variables from the root .env file
load_dotenv(Path(__file__).parent.parent / ".env")


@dataclass
class RetrievedChunk:
    chunk_id: str
    dense_score: float
    sparse_score: float
    hybrid_score: float
    document: Document


def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


class LegalRAGPipeline:
    def __init__(self, index_dir: Path, use_gemini: bool = True) -> None:
        self.index_dir = index_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vectorstore = FAISS.load_local(
            str(index_dir / "faiss_index"),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        with open(index_dir / "bm25.pkl", "rb") as fp:
            self.bm25 = pickle.load(fp)
        self.bm25.k = 12
        self.dense_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 12, "fetch_k": 20},
        )
        self.gemini_model_name = os.getenv("GEMINI_MODEL", "gemma-4-26b-a4b-it")
        self.gemini_enabled = use_gemini and bool(os.getenv("GEMINI_API_KEY"))
        self.gemini_client = genai.Client() if self.gemini_enabled else None

    @staticmethod
    def _doc_key(doc: Document) -> str:
        chunk_id = str(doc.metadata.get("chunk_id", ""))
        if chunk_id:
            return chunk_id
        return f"{doc.metadata.get('document', '')}:{doc.metadata.get('page', '')}:{hash(doc.page_content)}"

    def _hybrid_candidates(self, question: str, top_k: int = 8) -> List[RetrievedChunk]:
        dense_docs = self.dense_retriever.invoke(question)
        sparse_docs = self.bm25.invoke(question)

        dense_ranks: Dict[str, Tuple[float, Document]] = {}
        sparse_ranks: Dict[str, Tuple[float, Document]] = {}

        for rank, doc in enumerate(dense_docs):
            dense_ranks[self._doc_key(doc)] = (1.0 / (rank + 1), doc)

        for rank, doc in enumerate(sparse_docs):
            sparse_ranks[self._doc_key(doc)] = (1.0 / (rank + 1), doc)

        combined_keys = set(dense_ranks) | set(sparse_ranks)
        candidates: List[RetrievedChunk] = []
        for key in combined_keys:
            dense_score, dense_doc = dense_ranks.get(key, (0.0, None))
            sparse_score, sparse_doc = sparse_ranks.get(key, (0.0, None))
            doc = dense_doc or sparse_doc
            hybrid_score = 0.6 * dense_score + 0.4 * sparse_score
            candidates.append(
                RetrievedChunk(
                    chunk_id=key,
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    hybrid_score=hybrid_score,
                    document=doc,
                )
            )

        candidates.sort(key=lambda item: item.hybrid_score, reverse=True)
        return candidates[:top_k]

    @staticmethod
    def _sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _answer_from_context(self, question: str, retrieved: Sequence[RetrievedChunk]) -> str:
        question_terms = set(tokenize_for_bm25(question))
        best_sentence = None
        best_score = -1.0
        for item in retrieved:
            for sentence in self._sentences(item.document.page_content):
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
                    f"Document: {item.document.metadata['document']}\n"
                    f"Page: {item.document.metadata['page']}\n"
                    f"Chunk:\n{item.document.page_content}"
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
                "document": item.document.metadata["document"],
                "page": item.document.metadata["page"],
                "chunk": item.document.page_content,
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
