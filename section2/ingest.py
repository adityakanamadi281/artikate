from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import faiss
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Chunk:
    chunk_id: str
    document: str
    page: int
    text: str
    start_char: int
    end_char: int


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def chunk_page_text(
    document: str,
    page: int,
    text: str,
    chunk_size: int = 1100,
    overlap: int = 220,
) -> Iterable[Chunk]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []

    chunks: List[Chunk] = []
    start = 0
    chunk_num = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        window = cleaned[start:end]

        if end < len(cleaned):
            split_at = max(window.rfind(". "), window.rfind("; "), window.rfind(": "))
            if split_at > int(chunk_size * 0.6):
                end = start + split_at + 1
                window = cleaned[start:end]

        chunk_num += 1
        chunks.append(
            Chunk(
                chunk_id=f"{Path(document).stem}-p{page}-c{chunk_num}",
                document=document,
                page=page,
                text=window.strip(),
                start_char=start,
                end_char=end,
            )
        )

        if end >= len(cleaned):
            break
        start = max(0, end - overlap)

    return chunks


def extract_chunks_from_pdf(pdf_path: Path) -> List[Chunk]:
    reader = PdfReader(str(pdf_path))
    chunks: List[Chunk] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        chunks.extend(chunk_page_text(pdf_path.name, page_idx, text))
    return chunks


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    vectors = vectors.astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def ingest(pdf_dir: Path, index_dir: Path) -> dict:
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if len(pdf_paths) < 3:
        raise ValueError(
            f"Expected at least 3 PDFs in {pdf_dir}, found {len(pdf_paths)}."
        )

    all_chunks: List[Chunk] = []
    for pdf_path in pdf_paths:
        all_chunks.extend(extract_chunks_from_pdf(pdf_path))

    if not all_chunks:
        raise ValueError("No text chunks were extracted from the provided PDFs.")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chunk_texts = [chunk.text for chunk in all_chunks]
    embeddings = model.encode(
        chunk_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    bm25 = BM25Okapi([tokenize_for_bm25(text) for text in chunk_texts])
    faiss_index = build_faiss_index(embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(index_dir / "chunks.faiss"))

    with open(index_dir / "chunks.json", "w", encoding="utf-8") as fp:
        json.dump([asdict(chunk) for chunk in all_chunks], fp, ensure_ascii=False, indent=2)

    with open(index_dir / "bm25.pkl", "wb") as fp:
        pickle.dump(bm25, fp)

    np.save(index_dir / "embeddings.npy", embeddings.astype("float32"))

    manifest = {
        "pdf_count": len(pdf_paths),
        "chunk_count": len(all_chunks),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "index_dir": str(index_dir),
    }
    with open(index_dir / "manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", type=Path, default=Path("section2/data/pdfs"))
    parser.add_argument("--index-dir", type=Path, default=Path("section2/data/index"))
    args = parser.parse_args()

    manifest = ingest(args.pdf_dir, args.index_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
