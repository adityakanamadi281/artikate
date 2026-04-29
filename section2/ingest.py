from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_page_documents(pdf_path: Path) -> List[Document]:
    reader = PdfReader(str(pdf_path))
    page_docs: List[Document] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = normalize_text(text)
        if not cleaned:
            continue
        page_docs.append(
            Document(
                page_content=cleaned,
                metadata={
                    "document": pdf_path.name,
                    "page": page_idx,
                },
            )
        )
    return page_docs


def build_chunk_documents(page_docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100,
        chunk_overlap=220,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "; ", ": ", " "],
    )
    chunks = splitter.split_documents(page_docs)
    chunk_counts = {}

    for chunk in chunks:
        document_name = str(chunk.metadata["document"])
        page = int(chunk.metadata["page"])
        chunk_counts[(document_name, page)] = chunk_counts.get((document_name, page), 0) + 1
        chunk.metadata["chunk_id"] = (
            f"{Path(document_name).stem}-p{page}-c{chunk_counts[(document_name, page)]}"
        )
        start_char = int(chunk.metadata.get("start_index", 0))
        chunk.metadata["start_char"] = start_char
        chunk.metadata["end_char"] = start_char + len(chunk.page_content)

    return chunks


def ingest(pdf_dir: Path, index_dir: Path) -> dict:
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if len(pdf_paths) < 3:
        raise ValueError(
            f"Expected at least 3 PDFs in {pdf_dir}, found {len(pdf_paths)}."
        )

    page_docs: List[Document] = []
    for pdf_path in pdf_paths:
        page_docs.extend(extract_page_documents(pdf_path))

    all_chunks = build_chunk_documents(page_docs)

    if not all_chunks:
        raise ValueError("No text chunks were extracted from the provided PDFs.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    bm25 = BM25Retriever.from_documents(all_chunks)
    bm25.k = 12

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir / "faiss_index"))

    with open(index_dir / "chunks.json", "w", encoding="utf-8") as fp:
        json.dump(
            [
                {
                    "chunk_id": doc.metadata["chunk_id"],
                    "document": doc.metadata["document"],
                    "page": doc.metadata["page"],
                    "text": doc.page_content,
                    "start_char": doc.metadata["start_char"],
                    "end_char": doc.metadata["end_char"],
                }
                for doc in all_chunks
            ],
            fp,
            ensure_ascii=False,
            indent=2,
        )

    with open(index_dir / "bm25.pkl", "wb") as fp:
        pickle.dump(bm25, fp)

    manifest = {
        "pdf_count": len(pdf_paths),
        "chunk_count": len(all_chunks),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "index_dir": str(index_dir),
        "framework": "langchain",
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
