# Section 2 Design: Production-Grade Legal RAG

## Problem Shape

This corpus is legal, not general web text. Users ask clause-level questions that usually hinge on exact wording, entity names, thresholds, exceptions, and dates. That changes the RAG design in three ways:

1. Retrieval quality matters more than generation fluency.
2. Page-level provenance is mandatory.
3. The system should prefer refusal over speculative synthesis.

## Chunking Strategy

I use page-aware, overlapping text chunks built from extracted PDF text.

- Target chunk size: roughly 900-1,200 characters.
- Overlap: 180-220 characters.
- Metadata per chunk: document name, page number, chunk id, character offsets.

Why this strategy:

- Legal clauses are often longer than a sentence but shorter than a full page. Fixed-size chunks with overlap preserve nearby qualifiers like carve-outs, notice windows, and monetary caps.
- Page-aware chunking makes citations reliable. If a user asks about a limitation of liability clause, returning the page where that clause appears is more valuable than returning only a file name.
- I did not chunk by raw paragraph boundaries alone because PDF extraction quality is inconsistent. A pure paragraph strategy is brittle across vendor-generated contracts, scanned documents, and broken line wraps.

Trade-off:

- Smaller chunks improve pinpoint retrieval but risk splitting clause conditions across boundaries.
- Larger chunks preserve context but dilute embedding relevance and make BM25 noisier.

The chosen middle ground favors clause-level retrieval while keeping enough local context for grounded answering.

## Embedding Model Choice

I chose `sentence-transformers/all-MiniLM-L6-v2`.

Why:

- Strong quality-to-latency ratio on CPU.
- Small enough for local execution without GPU.
- Widely supported and stable across platforms.

Why not a larger embedding model:

- The corpus is only 500+ documents in the baseline scenario, so retrieval quality gains from a much larger encoder are unlikely to justify slower ingestion and query latency.
- The system also uses BM25, which helps recover exact legal terms, party names, and currency strings that dense retrieval can miss.

## Vector Store Choice

I chose FAISS.

Why FAISS over Chroma or Pinecone for this submission:

- It is local, fast, portable, and easy to serialize.
- It keeps the project self-contained and device-agnostic.
- For a few thousand to tens of thousands of chunks, FAISS on CPU is more than enough.

Why not Pinecone:

- Managed infrastructure is attractive in production, but this task asked for runnable code without assuming external services.

Why not Chroma:

- Chroma is also reasonable, but FAISS is simpler here because I only need high-speed vector search plus my own metadata store.

## Retrieval Strategy

I use hybrid retrieval:

1. Dense retrieval with FAISS over MiniLM embeddings.
2. Sparse retrieval with BM25 over tokenized chunk text.
3. Score normalization and weighted fusion.
4. MMR re-ranking for diversity in the final shortlist.

Why hybrid instead of naive top-k dense search:

- Legal questions mix semantic intent with exact-match requirements. Entity names like "Vendor X", clause names like "limitation of liability", and thresholds like "₹1 crore" are often better captured by BM25.
- Dense retrieval helps when users phrase the question differently from the source text.
- MMR reduces the chance that the top 3 results are three near-duplicate chunks from the same page while another relevant clause is ignored.

## Hallucination Mitigation

I implemented confidence scoring plus refusal.

Mechanism:

- Confidence combines normalized dense score, normalized BM25 score, and overlap between the answer sentence and retrieved evidence.
- If the best evidence is weak or the top retrieved set is inconsistent, the pipeline returns a refusal such as "I cannot answer confidently from the retrieved documents."

Why this choice:

- In legal workflows, a low-confidence refusal is safer than a smooth answer with a wrong citation.
- Confidence is simple, inspectable, and can be thresholded in production.

I deliberately did not rely on prompt-only instructions like "do not hallucinate." Those reduce but do not eliminate unsupported answers.

## Generation Strategy

The current implementation supports two generation modes:

1. **Grounded LLM Synthesis**: If a `GEMINI_API_KEY` is provided in the `.env` file, the pipeline uses Gemini to synthesize a concise, factual answer grounded strictly in the retrieved context.
2. **Extractive Answering**: If Gemini is disabled or no key is present, the system falls back to selecting the most relevant sentences from the top retrieved chunks based on lexical overlap and hybrid scores.

Why:

- The task explicitly says hallucinated answers are unacceptable.
- Using a grounded prompt (enforced via the `_gemini_answer` method) ensures the model only answers from provided snippets.
- Extractive fallback ensures the system is always functional even without external API access.

## Evaluation

The evaluation harness measures Precision@3 on 10 manually authored question-answer pairs.

Why Precision@3:

- It directly measures whether the retrieval layer surfaced the correct evidence high enough for a safe answer.
- It is more useful than only measuring final answer correctness when debugging chunking and retrieval.

Limitations:

- Precision@3 does not measure answer wording quality.
- Ten questions are enough for a submission harness, not for full production validation.

## What Changes at 50,000 Documents

At 50,000 documents, the bottlenecks shift.

1. Ingestion throughput becomes meaningful.
   Use batch embedding jobs, document queues, and incremental indexing instead of rebuilding everything.

2. FAISS exact search may become slower and memory-heavier.
   Move from a flat index to IVF or HNSW, and tune recall/latency explicitly.

3. BM25 over all chunks becomes more expensive.
   Use an engine built for sparse retrieval at scale, such as Elasticsearch or OpenSearch.

4. Metadata filtering becomes necessary.
   Add structured filters for vendor, contract type, date, and document family before vector search.

5. Evaluation needs to widen.
   Maintain a benchmark set by contract type and query class, then track Precision@k, citation accuracy, refusal rate, and answer-supportedness over time.

My production architecture at 50,000 documents would likely be:

- Object storage for PDFs
- Asynchronous ingestion workers
- OpenSearch or Elasticsearch for sparse retrieval + filters
- A managed or distributed vector index for dense retrieval
- A re-ranker for the top 20-50 candidates
- Strict answer grounding with citation validation before response delivery
