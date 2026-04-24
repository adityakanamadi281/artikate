## Repository Layout

```text
artikate/
├── README.md
├── DESIGN.md
├── ANSWERS.md
├── requirements.txt
├── .env.example
├── section1/
│   └── diagnose.py
├── section2/
│   ├── ingest.py
│   ├── pipeline.py
│   ├── evaluate.py
│   └── data/
│       ├── pdfs/
│       └── index/
└── section3/
    ├── generate_data.py
    ├── train.py
    ├── evaluate.py
    ├── predict.py
    └── data/
```

```bash
git clone https://github.com/adityakanamadi281/artikate.git
cd artikate
```

Create virtual environment:

```bash
uv venv
.venv\Scripts\activate
```

Install dependencies:

```bash
uv pip install -r requirements.txt
```

Create `.env` and add:

```text
GEMINI_API_KEY=your_api_key_here
```

## Run Guide

### Section 1: Diagnose failing LLM pipeline

This script prints a diagnosis of production issues.

```bash
python section1/diagnose.py
```

### Section 2: Production-Grade RAG Pipeline

Build the Index (Processes the PDFs):

```bash
python section2/ingest.py --pdf-dir section2/data/pdfs --index-dir section2/data/index
```

Run a Query:

```bash
python section2/pipeline.py --index-dir section2/data/index --question "Which document covers SEBI transaction rules from 2004?"
```

Evaluate Retrieval:

```bash
python section2/evaluate.py --index-dir section2/data/index
```

### Section 3: Ticket Classifier (Fine-tuning DistilBERT)

Generate Synthetic Data:

```bash
python section3/generate_data.py --output-dir section3/data
```

Train the Model (This may take a few minutes):

```bash
python section3/train.py --data-dir section3/data --model-dir section3/model
```

Evaluate the Classifier:

```bash
python section3/evaluate.py --data-dir section3/data --model-dir section3/model
```

Run Predictions & Latency Check:

```bash
python section3/predict.py --model-dir section3/model
```

## Portability

The code avoids machine-specific paths, assumes only Python package dependencies, and runs on CPU by default. It includes compatibility fixes for modern `transformers` versions (4.41.0+) regarding training arguments and tokenizer handling. File locations are configurable via CLI arguments and environment variables.
