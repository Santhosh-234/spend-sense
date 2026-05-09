# Spend-Sense AI — Technical Documentation

## Overview

Spend-Sense AI is a fully local, automated expense intelligence pipeline. It monitors a directory for receipt images, extracts text via OCR, classifies expenses using a classical ML model, stores semantic embeddings in a vector database, and exposes a natural-language query interface backed by a local LLM via RAG.

No cloud API keys are required at runtime.

---

## Component Breakdown

### 1. Directory Watcher (`src/ingestion/watcher.py`)

**Library:** `watchdog`

`ReceiptWatcher` wraps a `watchdog.Observer` with a custom `FileSystemEventHandler`. On `on_created` events it checks the file extension against a whitelist (`{.jpg, .jpeg, .png, .tiff, .bmp, .webp}`) before invoking the pipeline callback.

Design decisions:
- `recursive=False` on the observer — only top-level drops in `data/raw/` are processed; subdirectories are ignored to prevent accidental recursion.
- `pipeline_callback` is injected at construction time, keeping the watcher decoupled from the pipeline. Any callable accepting a `Path` works.
- Exceptions inside the callback are caught and logged without crashing the watcher loop, so one bad receipt does not kill the process.

---

### 2. OCR Engine (`src/ingestion/ocr_engine.py`)

**Libraries:** `easyocr`, `pytesseract`, `Pillow`

Uses the **Strategy pattern** via an abstract `_OCRBackend` base class with two concrete implementations:

| Backend | Class | Notes |
|---------|-------|-------|
| EasyOCR | `_EasyOCRBackend` | Default. No system install required. GPU optional (`gpu=False`). |
| Tesseract | `_TesseractBackend` | Requires system-level Tesseract binary. Faster on CPU for simple receipts. |

The active backend is selected from `config.yaml` (`ocr.engine`) — switching backends requires no code change.

**`ReceiptParser`** runs a series of regex patterns over the raw OCR text to extract structured metadata:
- `merchant`: first non-numeric, non-trivial line in the top 4 lines
- `date`: matches ISO (`2024-07-15`), slash-delimited (`07/15/24`), and long-form (`July 15, 2024`)
- `total_amount`: scans for keywords (`total`, `amount due`, `grand total`) followed by a dollar figure; falls back to the last dollar amount on the page

All heuristics are best-effort — missing fields are `None`, never a crash.

---

### 3. Expense Classifier (`src/models/classifier.py`)

**Library:** `scikit-learn`, `joblib`

A **scikit-learn Pipeline** of two stages:

```
TfidfVectorizer(ngram_range=(1,2), max_features=8000, sublinear_tf=True)
    ↓
RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1)
```

Key choices:
- **Bigrams** (`ngram_range=(1,2)`) capture merchant-adjacent context ("double shot", "monthly bill") that unigrams miss.
- **`sublinear_tf=True`** applies `log(1 + tf)` dampening, reducing the dominance of high-frequency filler words.
- **`class_weight='balanced'`** compensates for unequal class representation in the bootstrap seed data.
- **`n_jobs=-1`** parallelises tree training across all CPU cores.

**Bootstrap training data:** 29 hand-written synthetic examples across 9 categories (Coffee, Food, Transport, Utilities, Entertainment, Healthcare, Shopping, Travel, Other). This is enough for the model to be usable immediately but the CV F1 (~0.08) is low because 3-fold cross-validation on ~3 examples per class is statistically meaningless. The metric becomes meaningful after feeding real labelled receipts via `clf.train(texts, labels)`.

**Persistence:** `joblib.dump` / `joblib.load` to `data/models/classifier.pkl`. The pipeline auto-trains on synthetic data if no saved model exists.

---

### 4. Vector Store (`src/vector_store/embeddings.py`)

**Libraries:** `faiss-cpu`, `langchain-community`, `langchain-huggingface`, `sentence-transformers`

`ReceiptVectorStore` wraps `langchain_community.vectorstores.FAISS` with:
- **Embeddings model:** `sentence-transformers/all-MiniLM-L6-v2` — 384-dimensional, ~90 MB, fast on CPU, good semantic quality for short English text.
- **Normalised embeddings** (`normalize_embeddings=True`) — enables cosine similarity via dot product, which is what FAISS flat index optimises.

Each receipt is stored as a single `Document` whose `page_content` is a structured string:

```
Merchant: Starbucks
Date: 2024-07-03
Amount: $6.75
Category: Coffee
---
<raw OCR text>
```

The metadata dict (`source_file`, `merchant`, `date`, `total_amount`, `category`, `processed_at`) is stored alongside and surfaced in RAG source citations.

**Persistence:** `FAISS.save_local` / `FAISS.load_local` to `data/vector_store/faiss_index/` on every `add_receipt` call. This is synchronous and safe for single-threaded use.

**Import fix applied:** `langchain.schema.Document` was removed in LangChain 0.3. The correct import is `langchain_core.documents.Document`.

---

### 5. RAG Chain (`src/vector_store/rag_chain.py`)

**Libraries:** `langchain`, `langchain-ollama`

Uses `langchain.chains.RetrievalQA` with `chain_type="stuff"` — all retrieved documents are concatenated into a single prompt context block. Appropriate for receipt-scale documents (each is ~150–300 tokens).

**LLM:** `ChatOllama` pointing at a local Ollama server.

```python
ChatOllama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.1,
    num_predict=1024,
    num_ctx=4096,      # explicit KV cache cap
)
```

**Why `num_ctx=4096`:** Without this, Ollama may default to a large context window (up to 131072 for llama3.1) which would exhaust 8 GB VRAM. At `k=6` retrieved documents of ~150 tokens each, the total in-context token count is ~2200, well within 4096. The cap can be raised to 8192 (~1 GB KV cache) if more retrieval depth is needed.

**Prompt engineering:** A custom `PromptTemplate` instructs the model to cite specific receipts, sum amounts when asked for totals, and refuse to hallucinate. `temperature=0.1` biases toward deterministic, factual answers.

---

### 6. Pipeline Orchestrator (`src/pipeline.py`)

`SpendSensePipeline` owns one instance of each component and wires them in sequence:

```
process_receipt(image_path):
    receipt = ocr.process(image_path)           # raw_text + heuristic metadata
    category, confidence = classifier.predict(receipt['raw_text'])
    receipt['category'] = category
    vector_store.add_receipt(receipt)           # embed + persist FAISS index
    write JSON to data/processed/<stem>.json    # archive for audit trail

ask(question):
    if rag is None:
        rag = ExpenseRAGChain(vector_store.as_retriever(k=6), config)
    return rag.query(question)
```

`ExpenseRAGChain` is **lazily initialised** on the first `ask()` call. This means `python main.py watch` starts without touching Ollama at all — the LLM connection is only made when a query is issued.

---

### 7. CLI (`main.py`)

**Library:** `click`

Four commands exposed as a `click.Group`:

| Command | Action |
|---------|--------|
| `train` | Calls `ExpenseClassifier.train_on_synthetic()` |
| `watch` | Constructs pipeline + watcher, enters blocking watch loop |
| `ingest <path>` | Constructs pipeline, calls `process_receipt` once, prints summary |
| `ask "<question>"` | Constructs pipeline, calls `ask`, prints answer + sources |

The `config` object is stored on the Click context (`ctx.obj`) and shared across all commands. Components are not constructed at CLI group level — they are constructed inside each command, keeping cold-start fast for simple operations like `train`.

---

## Changes Made During Development

### Session 1 — Initial build
All files created from scratch:
- `config.yaml`, `Dockerfile`, `requirements.txt`, `.env.example`, `README.md`
- `src/ingestion/watcher.py`, `src/ingestion/ocr_engine.py`
- `src/models/classifier.py`
- `src/vector_store/embeddings.py`, `src/vector_store/rag_chain.py`
- `src/pipeline.py`, `main.py`

### Session 2 — LLM swap: Anthropic → Ollama
**Motivation:** No Anthropic API key available; preference for fully local inference.

**Model chosen:** `llama3.1:8b`
- Q4_K_M quantization: 4.7 GB VRAM
- 128k native context window (capped to 4096 via `num_ctx`)
- Strong instruction-following and arithmetic — both needed for expense totalling
- Fits RTX 4060 8 GB with ~2.5 GB headroom

**Files changed:**
- `config.yaml` — replaced `llm.model: claude-sonnet-4-6` with `llm.model: llama3.1:8b`; added `llm.base_url` and `llm.num_ctx`
- `src/vector_store/rag_chain.py` — replaced `ChatAnthropic` with `ChatOllama` from `langchain-ollama`; removed `ANTHROPIC_API_KEY` env check
- `requirements.txt` — replaced `langchain-anthropic` with `langchain-ollama`; removed `python-dotenv`
- `.env.example` — removed API key reference
- `main.py` — removed `load_dotenv()` call

### Session 3 — VRAM guard
**Motivation:** RTX 4060 has 8 GB VRAM; without an explicit context cap Ollama can default to llama3.1's full 131072-token window, causing OOM.

**Files changed:**
- `config.yaml` — added `llm.num_ctx: 4096`
- `src/vector_store/rag_chain.py` — passed `num_ctx` to `ChatOllama`

### Session 4 — LangChain import fixes (two separate bugs)

**Bug 1:** `ModuleNotFoundError: No module named 'langchain.schema'`
**Root cause:** LangChain 0.3 removed `langchain.schema`; `Document` moved to `langchain_core.documents`.
**File changed:** `src/vector_store/embeddings.py` — updated import to `from langchain_core.documents import Document`

**Bug 2:** `ModuleNotFoundError: No module named 'langchain.chains'`
**Root cause:** `RetrievalQA` from `langchain.chains` is a legacy abstraction removed in LangChain 0.3. Similarly `langchain.prompts` is gone.
**Fix:** Rewrote `src/vector_store/rag_chain.py` using **LCEL** (LangChain Expression Language), the 0.3+ standard.
The new chain is:
```python
docs   = retriever.invoke(question)           # explicit retrieval
answer = (prompt | llm | StrOutputParser()).invoke({"context": ..., "question": ...})
```
All imports now come from `langchain_core` (`PromptTemplate`, `StrOutputParser`, `Document`) and `langchain_ollama` (`ChatOllama`) — no dependency on the legacy `langchain` top-level package sub-modules.

---

## Known Limitations

| Area | Limitation | Mitigation |
|------|-----------|------------|
| Classifier accuracy | CV F1 ~0.08 on 29 synthetic samples | Feed real labelled receipts via `clf.train(texts, labels)` |
| OCR quality | Low-res or crumpled images produce noisy text | Pre-process images with OpenCV (deskew, contrast stretch) before OCR |
| FAISS at scale | Flat index is O(n·d) per query | Switch to `IndexIVFFlat` with `nlist=100` beyond ~10k receipts |
| Thread safety | `FAISS.add_documents` is not thread-safe | Add `threading.Lock` around writes if moving to async ingestion |
| Watcher reliability | `watchdog` on Windows uses polling for network drives | Use `--watchmedo` with native Win32 events on local drives |
| Date parsing | Heuristic regex; ambiguous formats (07/08 = Jul 8 or Aug 7?) | Normalise to ISO 8601 with `dateparser` library |

---

## Dependency Map

```
main.py
└── src/pipeline.py
    ├── src/ingestion/ocr_engine.py   → easyocr / pytesseract, Pillow
    ├── src/models/classifier.py      → scikit-learn, joblib
    └── src/vector_store/
        ├── embeddings.py             → faiss-cpu, langchain-community,
        │                                langchain-huggingface, sentence-transformers
        └── rag_chain.py              → langchain, langchain-ollama → Ollama server
src/ingestion/watcher.py              → watchdog
config.yaml                           → pyyaml
logging                               → loguru
CLI                                   → click
```
