# Spend-Sense AI

Automated expense intelligence: drop a receipt image → OCR → classify → embed → ask questions in plain English.
Runs fully locally — no cloud API keys required.

---

## Architecture

```
data/raw/  (drop receipt images here)
    │
    ▼
[ Watchdog Watcher ]          ← src/ingestion/watcher.py
    │
    ▼
[ OCR Engine ]                ← src/ingestion/ocr_engine.py
  EasyOCR | Tesseract
    │  raw_text + heuristic metadata
    ▼
[ RF Classifier ]             ← src/models/classifier.py
  TF-IDF (1,2)-gram → Random Forest
    │  category + confidence
    ▼
[ FAISS Vector Store ]        ← src/vector_store/embeddings.py
  all-MiniLM-L6-v2 embeddings
    │
    ▼
[ RAG Chain + Ollama ]        ← src/vector_store/rag_chain.py
  llama3.1:8b (local, no API key)
  "How much did I spend on coffee in July?"
```

---

## Prerequisites

### 1. Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Ollama (local LLM server)

Download and install from [https://ollama.com](https://ollama.com), then pull the model:

```bash
ollama pull llama3.1:8b
```

> **Hardware note:** `llama3.1:8b` (Q4_K_M) uses ~4.7 GB VRAM + ~0.5 GB KV cache.
> Verified to fit on an **RTX 4060 8 GB** with ~2.5 GB headroom.
> Ollama starts automatically after installation; verify with `ollama list`.

### 3. (Optional) Tesseract

Only needed if you set `ocr.engine: tesseract` in `config.yaml`.
Download the Windows installer from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki).
Default engine is EasyOCR, which requires no extra installation.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull the LLM (one-time, ~5 GB download)
ollama pull llama3.1:8b

# 3. Train the expense classifier on bootstrap data
python main.py train

# 4a. Start the directory watcher (auto-processes any image dropped in data/raw/)
python main.py watch

# 4b. Or ingest a single receipt manually
python main.py ingest path/to/receipt.jpg

# 5. Ask natural-language questions about your expenses
python main.py ask "How much did I spend on coffee in July?"
python main.py ask "What was my total transport spend last month?"
python main.py ask "Show me all receipts over $50"
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `python main.py train` | (Re)train the RF classifier on synthetic bootstrap data |
| `python main.py watch` | Start the directory watcher; blocks until Ctrl+C |
| `python main.py ingest <path>` | Manually process a single receipt image |
| `python main.py ask "<question>"` | RAG-powered expense query via local Ollama LLM |
| `python main.py --config alt.yaml <cmd>` | Use a non-default config file |

---

## Configuration

All tuneable parameters live in `config.yaml` — no code changes needed to switch models or backends.

| Key | Default | Notes |
|-----|---------|-------|
| `ocr.engine` | `easyocr` | `easyocr` or `tesseract` |
| `embedding.model` | `all-MiniLM-L6-v2` | Any HuggingFace sentence-transformer |
| `llm.model` | `llama3.1:8b` | Any model pulled via `ollama pull <name>` |
| `llm.base_url` | `http://localhost:11434` | Ollama server address |
| `llm.num_ctx` | `4096` | KV cache window — do not exceed `8192` on 8 GB VRAM |
| `classifier.n_estimators` | `100` | Random Forest trees |

### Swapping the LLM

```yaml
# config.yaml
llm:
  model: mistral:7b     # lighter — Q4_K_M ~4.1 GB VRAM
  # model: gemma2:9b    # stronger reasoning — ~5.5 GB VRAM
  # model: phi4:14b     # best quality — ~9 GB, will exceed 8 GB VRAM
```

---

## Docker

```bash
docker build -t spend-sense-ai .

# Watch mode — mount data/ so the index persists across restarts
docker run -it \
  --gpus all \
  -v $(pwd)/data:/app/data \
  --network host \
  spend-sense-ai watch
```

> `--network host` lets the container reach Ollama running on the host at `localhost:11434`.
> No API keys required.

---

## Azure Mapping

### Azure Data Factory (ADF) — Ingestion & Orchestration

| Spend-Sense Component | ADF Equivalent |
|-----------------------|----------------|
| `ReceiptWatcher` (watchdog) | **Blob Storage Event Trigger** — fires when a file lands in `receipts-raw` container |
| `OCREngine` | **ADF Data Flow** activity calling **Azure AI Document Intelligence** (Form Recognizer) |
| `SpendSensePipeline.process_receipt()` | **ADF Pipeline** with sequential activities: OCR → Classify → Index |
| `data/processed/*.json` | **Azure Data Lake Storage Gen2** — write transformed JSON to `receipts-processed/` |
| `config.yaml` | **ADF Linked Services** + **Azure Key Vault** for non-secret config |

**ADF Pipeline sketch:**

```
Blob Trigger (data/raw/*.jpg)
  → Activity: Call Azure Function (OCR via Document Intelligence)
  → Activity: Call Azure Function (Classify via RF model)
  → Activity: Call Azure Function (Embed + FAISS upsert)
  → Activity: Copy to ADLS Gen2 (archive JSON)
```

### Azure Kubernetes Service (AKS) — Runtime & Scaling

| Spend-Sense Component | AKS Equivalent |
|-----------------------|----------------|
| `Dockerfile` | **Container Image** pushed to **Azure Container Registry (ACR)** |
| `python main.py watch` | **Deployment** with 1 replica — single watcher pod |
| `python main.py ask` | **Deployment** with N replicas behind a **ClusterIP Service** — stateless query pods |
| Ollama LLM server | **Separate Deployment** with GPU node pool (Standard_NC4as_T4_v3 or similar) |
| FAISS index (`data/vector_store/`) | **Azure Files PersistentVolumeClaim** mounted at `/app/data` — shared across pods |
| `data/models/classifier.pkl` | Same PVC or baked into the image at build time |
| Loguru logs | **Azure Monitor / Container Insights** — stdout collected automatically |

**Kubernetes manifests sketch:**

```yaml
# watcher-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spend-sense-watcher
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: watcher
          image: myacr.azurecr.io/spend-sense-ai:latest
          args: ["watch"]
          env:
            - name: OLLAMA_BASE_URL
              value: "http://ollama-service:11434"
          volumeMounts:
            - name: data-pvc
              mountPath: /app/data
      volumes:
        - name: data-pvc
          persistentVolumeClaim:
            claimName: spend-sense-data
---
# ollama-deployment.yaml  (GPU node pool)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama
spec:
  replicas: 1
  template:
    spec:
      nodeSelector:
        accelerator: nvidia
      containers:
        - name: ollama
          image: ollama/ollama:latest
          ports:
            - containerPort: 11434
          resources:
            limits:
              nvidia.com/gpu: 1
```

### End-to-End Azure Flow

```
Azure Blob Storage (raw receipts)
        │  Event Grid trigger
        ▼
Azure Data Factory Pipeline
        │  OCR via Document Intelligence
        │  Classify via Azure Function
        │  Embed + FAISS upsert via Azure Function
        ▼
Azure Data Lake Gen2 (processed JSON)
        │
        ▼
AKS — spend-sense-query Pods
  User → REST API → RAG chain → Ollama (GPU pod) → Answer
```

---

## Project Structure

```
.
├── config.yaml               # All tuneable parameters
├── Dockerfile
├── main.py                   # Click CLI entry point
├── requirements.txt
├── TECHNICAL.md              # Implementation details & design decisions
├── data/
│   ├── raw/                  # Drop receipt images here
│   ├── processed/            # JSON metadata archives (one per receipt)
│   ├── vector_store/         # FAISS index (auto-persisted)
│   └── models/               # classifier.pkl (joblib)
├── logs/                     # Loguru rotating logs
└── src/
    ├── pipeline.py           # Orchestrator — owns all components
    ├── ingestion/
    │   ├── watcher.py        # watchdog FileSystemEventHandler
    │   └── ocr_engine.py     # Strategy: EasyOCRBackend | TesseractBackend
    ├── models/
    │   └── classifier.py     # TF-IDF + Random Forest + joblib persistence
    └── vector_store/
        ├── embeddings.py     # FAISS + LangChain + HuggingFace embeddings
        └── rag_chain.py      # RetrievalQA + ChatOllama
```
