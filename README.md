# Spend-Sense AI: Autonomous Document Intelligence

An end-to-end AI pipeline that transforms unstructured physical receipts into actionable financial insights. Using a hybrid architecture of Computer Vision, Classical ML, and Generative AI (RAG).

## 🚀 Key Features

- **Autonomous Ingestion:** Real-time directory monitoring using Python Watchdog.
- **Hybrid AI Engine:** Combines OCR (Tesseract/EasyOCR) with Scikit-learn for intelligent expense categorization.
- **Contextual Querying:** Retrieval-Augmented Generation (RAG) using FAISS/ChromaDB to answer natural language questions about spending.
- **Enterprise-Ready:** Fully containerized with Docker, structured for high-performance logging and configuration.

## 🛠️ Tech Stack

- **Languages:** Python 3.10+
- **AI/ML:** PyTorch, Scikit-learn, LangChain
- **Vector Store:** FAISS / ChromaDB
- **Automation:** Watchdog, Loguru
- **Infrastructure:** Docker, YAML-based Configuration

## 📂 Project Structure

- `src/`: Core logic (Ingestion, Extraction, LLM Chains)
- `data/`: Multi-stage data lake (Raw, Processed, Vector)
- `TECHNICAL.md`: Detailed architecture and system design.

## ⚙️ Quick Start

1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Add your API key to `.env`.
4. Run the engine: `python main.py`
