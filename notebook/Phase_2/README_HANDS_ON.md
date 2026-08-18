# Two-Day Enterprise RAG Hands-On Pack

## Delivery sequence

### Day 1 - Retrieval foundation
1. `03_document_ingestion_chunking.ipynb`
2. `04_embeddings_vector_search.ipynb`

### Day 2 - End-to-end RAG and evaluation
3. `05_end_to_end_rag.ipynb`
4. `06_rag_evaluation_governance.ipynb`

## Setup
1. Place this folder inside the same project that contains your `.env`, or copy `.env.example` to `.env` and fill the approved Azure values.
2. Install dependencies:
   `pip install -r requirements_rag.txt`
3. Run notebooks from this pack's root folder so relative paths resolve correctly.

## Design choices
- Azure-hosted chat and embedding models are consumed through the existing `.env` endpoint/key pattern.
- Chroma is used as a local persistent vector database for training so the lab does not require a separate Azure AI Search resource.
- The policy corpus is fully synthetic. No PHI or real member data is used.
- The same corpus intentionally contains plan-specific differences so learners can observe metadata filtering and retrieval failures.

## Folder structure
- `data/healthcare_policies/` - synthetic policy PDFs and manifest
- `data/evaluation/` - evaluation test set
- `artifacts/` - generated chunks, vector stores, evaluation traces
