# Multimodal_Ensemble_Retriever_Enterprise.py
# ============================================================================
# MODULE 5: ENTERPRISE MULTIMODAL RETRIEVER
# ============================================================================
#
# PURPOSE:
#   Query all 3 vector databases (text, tables, images)
#   Return top results per modality and merge intelligently
#   Optional cross-encoder reranking for final ranking
#   Send ranked context to LLM for answer generation
#   Log everything for audit, debugging, and cost tracking
#
# WHAT IT DOES:
#   1) Queries all vector databases (text, table, image) in parallel
#   2) Retrieves top-K results per modality
#   3) Deduplicates results (deterministic fingerprinting)
#   4) Optional: Reranks using cross-encoder (improves answer quality)
#   5) Selects final context and sends to LLM (gpt-4o)
#   6) Generates structured answer with traceability
#   7) Saves artifacts: answer.txt, context_blocks.csv, query_timeline.json
#
# INPUT REQUIRED:
#   - vector_db_text_chroma/ (from Module 3)
#   - vector_db_table_chroma/ (from Module 2)
#   - vector_db_image_vlm_chroma/ (from Module 4)
#   - .env with all Azure OpenAI deployments
#
# OUTPUT CREATED:
#   - outputs/run_YYYYMMDD_HHMMSS/
#     ├── answer.txt (final LLM-generated answer)
#     ├── context_blocks.csv (all retrieved chunks with scores)
#     ├── query_timeline.json (per-stage timings + costs)
#     └── manifest.json (metadata about the run)
#
# RUNTIME: 10-30 seconds per query (depends on LLM response time)
#
# PREREQUISITES:
#   Run Modules 1-4 BEFORE this script:
#   1. python Read_File_Docling.py
#   2. python Table_Embeddings_Langchain_vid.py
#   3. python Text_Embeddings_Langchain.py
#   4. python Image_VLM_Embeddings_Langchain_vid.py
#
# ============================================================================
# SETUP INSTRUCTIONS
# ============================================================================
#
# Step 1: ACTIVATE PYTHON ENVIRONMENT

#
# Step 2: INSTALL DEPENDENCIES
#   If you get import errors, uncomment and run:

# !pip install python-dotenv langchain-openai langchain-chroma langchain-core sentence-transformers
# OR for full requirements:
# !pip install -r requirements.txt

# Step 3: ENSURE MODULES 1-4 COMPLETED
#   Run all modules in order:
#   python Read_File_Docling.py
#   python Table_Embeddings_Langchain_vid.py
#   python Text_Embeddings_Langchain.py
#   python Image_VLM_Embeddings_Langchain_vid.py
#
#   Verify: Check that these folders exist:
#   - vector_db_text_chroma/
#   - vector_db_table_chroma/
#   - vector_db_image_vlm_chroma/
#
# Step 4: CONFIGURE PATHS & AZURE CREDENTIALS
#   - Search for "PUT YOUR PATH HERE" below
#   - Create/update .env file with Azure OpenAI credentials
#   - Verify all deployment names match Azure resource
#
# Step 5: RUN THE SCRIPT (interactive query loop)
#   python Multimodal_Ensemble_Retriever_Enterprise.py
#
# Step 6: ENTER QUERIES
#   When prompted, type your question (e.g., "What is the leave policy?")
#   The system will retrieve relevant content and generate an answer
#   Results saved to outputs/run_YYYYMMDD_HHMMSS/ for audit
#
# ============================================================================

from __future__ import annotations

from pathlib import Path
import os
import json
import csv
import re
import hashlib
import logging
import datetime as dt
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ========================
# CONFIG (keep aligned with your existing project)
# ========================

# PUT YOUR PATH HERE: Vector database folders (from Modules 2-4)
# These should match the PERSIST_DIR paths from Table/Text/Image embedding scripts
# From Module 3: Text_Embeddings_Langchain.py
TEXT_DB_DIR = Path("vector_db_text_chroma")
# From Module 2: Table_Embeddings_Langchain_vid.py
TABLE_DB_DIR = Path("vector_db_table_chroma")
# From Module 4: Image_VLM_Embeddings_Langchain_vid.py
VLM_DB_DIR = Path("vector_db_image_vlm_chroma")

TEXT_COLLECTION = "text_chunks_v1"
TABLE_COLLECTION = "table_chunks_v1"
VLM_COLLECTION = "image_vlm_chunks_v1"

# Retrieval parameters
# How many results to retrieve per modality (text, table, image)
TOP_K_PER_MODALITY = 4
FINAL_CONTEXT_K = 7     # Final number of chunks to send to LLM

# PUT YOUR PATH HERE: Where to save query outputs and results
OUTPUT_BASE_DIR = Path("outputs")

# Deduplication and reranking
ENABLE_DEDUPE = True    # Remove duplicate/near-duplicate results

# Reranking is OFF by default (so you don't change baseline behavior unless you choose to)
ENABLE_RERANK = True    # Optional: Use cross-encoder for more intelligent ranking
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder model
RERANK_TOP_K = FINAL_CONTEXT_K
# ========================


# ========================
# LOGGING
# ========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOG = logging.getLogger("multimodal_rag_enterprise")
# ========================


# ========================
# COST PLACEHOLDERS (no assumptions)
# Fill these values with your enterprise rates (USD).
# If left as None, we will log "not configured" and continue.
# ========================
@dataclass
class CostConfig:
    chat_cost_per_1k_input: Optional[float] = None
    chat_cost_per_1k_output: Optional[float] = None
    embed_cost_per_1k_input: Optional[float] = None
    vlm_cost_per_1k_input: Optional[float] = None
    vlm_cost_per_1k_output: Optional[float] = None


COST_CFG = CostConfig(
    chat_cost_per_1k_input=0.001,
    chat_cost_per_1k_output=0.001,
    embed_cost_per_1k_input=0.0001,
    vlm_cost_per_1k_input=0.005,
    vlm_cost_per_1k_output=0.005,
)


def estimate_cost(cfg: CostConfig, stage: str, input_tokens: Optional[int], output_tokens: Optional[int]) -> Dict[str, Any]:
    """
    Cost estimation WITHOUT assumptions:
    - If cfg values are not set or token counts not available, cost_usd remains None with a note.
    """
    out: Dict[str, Any] = {
        "stage": stage,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": None,
        "note": None,
        "rates_used": None,
    }

    if input_tokens is None and output_tokens is None:
        out["note"] = "Token counts not available from this call path."
        return out

    if stage == "chat":
        if cfg.chat_cost_per_1k_input is None or cfg.chat_cost_per_1k_output is None:
            out["note"] = "Chat costs not configured."
            return out
        out["rates_used"] = {
            "chat_cost_per_1k_input": cfg.chat_cost_per_1k_input,
            "chat_cost_per_1k_output": cfg.chat_cost_per_1k_output,
        }
        out["cost_usd"] = (
            (input_tokens or 0) / 1000.0 * cfg.chat_cost_per_1k_input
            + (output_tokens or 0) / 1000.0 * cfg.chat_cost_per_1k_output
        )
        return out

    if stage == "embeddings":
        if cfg.embed_cost_per_1k_input is None:
            out["note"] = "Embedding costs not configured."
            return out
        out["rates_used"] = {
            "embed_cost_per_1k_input": cfg.embed_cost_per_1k_input}
        out["cost_usd"] = (input_tokens or 0) / 1000.0 * \
            cfg.embed_cost_per_1k_input
        return out

    if stage == "vlm":
        if cfg.vlm_cost_per_1k_input is None or cfg.vlm_cost_per_1k_output is None:
            out["note"] = "VLM costs not configured."
            return out
        out["rates_used"] = {
            "vlm_cost_per_1k_input": cfg.vlm_cost_per_1k_input,
            "vlm_cost_per_1k_output": cfg.vlm_cost_per_1k_output,
        }
        out["cost_usd"] = (
            (input_tokens or 0) / 1000.0 * cfg.vlm_cost_per_1k_input
            + (output_tokens or 0) / 1000.0 * cfg.vlm_cost_per_1k_output
        )
        return out

    out["note"] = f"Unknown stage '{stage}'."
    return out
# ========================


# ========================
# OUTPUT HELPERS
# ========================
def make_run_dir(base_dir: Path = OUTPUT_BASE_DIR) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False,
                    indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    cols = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
# ========================


# ========================
# ENV
# ========================
def require_env() -> None:
    # PUT YOUR PATH HERE: Location of .env file with Azure OpenAI credentials
    load_dotenv(dotenv_path=Path(
        r"C:\Users\ShonrajBallae\OneDrive - Protiviti Member Firm\Documents\learning_and_projects\ZS\zs_ai_participants\notebook\Phase_2\.env"), override=False)
    for k in [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        #"AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_EMBEDDING_MODEL",
        "AZURE_OPENAI_MODEL",
    ]:
        if not os.getenv(k):
            raise RuntimeError(f"Missing env var: {k}")
# ========================


# ========================
# VECTORSTORE HELPERS
# ========================
def load_vectorstore(persist_dir: Path, collection: str, embeddings) -> Chroma:
    return Chroma(
        persist_directory=str(persist_dir),
        collection_name=collection,
        embedding_function=embeddings,
    )


def tag_results_scored(docs_scored: List[Tuple[Document, float]], modality: str) -> List[Document]:
    out: List[Document] = []
    for d, score in docs_scored:
        d.metadata = d.metadata or {}
        d.metadata["modality"] = modality
        d.metadata["retrieval_score"] = float(score)
        out.append(d)
    return out
# ========================


# ========================
# HUMAN-READABLE CHUNK RENDERING
# ========================
def _safe_preview(text: str, n: int = 600) -> str:
    t = (text or "").strip().replace("\r", "")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t[:n] + ("…" if len(t) > n else "")


def render_chunk(doc: Document) -> str:
    md = doc.metadata or {}
    modality = md.get("modality", "unknown")

    # Prefer stable IDs if present in metadata; otherwise show "N/A"
    chunk_id = md.get("chunk_uid") or md.get(
        "vector_id") or md.get("id") or "N/A"
    doc_id = md.get("doc_id", "N/A")

    header = (
        f"chunk_id={chunk_id} | modality={modality} | doc_id={doc_id} | "
        f"score={md.get('retrieval_score')}"
    )

    if modality == "table":
        return header + "\n" + _safe_preview(doc.page_content, 900)

    if modality in {"image_vlm", "image_ocr"}:
        # VLM chunks often store JSON-as-text. If it's JSON, display summary & key_text.
        try:
            payload = json.loads(doc.page_content)
            summary = payload.get("summary", "")
            key_text = payload.get("key_text", [])
            warnings = payload.get("warnings", [])
            lines = [
                header,
                f"summary: {summary}".strip(),
                "key_text: " +
                "; ".join(key_text[:10]) if key_text else "key_text: (none)",
                "warnings: " +
                "; ".join(
                    warnings[:5]) if warnings else "warnings: (none)",
            ]
            return "\n".join(lines)
        except Exception:
            return header + "\n" + _safe_preview(doc.page_content, 900)

    return header + "\n" + _safe_preview(doc.page_content, 900)
# ========================


# ========================
# DEDUPE
# ========================
def normalize_text(t: str) -> str:
    t = (t or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def fingerprint(doc: Document) -> str:
    md = doc.metadata or {}

    # Best-case: stable ids you control
    for key in ("chunk_uid", "vector_id", "id"):
        if md.get(key):
            return f"{key}::{md[key]}"

    # Next: doc+anchor if present
    anchor = md.get("table_id") or md.get("figure_id") or md.get(
        "source_csv") or md.get("source_image")
    if md.get("doc_id") and anchor:
        return f"anchor::{md['doc_id']}::{anchor}"

    # Fallback: normalized text hash (still deterministic)
    h = hashlib.sha256(normalize_text(
        doc.page_content).encode("utf-8")).hexdigest()[:16]
    return f"texthash::{h}"


def dedupe_docs(docs: List[Document]) -> Tuple[List[Document], List[Dict[str, Any]]]:
    seen = set()
    kept: List[Document] = []
    removed: List[Dict[str, Any]] = []

    for d in docs:
        fp = fingerprint(d)
        if fp in seen:
            removed.append({"fingerprint": fp, "metadata": d.metadata})
            continue
        seen.add(fp)
        kept.append(d)

    return kept, removed
# ========================


# ========================
# OPTIONAL RERANKING (dedicated library: sentence-transformers)
# ========================
def rerank_cross_encoder(query: str, docs: List[Document], model_name: str, top_k: int) -> List[Document]:
    try:
        from sentence_transformers import CrossEncoder
    except Exception as e:
        raise RuntimeError(
            "Reranking is enabled, but sentence-transformers is not installed.\n"
            "Install it with: pip install sentence-transformers"
        ) from e

    ce = CrossEncoder(model_name)
    pairs = [(query, d.page_content) for d in docs]
    scores = ce.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    out: List[Document] = []
    for d, s in ranked[:top_k]:
        d.metadata = d.metadata or {}
        d.metadata["rerank_score"] = float(s)
        out.append(d)
    return out
# ========================


# ========================
# RETRIEVAL
# ========================
def ensemble_retrieve(query: str, embeddings) -> List[Document]:
    """
    Retrieves with scores from each modality, merges deterministically by score.
    NOTE: We do NOT silently fall back to score-less retrieval; if scores are unavailable, raise.
    """
    text_vs = load_vectorstore(TEXT_DB_DIR,  TEXT_COLLECTION,  embeddings)
    table_vs = load_vectorstore(TABLE_DB_DIR, TABLE_COLLECTION, embeddings)
    # ocr_vs = load_vectorstore(OCR_DB_DIR,   OCR_COLLECTION,   embeddings)
    vlm_vs = load_vectorstore(VLM_DB_DIR,   VLM_COLLECTION,   embeddings)

    # Chroma returns (Document, distance) in similarity_search_with_score.
    # Smaller distance typically means closer match.
    text_hits = tag_results_scored(text_vs.similarity_search_with_score(
        query, k=TOP_K_PER_MODALITY),  "text")
    table_hits = tag_results_scored(
        table_vs.similarity_search_with_score(query, k=TOP_K_PER_MODALITY), "table")
    # ocr_hits = tag_results_scored(ocr_vs.similarity_search_with_score(
    # query, k=TOP_K_PER_MODALITY),   "image_ocr")
    vlm_hits = tag_results_scored(vlm_vs.similarity_search_with_score(
        query, k=TOP_K_PER_MODALITY),   "image_vlm")

    # all_hits = text_hits + table_hits + ocr_hits + vlm_hits
    all_hits = text_hits + table_hits + vlm_hits

    merged = sorted(all_hits, key=lambda d: d.metadata.get(
        "retrieval_score", 1e9))
    return merged[:FINAL_CONTEXT_K]
# ========================


# ========================
# CONTEXT BUILDER (with stable headers)
# ========================
def _source_hint(meta: Dict[str, Any]) -> str:
    return (
        meta.get("source_docx")
        or meta.get("source_csv")
        or meta.get("source_image")
        or meta.get("source_file")
        or "N/A"
    )


def build_context_block(docs: List[Document]) -> str:
    blocks: List[str] = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        chunk_id = meta.get("chunk_uid") or meta.get(
            "vector_id") or meta.get("id") or "N/A"
        header = (
            f"[{i}] chunk_id={chunk_id} | modality={meta.get('modality')} | "
            f"doc_id={meta.get('doc_id')} | source={_source_hint(meta)} | "
            f"score={meta.get('retrieval_score')} | rerank={meta.get('rerank_score')}"
        )
        blocks.append(header + "\n" + (d.page_content or ""))
    return "\n\n".join(blocks)
# ========================


def main() -> None:
    require_env()

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        #api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_MODEL"],
    )

    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        #api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        deployment_name=os.environ["AZURE_OPENAI_MODEL"],
        temperature=0.2,
    )

    print("Multimodal RAG (Enterprise) ready. Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Ask a question: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        run_dir = make_run_dir(OUTPUT_BASE_DIR)

        t0 = time.perf_counter()
        LOG.info("QUERY: %s", query)

        # ---- Retrieval ----
        t_retr0 = time.perf_counter()
        hits = ensemble_retrieve(query, embeddings)
        t_retr1 = time.perf_counter()

        # ---- Outputs: raw retrieval ----
        retrieval_rows: List[Dict[str, Any]] = []
        for rank, d in enumerate(hits, start=1):
            md = d.metadata or {}
            retrieval_rows.append({
                "rank": rank,
                "modality": md.get("modality"),
                "doc_id": md.get("doc_id"),
                "chunk_uid": md.get("chunk_uid"),
                "vector_id": md.get("vector_id"),
                "table_id": md.get("table_id"),
                "figure_id": md.get("figure_id"),
                "source_docx": md.get("source_docx"),
                "source_csv": md.get("source_csv"),
                "source_image": md.get("source_image"),
                "retrieval_score": md.get("retrieval_score"),
                "preview": (d.page_content or "")[:220].replace("\n", " "),
            })

        write_csv(run_dir / "retrieval_raw.csv", retrieval_rows)
        write_text(run_dir / "retrieval_raw.txt",
                   "\n\n".join(render_chunk(d) for d in hits))

        LOG.info("Retrieved raw=%d in %.3fs. Outputs: %s",
                 len(hits), (t_retr1 - t_retr0), run_dir)

        # ---- Deduplication ----
        removed: List[Dict[str, Any]] = []
        if ENABLE_DEDUPE:
            t_dd0 = time.perf_counter()
            hits, removed = dedupe_docs(hits)
            t_dd1 = time.perf_counter()
            write_json(run_dir / "dedupe_removed.json", removed)
            LOG.info("Deduped kept=%d removed=%d in %.3fs",
                     len(hits), len(removed), (t_dd1 - t_dd0))

        # ---- Optional reranking ----
        if ENABLE_RERANK:
            t_rr0 = time.perf_counter()
            hits = rerank_cross_encoder(
                query=query,
                docs=hits,
                model_name=RERANK_MODEL_NAME,
                top_k=RERANK_TOP_K,
            )
            t_rr1 = time.perf_counter()
            LOG.info("Reranked top_k=%d in %.3fs using %s",
                     RERANK_TOP_K, (t_rr1 - t_rr0), RERANK_MODEL_NAME)

        # ---- Build context + prompt ----
        t_ctx0 = time.perf_counter()
        context = build_context_block(hits)
        t_ctx1 = time.perf_counter()

        write_text(run_dir / "final_context.txt", context)

        prompt = (
            "You are answering using multimodal company knowledge.\n\n"
            "Context (mixed text, tables, OCR, diagrams):\n"
            f"{context}\n\n"
            f"Question: {query}\n\n"
            "Answer clearly and cite sources by [index]."
        )
        write_text(run_dir / "final_prompt.txt", prompt)

        # ---- LLM ----
        t_llm0 = time.perf_counter()
        response = llm.invoke(prompt)
        t_llm1 = time.perf_counter()

        answer = getattr(response, "content", "") or ""
        write_text(run_dir / "answer.txt", answer)

        # ---- Token usage & cost placeholders ----
        # Not all wrappers expose token usage consistently; we do not guess.
        chat_cost = estimate_cost(
            COST_CFG, stage="chat", input_tokens=None, output_tokens=None)
        embed_cost = estimate_cost(
            COST_CFG, stage="embeddings", input_tokens=None, output_tokens=None)
        vlm_cost = estimate_cost(
            COST_CFG, stage="vlm", input_tokens=None, output_tokens=None)

        trace = {
            "query": query,
            "config": {
                "TOP_K_PER_MODALITY": TOP_K_PER_MODALITY,
                "FINAL_CONTEXT_K": FINAL_CONTEXT_K,
                "ENABLE_DEDUPE": ENABLE_DEDUPE,
                "ENABLE_RERANK": ENABLE_RERANK,
                "RERANK_MODEL_NAME": RERANK_MODEL_NAME if ENABLE_RERANK else None,
                "RERANK_TOP_K": RERANK_TOP_K if ENABLE_RERANK else None,
            },
            "counts": {
                "retrieved_raw": len(retrieval_rows),
                "dedupe_removed": len(removed),
                "final_context": len(hits),
            },
            "timings_sec": {
                "retrieval": round(t_retr1 - t_retr0, 4),
                "dedupe": round((t_dd1 - t_dd0), 4) if ENABLE_DEDUPE else None,
                "rerank": round((t_rr1 - t_rr0), 4) if ENABLE_RERANK else None,
                "context_build": round(t_ctx1 - t_ctx0, 4),
                "llm": round(t_llm1 - t_llm0, 4),
                "total": round(time.perf_counter() - t0, 4),
            },
            "cost_placeholders": {
                "chat": chat_cost,
                "embeddings": embed_cost,
                "vlm": vlm_cost,
                "rates_config": asdict(COST_CFG),
            },
            "artifacts": {
                "retrieval_raw_csv": str((run_dir / "retrieval_raw.csv").as_posix()),
                "retrieval_raw_txt": str((run_dir / "retrieval_raw.txt").as_posix()),
                "dedupe_removed_json": str((run_dir / "dedupe_removed.json").as_posix()) if ENABLE_DEDUPE else None,
                "final_context_txt": str((run_dir / "final_context.txt").as_posix()),
                "final_prompt_txt": str((run_dir / "final_prompt.txt").as_posix()),
                "answer_txt": str((run_dir / "answer.txt").as_posix()),
            },
        }
        write_json(run_dir / "trace.json", trace)

        # ---- Console output ----
        print("\n=== ANSWER ===\n")
        print(answer)
        print("\n(Artifacts saved to:", run_dir.as_posix() + " )\n")


if __name__ == "__main__":
    main()
