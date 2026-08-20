# rag_semantic_chunking_faiss_azure_eval_RBAC_prompted.py
# ------------------------------------------------------------
# Clean run: no CLI args. RBAC prompts ONLY on salary questions.
#
# Salary RBAC rules:
# - L1–L4: can see ONLY their own salary (requires identity)
# - L5–L6: can see their own + anyone strictly below them
# - L7 (CEO): can see everything
#
# Notes:
# - For "own salary" enforcement, level alone is insufficient. We ask employee_id
#   only when needed (L1–L4 salary access OR "my salary" requests).
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document as LCDocument
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings


# =========================
# CONFIG
# =========================
CONFIG = {
    "DOCS_DIR": "docs",   # folder with HR policies (.docx)
    # your working salary roster
    "SALARY_CSV": "small_company_20_employees_7_roles_salary.csv",
    "TOP_K": 6,

    # overfetch then filter (prevents “close but unauthorized” salary docs from leaking)
    "OVERFETCH_MULTIPLIER": 6,
    "MAX_CANDIDATES_CAP": 60,

    "TEMPERATURE": 0.0,
    "MAX_TOKENS": 256,
}
# =========================


# -------------------------
# Helpers
# -------------------------
def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def here(rel: str) -> Path:
    return (Path(__file__).resolve().parent / rel).resolve()


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


# -------------------------
# Azure env
# -------------------------
def load_env() -> None:
    load_dotenv(dotenv_path=Path(
        r"C:\Users\shonr\OneDrive - Tekframeworks\Secret_keys\.env"), override=False)
    required = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
    ]
    missing = [k for k in required if not os.getenv(k)]
    require(not missing, f"Missing required .env variables: {missing}")


def make_llm() -> AzureChatOpenAI:
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        temperature=float(CONFIG["TEMPERATURE"]),
        max_tokens=int(CONFIG["MAX_TOKENS"]),
    )
    # fail fast
    resp = llm.invoke("Reply with exactly: OK")
    txt = getattr(resp, "content", "")
    require(isinstance(txt, str) and "OK" in txt, "Azure LLM probe failed.")
    return llm


def make_embeddings() -> AzureOpenAIEmbeddings:
    emb = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    v = emb.embed_query("ping")
    require(isinstance(v, list) and len(v) > 0,
            "Azure embeddings probe failed.")
    return emb


# -------------------------
# Load policy docs
# -------------------------
def load_policy_docs(docs_dir: Path) -> List[LCDocument]:
    require(docs_dir.exists() and docs_dir.is_dir(),
            f"DOCS_DIR not found: {docs_dir}")
    files = sorted(docs_dir.glob("*.docx"))
    require(files, f"No .docx files found in: {docs_dir}")

    docs: List[LCDocument] = []
    for fp in files:
        loaded = Docx2txtLoader(str(fp)).load()
        require(loaded, f"No content loaded from: {fp.name}")
        for d in loaded:
            d.metadata = d.metadata or {}
            d.metadata["source"] = fp.name
            d.metadata["doc_type"] = "policy"
        docs.extend(loaded)

    return docs


# -------------------------
# Salary roster -> docs
# -------------------------
SALARY_INTENT_RE = re.compile(
    r"\b(salary|ctc|compensation|pay|paid|earn|income|stipend|monthly|per month|annual|per annum|lpa)\b",
    re.IGNORECASE,
)
EMP_ID_RE = re.compile(r"\bEMP-\d{4}\b", re.IGNORECASE)


def is_salary_question(q: str) -> bool:
    return bool(SALARY_INTENT_RE.search(q))


def load_salary_roster(path: Path) -> pd.DataFrame:
    require(path.exists(), f"Salary CSV not found: {path}")
    df = pd.read_csv(path)

    required_cols = {"employee_id", "employee_name", "level",
                     "role", "monthly_salary_inr", "annual_salary_inr"}
    require(required_cols.issubset(df.columns),
            f"Salary CSV missing required columns: {sorted(required_cols)}")

    df = df.copy()
    df["employee_id"] = df["employee_id"].astype(str).str.strip()
    df["employee_name"] = df["employee_name"].astype(str).str.strip()
    df["level"] = df["level"].astype(str).str.strip()
    df["level_num"] = df["level"].str.extract(r"(\d+)").astype(int)

    require(df["employee_id"].is_unique,
            "employee_id must be unique in salary CSV.")
    return df


def salary_docs_from_roster(roster: pd.DataFrame) -> List[LCDocument]:
    docs: List[LCDocument] = []
    for _, r in roster.iterrows():
        content = (
            f"Employee ID: {r['employee_id']}\n"
            f"Employee Name: {r['employee_name']}\n"
            f"Level: {r['level']}\n"
            f"Role: {r['role']}\n"
            f"Monthly Salary (INR): {int(r['monthly_salary_inr'])}\n"
            f"Annual Salary (INR): {int(r['annual_salary_inr'])}\n"
        )
        docs.append(
            LCDocument(
                page_content=content,
                metadata={
                    "source": "salary_roster.csv",
                    "doc_type": "salary",
                    "employee_id": r["employee_id"],
                    "employee_name_norm": normalize(r["employee_name"]),
                    "level": r["level"],
                    "level_num": int(r["level_num"]),
                    "role": r["role"],
                },
            )
        )
    return docs


# -------------------------
# Build index
# -------------------------
def build_index(all_docs: List[LCDocument], embeddings) -> FAISS:
    splitter = SemanticChunker(embeddings=embeddings)
    chunks = splitter.split_documents(all_docs)
    require(chunks, "SemanticChunker produced zero chunks.")
    vs = FAISS.from_documents(chunks, embeddings)
    require(vs is not None, "FAISS build failed.")
    return vs


def overfetch_n(k: int) -> int:
    n = k * int(CONFIG["OVERFETCH_MULTIPLIER"])
    return min(max(n, k), int(CONFIG["MAX_CANDIDATES_CAP"]))


# -------------------------
# RBAC core
# -------------------------
def parse_level(level_str: str) -> int:
    m = re.search(r"(\d+)", level_str.strip().upper())
    require(m is not None, "Invalid level. Use L1, L2, ... L7.")
    lvl = int(m.group(1))
    require(1 <= lvl <= 7, "Level must be between L1 and L7.")
    return lvl


def resolve_employee_id(roster: pd.DataFrame, user_input: str) -> Optional[str]:
    """
    Accepts EMP-#### or exact employee name (case-insensitive).
    """
    s = user_input.strip()
    if not s:
        return None

    m = EMP_ID_RE.search(s)
    if m:
        eid = m.group(0).upper()
        if (roster["employee_id"].str.upper() == eid).any():
            return eid
        return None

    # try name match
    name_norm = normalize(s)
    match = roster.loc[roster["employee_name"].apply(
        lambda x: normalize(x) == name_norm)]
    if len(match) == 1:
        return str(match.iloc[0]["employee_id"]).upper()

    return None


def allowed_ids_for_level(roster: pd.DataFrame, requester_level: int, requester_emp_id: Optional[str]) -> Set[str]:
    """
    Returns the set of employee_ids visible to requester, based on level and (optionally) identity.
    - L7: all
    - L5–L6: own + strictly below (own requires requester_emp_id if you want it)
    - L1–L4: own only (requires requester_emp_id)
    """
    all_ids = set(roster["employee_id"].str.upper().tolist())

    if requester_level >= 7:
        return all_ids

    if requester_level <= 4:
        # must be only own
        require(requester_emp_id is not None,
                "Employee identity is required for L1–L4 salary access.")
        return {requester_emp_id.upper()}

    # L5–L6
    below = set(roster.loc[roster["level_num"] <
                requester_level, "employee_id"].str.upper().tolist())
    if requester_emp_id:
        below.add(requester_emp_id.upper())
    return below


def extract_targets(question: str, roster: pd.DataFrame, requester_emp_id: Optional[str]) -> Tuple[Set[str], bool]:
    """
    Returns (target_employee_ids, ambiguous)
    - supports: EMP-####, exact name, "my salary", "all employees"
    """
    q = question.strip()
    qlow = q.lower()

    # "my salary"
    if re.search(r"\bmy\b", qlow) or re.search(r"\bmine\b", qlow):
        if requester_emp_id:
            return {requester_emp_id.upper()}, False
        return set(), True

    # bulk
    if re.search(r"\b(all employees|everyone|salary list|all salaries|entire company)\b", qlow):
        return set(roster["employee_id"].str.upper().tolist()), False

    # explicit ids
    ids = {m.group(0).upper() for m in EMP_ID_RE.finditer(q)}
    if ids:
        known = set(roster["employee_id"].str.upper().tolist())
        resolved = ids & known
        return resolved, (len(resolved) == 0)

    # exact name (synthetic roster uses clean full names)
    qnorm = normalize(q)
    hits = set()
    for _, r in roster.iterrows():
        if normalize(r["employee_name"]) in qnorm:
            hits.add(str(r["employee_id"]).upper())
    if hits:
        return hits, False

    return set(), True


def rbac_check(question: str, roster: pd.DataFrame, requester_level: int, requester_emp_id: Optional[str]) -> Tuple[bool, str, Set[str], Set[str]]:
    """
    Returns: allowed?, deny_message, targets, allowed_ids
    """
    allowed_ids = allowed_ids_for_level(
        roster, requester_level, requester_emp_id)
    targets, ambiguous = extract_targets(question, roster, requester_emp_id)

    if ambiguous:
        return False, "Salary access check: please specify the employee (EMP-#### or full name) or ask 'my salary'.", set(), allowed_ids

    if not targets:
        return False, "Salary access check: could not resolve the employee target.", set(), allowed_ids

    if not targets.issubset(allowed_ids):
        return False, "Access denied: you are not permitted to view that salary based on your level.", targets, allowed_ids

    return True, "", targets, allowed_ids


# -------------------------
# Retrieval with RBAC filter
# -------------------------
def retrieve_contexts(vs: FAISS, query: str, k: int, allowed_salary_ids: Optional[Set[str]]) -> List[str]:
    """
    - policy docs: always allowed
    - salary docs: allowed only if allowed_salary_ids is provided and employee_id in allowed_salary_ids
    - if allowed_salary_ids is None: salary docs excluded completely
    """
    candidates = vs.similarity_search_with_score(query, k=overfetch_n(k))
    require(candidates, "Retriever returned zero results.")

    out: List[str] = []
    allowed_up = {x.upper()
                  for x in allowed_salary_ids} if allowed_salary_ids else set()

    for doc, _score in candidates:
        meta = doc.metadata or {}
        doc_type = meta.get("doc_type", "policy")

        if doc_type != "salary":
            out.append(doc.page_content)
        else:
            if allowed_salary_ids is None:
                continue
            eid = str(meta.get("employee_id", "")).upper()
            if eid and eid in allowed_up:
                out.append(doc.page_content)

        if len(out) >= k:
            break

    require(out, "RBAC filtering removed all contexts.")
    return out[:k]


def make_prompt(question: str, contexts: List[str]) -> str:
    ctx = "\n\n".join(
        [f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)])
    return f"""You are a careful assistant. Answer ONLY using the provided context.
If the context does not contain the answer, say: "I don't know based on the provided documents."
Do not reveal any salary information unless it is explicitly present in the provided context.

{ctx}

Question: {question}
Answer:"""


# -------------------------
# Interactive chat loop
# -------------------------
def main() -> int:
    load_env()
    embeddings = make_embeddings()
    llm = make_llm()

    docs_dir = here(CONFIG["DOCS_DIR"])
    salary_csv = here(CONFIG["SALARY_CSV"])

    policy_docs = load_policy_docs(docs_dir)
    roster = load_salary_roster(salary_csv)
    salary_docs = salary_docs_from_roster(roster)

    vs = build_index(policy_docs + salary_docs, embeddings)

    # RBAC session state (only asked when needed)
    session_level: Optional[int] = None
    session_emp_id: Optional[str] = None

    print("RAG system ready. Type 'exit' or 'quit' to stop.\n")

    while True:
        q = input("Q> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        # If salary question, invoke RBAC prompting
        if is_salary_question(q):
            if session_level is None:
                lvl_in = input("RBAC: Enter your level (L1–L7): ").strip()
                try:
                    session_level = parse_level(lvl_in)
                except Exception as e:
                    print(f"A> {e}\n")
                    session_level = None
                    continue

            # If level requires identity to enforce "only yours", ask only then
            # Also needed for "my salary" at any level.
            needs_identity = (session_level <= 4) or bool(
                re.search(r"\bmy\b|\bmine\b", q.lower()))
            if needs_identity and session_emp_id is None:
                emp_in = input(
                    "RBAC: Enter your Employee ID (EMP-####) or exact full name: ").strip()
                eid = resolve_employee_id(roster, emp_in)
                if eid is None:
                    print(
                        "A> RBAC: Could not match that employee. Use EMP-#### or the exact roster name.\n")
                    continue
                session_emp_id = eid

            ok, deny_msg, _targets, allowed_ids = rbac_check(
                q, roster, session_level, session_emp_id)
            if not ok:
                print(f"A> {deny_msg}\n")
                continue

            contexts = retrieve_contexts(vs, q, k=int(
                CONFIG["TOP_K"]), allowed_salary_ids=allowed_ids)
            resp = llm.invoke(make_prompt(q, contexts))
            ans = getattr(resp, "content", "").strip()
            print(f"A> {ans}\n")
            continue

        # Non-salary question: retrieve only policy contexts
        contexts = retrieve_contexts(vs, q, k=int(
            CONFIG["TOP_K"]), allowed_salary_ids=None)
        resp = llm.invoke(make_prompt(q, contexts))
        ans = getattr(resp, "content", "").strip()
        print(f"A> {ans}\n")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\nERROR: {e}\n", file=sys.stderr)
        raise
