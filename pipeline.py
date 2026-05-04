"""
pipeline.py — MedDRA Coding Pipeline
Dataset: MedDRA_LLT_PT_v28.1.xlsx
Columns: LLT Code | Decode | PT Code
Model  : openai/gpt-oss-120b via Euron
Agent  : LangGraph StateGraph + ToolNode
Search : 3-Tier Hybrid — Fuzzy → BM25 → Vector (fastembed, no torch)
"""

import os, re, json, pickle
import pandas as pd
import numpy as np
import nltk
from typing import List, Dict
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

from rapidfuzz import process, fuzz
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding
import faiss

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from pydantic import BaseModel, Field

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# ─── Pydantic output model ────────────────────────────────────────────────────
class LLTResult(BaseModel):
    verbatim:    str   = Field(..., description="Exact patient phrase")
    decode:      str   = Field(..., description="Matched MedDRA Decode (LLT name)")
    llt_code:    int   = Field(..., description="MedDRA LLT Code")
    pt_code:     int   = Field(..., description="MedDRA PT Code")
    search_tier: str   = Field(..., description="fuzzy | bm25 | vector | fuzzy_fallback")
    confidence:  float = Field(..., description="Match confidence score")


# ─── Config ───────────────────────────────────────────────────────────────────
EXCEL_FILE        = "MedDRA_LLT_PT_v28.1.xlsx"
EURON_BASE_URL    = "https://api.euron.one/api/v1/euri"
MODEL_NAME        = "openai/gpt-oss-120b"

# ── Column names from the new dataset ────────────────────────────
COL_LLT_CODE  = "LLT Code"
COL_DECODE    = "Decode"
COL_PT_CODE   = "PT Code"

# ── Search thresholds ─────────────────────────────────────────────
FUZZY_THRESHOLD   = 85
BM25_THRESHOLD    = 3.0
VECTOR_TOP_N      = 3
EMBED_MODEL_NAME  = "BAAI/bge-small-en-v1.5"
VECTOR_INDEX_PATH = "meddra_faiss.index"
VECTOR_META_PATH  = "meddra_faiss_meta.pkl"
STOP_WORDS        = set(stopwords.words("english"))


# ─── Globals ─────────────────────────────────────────────────────────────────
llt_df          = None
bm25_index      = None
decode_list_clean = None
embed_model     = None
faiss_index     = None
faiss_meta      = None
llm             = None
agent           = None


# ─── Prompts ─────────────────────────────────────────────────────────────────
_AGENT_SYSTEM_PROMPT = """
You are a MedDRA coding specialist for Pharmacovigilance.
For EACH verbatim term given, call search_llt_tool and use the top_match result.
After processing ALL terms return ONLY a valid JSON array — no explanation,
no markdown, no commentary, no reasoning text:
[
  {
    "verbatim":    "exact patient phrase",
    "decode":      "matched MedDRA Decode term",
    "llt_code":    12345678,
    "pt_code":     12345678,
    "search_tier": "fuzzy | bm25 | vector | fuzzy_fallback",
    "confidence":  85.5
  }
]
"""

_VERBATIM_PROMPT = """
You are a Pharmacovigilance expert. Extract verbatim symptom/adverse event terms
from the patient narrative.

Rules:
- Keep the patient's exact words. Do NOT rephrase or medically interpret.
- Exclude drug names, dosages, and demographics.
- Return ONLY a valid JSON array of strings. No other text whatsoever.
- If no symptoms found return: []

Examples:
  Input:  "my temple was paining and felt sick to my tummy"
  Output: ["my temple was paining", "felt sick to my tummy"]

  Input:  "I was seeing double and couldn't catch my breath"
  Output: ["seeing double", "couldn't catch my breath"]
"""


# ─── Init ────────────────────────────────────────────────────────────────────
def init_pipeline(api_key: str, excel_path: str = EXCEL_FILE):
    global llt_df, bm25_index, decode_list_clean
    global embed_model, faiss_index, faiss_meta, llm, agent

    # ── 1. Load dataset ───────────────────────────────────────────
    llt_df = pd.read_excel(excel_path)

    # Normalize column names — strip whitespace
    llt_df.columns = llt_df.columns.str.strip()

    assert COL_LLT_CODE in llt_df.columns, f"Missing column: {COL_LLT_CODE}"
    assert COL_DECODE   in llt_df.columns, f"Missing column: {COL_DECODE}"
    assert COL_PT_CODE  in llt_df.columns, f"Missing column: {COL_PT_CODE}"

    # Clean decode column for matching
    llt_df["DECODE_CLEAN"] = llt_df[COL_DECODE].str.strip().str.lower()
    decode_list_clean = llt_df["DECODE_CLEAN"].tolist()

    # ── 2. BM25 index on Decode column ───────────────────────────
    tokenized_corpus = [_tokenize(n) for n in decode_list_clean]
    bm25_index = BM25Okapi(tokenized_corpus)

    # ── 3. Fastembed + FAISS ──────────────────────────────────────
    embed_model = TextEmbedding(EMBED_MODEL_NAME)
    faiss_index, faiss_meta = _build_or_load_faiss()

    # ── 4. LLM ────────────────────────────────────────────────────
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=api_key,
        base_url=EURON_BASE_URL,
        temperature=0,
        max_tokens=1500,
    )

    # ── 5. LangGraph StateGraph agent ────────────────────────────
    llm_with_tools = llm.bind_tools([search_llt_tool])

    def call_model(state: MessagesState):
        sys_msg  = SystemMessage(content=_AGENT_SYSTEM_PROMPT)
        response = llm_with_tools.invoke([sys_msg] + state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode([search_llt_tool])

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    agent = builder.compile()


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _tokenize(text: str) -> List[str]:
    tokens = word_tokenize(str(text).lower())
    return [t for t in tokens if t.isalpha() and t not in STOP_WORDS]


def _row_to_dict(idx: int, score: float, method: str) -> Dict:
    row = llt_df.iloc[idx]
    return {
        "decode":      row[COL_DECODE],
        "llt_code":    int(row[COL_LLT_CODE]),
        "pt_code":     int(row[COL_PT_CODE]),
        "score":       round(float(score), 4),
        "method":      method,
    }


def _build_or_load_faiss():
    if os.path.exists(VECTOR_INDEX_PATH) and os.path.exists(VECTOR_META_PATH):
        index = faiss.read_index(VECTOR_INDEX_PATH)
        with open(VECTOR_META_PATH, "rb") as f:
            meta = pickle.load(f)
        return index, meta

    decode_names = llt_df[COL_DECODE].tolist()
    embeddings   = np.array(
        list(embed_model.embed(decode_names)), dtype="float32"
    )
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, VECTOR_INDEX_PATH)

    meta = {
        "decodes":    llt_df[COL_DECODE].tolist(),
        "llt_codes":  llt_df[COL_LLT_CODE].tolist(),
        "pt_codes":   llt_df[COL_PT_CODE].tolist(),
    }
    with open(VECTOR_META_PATH, "wb") as f:
        pickle.dump(meta, f)
    return index, meta


# ─── Tier 1: Fuzzy ───────────────────────────────────────────────────────────
def _fuzzy_search(verbatim: str) -> List[Dict]:
    results = process.extract(
        verbatim.lower(),
        decode_list_clean,
        scorer=fuzz.token_sort_ratio,
        limit=3,
    )
    return [_row_to_dict(idx, score, "fuzzy") for _, score, idx in results]


# ─── Tier 2: BM25 ────────────────────────────────────────────────────────────
def _bm25_search(verbatim: str) -> List[Dict]:
    tokens = _tokenize(verbatim)
    if not tokens:
        return []
    scores   = bm25_index.get_scores(tokens)
    top_idxs = np.argsort(scores)[::-1][:5]
    return [
        _row_to_dict(idx, scores[idx], "bm25")
        for idx in top_idxs if scores[idx] > 0
    ]


# ─── Tier 3: Vector ──────────────────────────────────────────────────────────
def _vector_search(verbatim: str) -> List[Dict]:
    qvec = np.array(list(embed_model.embed([verbatim])), dtype="float32")
    faiss.normalize_L2(qvec)
    scores, indices = faiss_index.search(qvec, VECTOR_TOP_N)
    output = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        output.append({
            "decode":   faiss_meta["decodes"][idx],
            "llt_code": int(faiss_meta["llt_codes"][idx]),
            "pt_code":  int(faiss_meta["pt_codes"][idx]),
            "score":    round(float(score), 4),
            "method":   "vector",
        })
    return output


# ─── Hybrid router ───────────────────────────────────────────────────────────
def hybrid_search(verbatim: str) -> Dict:
    # Tier 1 — Fuzzy
    fuzzy = _fuzzy_search(verbatim)
    if fuzzy and fuzzy[0]["score"] >= FUZZY_THRESHOLD:
        return {"verbatim": verbatim, "top_match": fuzzy[0], "search_used": "fuzzy"}

    # Tier 2 — BM25
    bm25 = _bm25_search(verbatim)
    if bm25 and bm25[0]["score"] >= BM25_THRESHOLD:
        return {"verbatim": verbatim, "top_match": bm25[0], "search_used": "bm25"}

    # Tier 3 — Vector
    vector = _vector_search(verbatim)
    if vector:
        return {"verbatim": verbatim, "top_match": vector[0], "search_used": "vector"}

    # Fallback
    return {
        "verbatim":    verbatim,
        "top_match":   fuzzy[0] if fuzzy else None,
        "search_used": "fuzzy_fallback",
    }


# ─── LangChain Tool ───────────────────────────────────────────────────────────
@tool
def search_llt_tool(verbatim: str) -> str:
    """
    Search MedDRA LLT dataset for a verbatim symptom term.
    Uses 3-tier hybrid search: Fuzzy → BM25 → Vector.
    Returns matched Decode, LLT Code, PT Code.
    Input: single verbatim symptom string.
    """
    return json.dumps(hybrid_search(verbatim))


# ─── Verbatim extractor ───────────────────────────────────────────────────────
def extract_verbatim(narrative: str) -> List[str]:
    messages = [
        SystemMessage(content=_VERBATIM_PROMPT),
        HumanMessage(content=f"Narrative:\n{narrative}"),
    ]
    resp = llm.invoke(messages)
    raw  = re.sub(r"```json|```", "", resp.content.strip()).strip()
    try:
        result = json.loads(raw)
        assert isinstance(result, list)
        return result
    except Exception:
        return []


# ─── Main pipeline ────────────────────────────────────────────────────────────
def run_pipeline(narrative: str) -> List[LLTResult]:
    verbatims = extract_verbatim(narrative)
    if not verbatims:
        return []

    agent_input = (
        f"Match these verbatim terms to MedDRA LLTs: {json.dumps(verbatims)}"
    )

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        output = agent.invoke({"messages": [HumanMessage(content=agent_input)]})

    raw = output["messages"][-1].content.strip()

    # Strip any reasoning text before the JSON array
    json_start = raw.find("[")
    json_end   = raw.rfind("]")
    if json_start != -1 and json_end != -1:
        raw = raw[json_start: json_end + 1]

    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        coded = json.loads(raw)
        return [LLTResult(**item) for item in coded if isinstance(item, dict)]
    except Exception:
        return []


# ─── Lookup helper ────────────────────────────────────────────────────────────
def lookup_llt(search_value: str) -> pd.DataFrame:
    """Search by LLT Code, PT Code, or Decode name (partial match)."""
    search_value = str(search_value).strip()
    if search_value.isdigit():
        code      = int(search_value)
        by_llt    = llt_df[llt_df[COL_LLT_CODE] == code]
        by_pt     = llt_df[llt_df[COL_PT_CODE]  == code]
        result    = pd.concat([by_llt, by_pt]).drop_duplicates()
    else:
        result = llt_df[
            llt_df[COL_DECODE].str.contains(search_value, case=False, na=False)
        ]
    return result[[COL_LLT_CODE, COL_DECODE, COL_PT_CODE]].reset_index(drop=True)
