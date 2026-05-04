"""
pipeline.py — MedDRA Coding Pipeline (Production Grade)
Dataset : MedDRA_LLT_PT_v28.1.xlsx  (LLT Code | Decode | PT Code)
Model   : openai/gpt-oss-120b via Euron
Agent   : LangGraph StateGraph + ToolNode
Search  : 2-Tier Hybrid — Fuzzy → BM25 (+ GPT simplification for complex phrases)
Consistency: temperature=0 everywhere + result cache per narrative hash
"""

import os, re, json, pickle, hashlib
import pandas as pd
import numpy as np
import nltk
from typing import List, Dict, Optional
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

from rapidfuzz import process, fuzz
from rank_bm25 import BM25Okapi

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


# ─── Pydantic Models ──────────────────────────────────────────────────────────
class LLTResult(BaseModel):
    verbatim:       str   = Field(..., description="Exact patient phrase")
    simplified:     str   = Field(..., description="Simplified medical concept used for search")
    event_type:     str   = Field(..., description="adverse_event | device_issue | medication_error | product_quality | lack_of_efficacy | other")
    decode:         str   = Field(..., description="Matched MedDRA Decode term")
    llt_code:       int   = Field(..., description="MedDRA LLT Code")
    pt_code:        int   = Field(..., description="MedDRA PT Code")
    search_tier:    str   = Field(..., description="fuzzy | bm25 | fuzzy_fallback")
    confidence:     float = Field(..., description="Match confidence score 0-100")


# ─── Config ───────────────────────────────────────────────────────────────────
EXCEL_FILE      = "MedDRA_LLT_PT_v28.1.xlsx"
EURON_BASE_URL  = "https://api.euron.one/api/v1/euri"
MODEL_NAME      = "openai/gpt-oss-120b"

COL_LLT_CODE    = "LLT Code"
COL_DECODE      = "Decode"
COL_PT_CODE     = "PT Code"

FUZZY_THRESHOLD = 80
BM25_THRESHOLD  = 2.0
BM25_CACHE_PATH = "bm25_index.pkl"
STOP_WORDS      = set(stopwords.words("english"))


# ─── Globals ──────────────────────────────────────────────────────────────────
llt_df            = None
bm25_index        = None
decode_list_clean = None
llm               = None
agent             = None

# ── Result cache: narrative_hash → List[LLTResult] ────────────────────────────
# Ensures same narrative always returns same output in same session
_result_cache: Dict[str, List[LLTResult]] = {}


# ─── Prompts ──────────────────────────────────────────────────────────────────
_VERBATIM_PROMPT = """
You are a senior Pharmacovigilance medical reviewer with 20 years experience in 
adverse event reporting, medical device vigilance, and medication error detection.

Your job: extract ALL reportable events from the patient narrative and categorize each one.

EVENT TYPES:
─────────────────────────────────────────────────────────────────────────────
1. adverse_event     → Any symptom, reaction, or side effect experienced by patient
                       e.g. "difficulty breathing", "skin turned red", "felt dizzy"

2. device_issue      → Problem with a medical device, pen, injection device, delivery system
                       e.g. "pen stopped working", "needle broke", "difficulty injecting",
                            "device returned", "pen malfunction", "injection device issue"

3. medication_error  → Wrong dose, missed dose, wrong drug, wrong biosimilar, dispensing error
                       e.g. "dispensed wrong product", "wrong dose given", "missed daily dose",
                            "biosimilar dispensed instead of originator"

4. product_quality   → Contamination, packaging defect, unusual appearance, labelling issue
                       e.g. "tablet discoloured", "particles in vial", "SPC missing information",
                            "incomplete product information"

5. lack_of_efficacy  → Drug or device not working as expected
                       e.g. "dose could not be delivered", "treatment not effective"

6. other             → Any other reportable event
─────────────────────────────────────────────────────────────────────────────

CRITICAL RULES:
- Extract verbatim — EXACT words from the narrative. Do NOT rephrase.
- Extract EVERY reportable event — do not skip any.
- Each term maps to exactly ONE event type.
- Do NOT include: enquirer questions, background info, dates, follow-up calls.
- Return ONLY a valid JSON array. No other text.

Output format:
[
  {"verbatim": "exact phrase from narrative", "event_type": "medication_error"},
  {"verbatim": "exact phrase from narrative", "event_type": "device_issue"}
]
"""

_SIMPLIFY_PROMPT = """
You are a MedDRA coding expert with deep knowledge of MedDRA LLT terminology.

Given a verbatim phrase from a pharmacovigilance report, extract ONLY the core 
medical/clinical concept that would match a MedDRA LLT term.

RULES:
- Remove: brand names, drug names, IU doses, dates, numbers, patient details
- Keep: the core medical event, device problem, or medication error concept
- Output must be a short clean phrase that exists or is close to MedDRA terminology
- Return ONLY the simplified term. No explanation. No punctuation at the end.

Examples:
  Input:  "dispensed a Bemfola 450 IU on 02-Apr-2026 to inject 50 IU"
  Output: wrong dose dispensed

  Input:  "difficulty injecting the required prescribed 50 IU dose"
  Output: injection difficulty

  Input:  "couldn't have the daily dose"
  Output: dose omission

  Input:  "returns of pen because of that"
  Output: device return

  Input:  "Bemfola pen 75 IU was given to the patient"
  Output: wrong dose administered

  Input:  "dispensed Bemfola 450 IU as biosimilar for the whole treatment"
  Output: wrong product dispensed

  Input:  "SPC does not include information on the dose"
  Output: inadequate product information

  Input:  "thinking that Bemfola is a multiple use pen"
  Output: device use error

  Input:  "patient brought the pen back"
  Output: device returned by patient

  Input:  "felt nauseous and dizzy after injection"
  Output: nausea and dizziness
"""

_AGENT_SYSTEM_PROMPT = """
You are a senior MedDRA coding specialist for Pharmacovigilance case processing.

You will receive a list of objects with verbatim terms, their simplified forms, 
and event_types extracted from a patient narrative.

For EACH item:
1. Call search_llt_tool using the "simplified" field as input
2. Use the top_match result to fill llt_code, pt_code, decode
3. Preserve the original verbatim and event_type exactly as given

CRITICAL: Return ONLY a valid JSON array. No explanation, no markdown, no reasoning:
[
  {
    "verbatim":    "exact original patient phrase",
    "simplified":  "simplified medical term used for search",
    "event_type":  "medication_error | device_issue | adverse_event | product_quality | lack_of_efficacy | other",
    "decode":      "matched MedDRA Decode term",
    "llt_code":    12345678,
    "pt_code":     12345678,
    "search_tier": "fuzzy | bm25 | fuzzy_fallback",
    "confidence":  85.5
  }
]
"""


# ─── Init ─────────────────────────────────────────────────────────────────────
def init_pipeline(api_key: str, excel_path: str = EXCEL_FILE):
    global llt_df, bm25_index, decode_list_clean, llm, agent

    # ── 1. Load Excel ─────────────────────────────────────────────
    llt_df = pd.read_excel(excel_path)
    llt_df.columns = llt_df.columns.str.strip()

    assert COL_LLT_CODE in llt_df.columns, f"Missing: {COL_LLT_CODE}"
    assert COL_DECODE   in llt_df.columns, f"Missing: {COL_DECODE}"
    assert COL_PT_CODE  in llt_df.columns, f"Missing: {COL_PT_CODE}"

    llt_df["DECODE_CLEAN"] = llt_df[COL_DECODE].str.strip().str.lower()
    decode_list_clean = llt_df["DECODE_CLEAN"].tolist()

    # ── 2. Load or build BM25 ─────────────────────────────────────
    if os.path.exists(BM25_CACHE_PATH):
        with open(BM25_CACHE_PATH, "rb") as f:
            bm25_index = pickle.load(f)
    else:
        tokenized  = [_tokenize(n) for n in decode_list_clean]
        bm25_index = BM25Okapi(tokenized)
        with open(BM25_CACHE_PATH, "wb") as f:
            pickle.dump(bm25_index, f)

    # ── 3. LLM — temperature=0 EVERYWHERE for consistency ─────────
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=api_key,
        base_url=EURON_BASE_URL,
        temperature=0,       # ✅ deterministic output always
        max_tokens=1500,
        seed=42,             # ✅ fixed seed for reproducibility
    )

    # ── 4. Build agent ────────────────────────────────────────────
    _build_agent()


def _build_agent():
    global agent
    llm_with_tools = llm.bind_tools([search_llt_tool])

    def call_model(state: MessagesState):
        sys_msg  = SystemMessage(content=_AGENT_SYSTEM_PROMPT)
        response = llm_with_tools.invoke([sys_msg] + state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode([search_llt_tool])
    builder   = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    agent = builder.compile()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _tokenize(text: str) -> List[str]:
    tokens = word_tokenize(str(text).lower())
    return [t for t in tokens if t.isalpha() and t not in STOP_WORDS]


def _row_to_dict(idx: int, score: float, method: str) -> Dict:
    row = llt_df.iloc[idx]
    return {
        "decode":      row[COL_DECODE],
        "llt_code":    int(row[COL_LLT_CODE]),
        "pt_code":     int(row[COL_PT_CODE]),
        "score":       round(float(score), 2),
        "method":      method,
    }


def _narrative_hash(narrative: str) -> str:
    """Generate consistent hash for caching narrative results."""
    return hashlib.md5(narrative.strip().lower().encode()).hexdigest()


# ─── Tier 1: Fuzzy ───────────────────────────────────────────────────────────
def _fuzzy_search(term: str) -> List[Dict]:
    results = process.extract(
        term.lower(),
        decode_list_clean,
        scorer=fuzz.token_sort_ratio,
        limit=3,
    )
    return [_row_to_dict(idx, score, "fuzzy") for _, score, idx in results]


# ─── Tier 2: BM25 ────────────────────────────────────────────────────────────
def _bm25_search(term: str) -> List[Dict]:
    tokens = _tokenize(term)
    if not tokens:
        return []
    scores   = bm25_index.get_scores(tokens)
    top_idxs = np.argsort(scores)[::-1][:5]
    return [
        _row_to_dict(idx, scores[idx], "bm25")
        for idx in top_idxs if scores[idx] > 0
    ]


# ─── Hybrid Search ───────────────────────────────────────────────────────────
def hybrid_search(term: str) -> Dict:
    # Tier 1 — Fuzzy
    fuzzy = _fuzzy_search(term)
    if fuzzy and fuzzy[0]["score"] >= FUZZY_THRESHOLD:
        return {"term": term, "top_match": fuzzy[0], "search_used": "fuzzy"}

    # Tier 2 — BM25
    bm25 = _bm25_search(term)
    if bm25 and bm25[0]["score"] >= BM25_THRESHOLD:
        return {"term": term, "top_match": bm25[0], "search_used": "bm25"}

    # Fallback — return best fuzzy even if low confidence
    return {
        "term":        term,
        "top_match":   fuzzy[0] if fuzzy else None,
        "search_used": "fuzzy_fallback",
    }


# ─── GPT Simplifier ───────────────────────────────────────────────────────────
def simplify_verbatim(verbatim: str) -> str:
    """
    Strip brand names, doses, dates from verbatim.
    Extract core MedDRA-matchable medical concept.
    temperature=0 ensures same input → same output always.
    """
    messages = [
        SystemMessage(content=_SIMPLIFY_PROMPT),
        HumanMessage(content=verbatim),
    ]
    resp = llm.invoke(messages)
    return resp.content.strip().lower()


# ─── LangChain Tool ───────────────────────────────────────────────────────────
@tool
def search_llt_tool(simplified_term: str) -> str:
    """
    Search MedDRA LLT dataset for a simplified medical term.
    Uses Fuzzy → BM25 hybrid search.
    Input: simplified medical concept (NOT raw verbatim).
    Returns: matched Decode, LLT Code, PT Code, confidence.
    """
    result = hybrid_search(simplified_term)
    return json.dumps(result)


# ─── Verbatim Extractor ───────────────────────────────────────────────────────
def extract_verbatim(narrative: str) -> List[Dict]:
    """
    Extract verbatim terms + event types from narrative.
    temperature=0 ensures deterministic output.
    """
    messages = [
        SystemMessage(content=_VERBATIM_PROMPT),
        HumanMessage(content=f"Patient Narrative:\n{narrative}"),
    ]
    resp = llm.invoke(messages)
    raw  = re.sub(r"```json|```", "", resp.content.strip()).strip()

    # Extract JSON array
    json_start = raw.find("[")
    json_end   = raw.rfind("]")
    if json_start != -1 and json_end != -1:
        raw = raw[json_start: json_end + 1]

    try:
        result = json.loads(raw)
        assert isinstance(result, list)
        normalized = []
        for item in result:
            if isinstance(item, str):
                normalized.append({"verbatim": item, "event_type": "adverse_event"})
            elif isinstance(item, dict):
                normalized.append(item)
        return normalized
    except Exception:
        return []


# ─── Main Pipeline ────────────────────────────────────────────────────────────
def run_pipeline(narrative: str) -> List[LLTResult]:
    """
    Full pipeline with caching for consistency.
    Same narrative always returns same result within a session.

    Narrative → Verbatim + Event Type
             → Simplify each verbatim (GPT strips drug names/doses)
             → Search MedDRA LLT (Fuzzy + BM25)
             → Structured LLTResult list
    """
    # ── Check cache first ─────────────────────────────────────────
    cache_key = _narrative_hash(narrative)
    if cache_key in _result_cache:
        return _result_cache[cache_key]

    # ── Step 1: Extract verbatims + event types ───────────────────
    extracted = extract_verbatim(narrative)
    if not extracted:
        return []

    # ── Step 2: Simplify each verbatim ────────────────────────────
    for item in extracted:
        item["simplified"] = simplify_verbatim(item["verbatim"])

    # ── Step 3: LLT matching via agent ────────────────────────────
    agent_input = (
        f"Match these terms to MedDRA LLTs using the simplified field for search:\n"
        f"{json.dumps(extracted, indent=2)}"
    )

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        output = agent.invoke({"messages": [HumanMessage(content=agent_input)]})

    raw = output["messages"][-1].content.strip()

    # Strip reasoning text before JSON
    json_start = raw.find("[")
    json_end   = raw.rfind("]")
    if json_start != -1 and json_end != -1:
        raw = raw[json_start: json_end + 1]

    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        coded   = json.loads(raw)
        results = [LLTResult(**item) for item in coded if isinstance(item, dict)]
    except Exception:
        results = []

    # ── Cache result ──────────────────────────────────────────────
    _result_cache[cache_key] = results
    return results


# ─── Lookup Helper ────────────────────────────────────────────────────────────
def lookup_llt(search_value: str) -> pd.DataFrame:
    """Search by LLT Code, PT Code, or partial Decode name."""
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
