"""
pipeline.py — MedDRA Coding Pipeline (Production — Full 4-Layer System)
Dataset : MedDRA_LLT_PT_v28.1.xlsx  (LLT Code | Decode | PT Code)
Model   : openai/gpt-oss-120b via Euron
Search  : 3-Tier Hybrid — Fuzzy → BM25 → Pinecone Vector
Validate: GPT validation — OK | NEEDS_REVIEW | MANUAL_CODING_REQUIRED
Fixes   : Full error handling + tenacity retry on all LLM calls
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
from pinecone import Pinecone
from fastembed import TextEmbedding
from tenacity import retry, stop_after_attempt, wait_exponential

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


# ─── Pydantic Output Model ────────────────────────────────────────────────────
class LLTResult(BaseModel):
    verbatim:        str   = Field(..., description="Exact patient phrase")
    simplified:      str   = Field(..., description="Simplified term used for MedDRA search")
    event_type:      str   = Field(..., description="adverse_event | device_issue | medication_error | product_quality | lack_of_efficacy | other")
    decode:          str   = Field(..., description="Matched MedDRA Decode term")
    llt_code:        int   = Field(..., description="MedDRA LLT Code")
    pt_code:         int   = Field(..., description="MedDRA PT Code")
    search_tier:     str   = Field(..., description="fuzzy | bm25 | vector | fuzzy_fallback")
    confidence:      float = Field(..., description="Match confidence 0-100")
    is_valid:        bool  = Field(..., description="GPT validation result")
    validation_note: str   = Field(..., description="GPT validation reason")
    review_flag:     str   = Field(..., description="OK | NEEDS_REVIEW | MANUAL_CODING_REQUIRED")


# ─── Config ───────────────────────────────────────────────────────────────────
EXCEL_FILE          = "MedDRA_LLT_PT_v28.1.xlsx"
EURON_BASE_URL      = "https://api.euron.one/api/v1/euri"
MODEL_NAME          = "openai/gpt-oss-120b"

COL_LLT_CODE        = "LLT Code"
COL_DECODE          = "Decode"
COL_PT_CODE         = "PT Code"

FUZZY_THRESHOLD     = 90      # high = only very close string matches pass
BM25_THRESHOLD      = 12.0     # high = only strong keyword matches pass
VECTOR_TOP_N        = 3
VECTOR_SCORE_MIN    = 0.55    # cosine similarity minimum for Pinecone

BM25_CACHE_PATH     = "bm25_index.pkl"
PINECONE_INDEX_NAME = "meddra-llt"
EMBED_MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"

CONFIDENCE_OK           = 75
CONFIDENCE_NEEDS_REVIEW = 40

STOP_WORDS = set(stopwords.words("english"))


# ─── Globals ──────────────────────────────────────────────────────────────────
llt_df            = None
bm25_index        = None
decode_list_clean = None
embed_model       = None
pinecone_index    = None
llm               = None
agent             = None

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
                       e.g. "pen stopped working", "difficulty injecting",
                            "device returned", "pen malfunction"

3. medication_error  → Wrong dose, missed dose, wrong drug, wrong biosimilar, dispensing error
                       e.g. "dispensed wrong product", "wrong dose given",
                            "biosimilar dispensed instead of originator", "missed daily dose"

4. product_quality   → Labelling issue, SPC missing information, packaging defect,
                       contamination, unusual appearance
                       e.g. "SPC does not include dose information",
                            "tablet discoloured", "inadequate product labelling"

5. lack_of_efficacy  → Drug or device not working as expected
                       e.g. "dose could not be delivered", "treatment not effective"

6. other             → Any other reportable event not fitting above
─────────────────────────────────────────────────────────────────────────────

CRITICAL RULES:
- Extract verbatim EXACT words from the narrative. Do NOT rephrase.
- Extract EVERY reportable event do not skip any.
- Each term maps to exactly ONE event type.
- Do NOT include: enquirer questions, pharmacist background questions, dates,
  follow-up call details, or non-reportable administrative info.
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
- Remove: brand names (Bemfola, Gonal-F etc), drug names, IU doses, dates,
  numbers, patient details, route of admin, lot numbers
- Keep: the core medical event, device problem, or medication error concept
- Output must be a short clean phrase matching MedDRA terminology style
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

  Input:  "SPC does not include information on the dose to be injected per pen"
  Output: inadequate product information

  Input:  "thinking that Bemfola is a multiple use pen"
  Output: device use error

  Input:  "patient brought the pen back"
  Output: device returned by patient

  Input:  "felt nauseous and dizzy after injection"
  Output: nausea and dizziness

  Input:  "my temple is paining"
  Output: headache

  Input:  "could not deliver the required dose"
  Output: drug delivery failure

  Input:  "SPC does not include information on the dose to be injected per pen"
  Output: product information content complaint

  Input:  "label does not include dosing information"
  Output: product information content complaint

  Input:  "missing information in product leaflet"
  Output: product information content complaint

  Input:  "information not provided to patient about dose"
  Output: product information not provided to patient

  Input:  "incorrect information in SPC"
  Output: misleading product information
"""


_AGENT_SYSTEM_PROMPT = """
You are a senior MedDRA coding specialist for Pharmacovigilance case processing.

You will receive a list of objects with verbatim terms, their simplified forms,
and event_types extracted from a patient narrative.

For EACH item:
1. Call search_llt_tool using the "simplified" field as the search input
2. Use the top_match to get llt_code, pt_code, decode
3. Preserve verbatim, event_type, simplified EXACTLY as given

Return ONLY a valid JSON array — no explanation, no markdown, no reasoning:
[
  {
    "verbatim":    "exact original patient phrase",
    "simplified":  "simplified medical term used for search",
    "event_type":  "medication_error | device_issue | adverse_event | product_quality | lack_of_efficacy | other",
    "decode":      "matched MedDRA Decode term",
    "llt_code":    12345678,
    "pt_code":     12345678,
    "search_tier": "fuzzy | bm25 | vector | fuzzy_fallback",
    "confidence":  85.5
  }
]
"""


_VALIDATE_PROMPT = """
You are a senior MedDRA coding specialist reviewing a pharmacovigilance coding decision.

Given:
- Original verbatim phrase from the case narrative
- Simplified medical term used for MedDRA search
- Matched MedDRA LLT (Decode) term

Decide if the MedDRA match is medically appropriate for pharmacovigilance reporting.

Consider:
1. Does the LLT correctly represent the medical event described?
2. Is this LLT appropriate for the event type?
3. Would a trained PV medical reviewer accept this coding?

Return ONLY valid JSON no other text:
{
  "is_valid": true or false,
  "confidence_adjustment": number between -30 and +10,
  "validation_note": "brief clinical reason max 1 sentence",
  "review_flag": "OK" | "NEEDS_REVIEW" | "MANUAL_CODING_REQUIRED"
}

Review flag rules:
  OK                     → match is medically appropriate
  NEEDS_REVIEW           → match is plausible but reviewer should verify
  MANUAL_CODING_REQUIRED → match is wrong or no appropriate LLT found

Examples:
  Verbatim:   "difficulty injecting the prescribed dose"
  Simplified: "injection difficulty"
  Decode:     "Urination difficulty"
  Response:   {"is_valid": false, "confidence_adjustment": -25,
               "validation_note": "Urination difficulty is incorrect for injection difficulty",
               "review_flag": "MANUAL_CODING_REQUIRED"}

  Verbatim:   "couldn't have the daily dose"
  Simplified: "dose omission"
  Decode:     "Drug dose omission"
  Response:   {"is_valid": true, "confidence_adjustment": 8,
               "validation_note": "Exact MedDRA match for missed dose scenario",
               "review_flag": "OK"}

  Verbatim:   "returns of pen because of that"
  Simplified: "device return"
  Decode:     "Device rupture"
  Response:   {"is_valid": false, "confidence_adjustment": -30,
               "validation_note": "Device rupture is incorrect pen return is a complaint not rupture",
               "review_flag": "MANUAL_CODING_REQUIRED"}
"""


# ─── Init ─────────────────────────────────────────────────────────────────────
def init_pipeline(api_key: str, pinecone_api_key: str, excel_path: str = EXCEL_FILE):
    global llt_df, bm25_index, decode_list_clean
    global embed_model, pinecone_index, llm, agent

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

    # ── 3. Fastembed model for Pinecone queries ────────────────────
    embed_model = TextEmbedding(EMBED_MODEL_NAME)

    # ── 4. Pinecone connection ─────────────────────────────────────
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    # ── 5. LLM temperature=0 + seed=42 for consistency ────────────
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=api_key,
        base_url=EURON_BASE_URL,
        temperature=0,
        max_tokens=1500,
        seed=42,
    )

    # ── 6. Build LangGraph agent ───────────────────────────────────
    _build_agent()


def _build_agent():
    global agent
    llm_with_tools = llm.bind_tools([search_llt_tool])

    def call_model(state: MessagesState):
        try:
            sys_msg  = SystemMessage(content=_AGENT_SYSTEM_PROMPT)
            response = llm_with_tools.invoke([sys_msg] + state["messages"])
            return {"messages": [response]}
        except Exception as e:
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content="[]")]}

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
        "decode":   row[COL_DECODE],
        "llt_code": int(row[COL_LLT_CODE]),
        "pt_code":  int(row[COL_PT_CODE]),
        "score":    round(float(score), 2),
        "method":   method,
    }


def _narrative_hash(narrative: str) -> str:
    return hashlib.md5(narrative.strip().lower().encode()).hexdigest()


def _get_review_flag(confidence: float, is_valid: bool) -> str:
    if not is_valid:
        return "MANUAL_CODING_REQUIRED"
    if confidence >= CONFIDENCE_OK:
        return "OK"
    if confidence >= CONFIDENCE_NEEDS_REVIEW:
        return "NEEDS_REVIEW"
    return "MANUAL_CODING_REQUIRED"


# ─── Tier 1: Fuzzy ───────────────────────────────────────────────────────────
def _fuzzy_search(term: str) -> List[Dict]:
    try:
        results = process.extract(
            term.lower(),
            decode_list_clean,
            scorer=fuzz.token_sort_ratio,
            limit=3,
        )
        return [_row_to_dict(idx, score, "fuzzy") for _, score, idx in results]
    except Exception:
        return []


# ─── Tier 2: BM25 ────────────────────────────────────────────────────────────
def _bm25_search(term: str) -> List[Dict]:
    try:
        tokens = _tokenize(term)
        if not tokens:
            return []
        scores   = bm25_index.get_scores(tokens)
        top_idxs = np.argsort(scores)[::-1][:5]
        return [
            _row_to_dict(idx, scores[idx], "bm25")
            for idx in top_idxs if scores[idx] > 0
        ]
    except Exception:
        return []


# ─── Tier 3: Pinecone Vector Search ──────────────────────────────────────────
def _vector_search(term: str) -> List[Dict]:
    try:
        query_vec = list(embed_model.embed([term]))[0].tolist()
        results   = pinecone_index.query(
            vector=query_vec,
            top_k=VECTOR_TOP_N,
            include_metadata=True
        )
        output = []
        for match in results["matches"]:
            score = float(match["score"])
            if score < VECTOR_SCORE_MIN:
                continue
            output.append({
                "decode":   match["metadata"]["decode"],
                "llt_code": int(match["metadata"]["llt_code"]),
                "pt_code":  int(match["metadata"]["pt_code"]),
                "score":    round(score * 100, 2),
                "method":   "vector",
            })
        return output
    except Exception as e:
        print(f"Pinecone error: {e}")
        return []


# ─── Hybrid Search Router ─────────────────────────────────────────────────────
def hybrid_search(term: str) -> Dict:
    """
    Tier 1 Fuzzy  — exact/close string matches   score >= 90
    Tier 2 BM25   — strong keyword matches        score >= 8.0
    Tier 3 Vector — semantic meaning via Pinecone everything else
    """
    # Tier 1 — Fuzzy
    fuzzy = _fuzzy_search(term)
    if fuzzy and fuzzy[0]["score"] >= FUZZY_THRESHOLD:
        return {"term": term, "top_match": fuzzy[0], "search_used": "fuzzy"}

    # # Tier 2 — BM25
    # bm25 = _bm25_search(term)
    # if bm25 and bm25[0]["score"] >= BM25_THRESHOLD:
    #     return {"term": term, "top_match": bm25[0], "search_used": "bm25"}

    # Tier 3 — Pinecone Vector
    vector = _vector_search(term)
    if vector:
        return {"term": term, "top_match": vector[0], "search_used": "vector"}

    # Absolute fallback
    return {
        "term":        term,
        "top_match":   fuzzy[0] if fuzzy else None,
        "search_used": "fuzzy_fallback",
    }


# ─── LangChain Tool ───────────────────────────────────────────────────────────
@tool
def search_llt_tool(simplified_term: str) -> str:
    """
    Search MedDRA LLT dataset for a simplified medical term.
    Uses 3-tier hybrid: Fuzzy → BM25 → Pinecone Vector.
    Input: simplified medical concept NOT raw verbatim.
    """
    try:
        return json.dumps(hybrid_search(simplified_term))
    except Exception as e:
        return json.dumps({
            "term":        simplified_term,
            "top_match":   None,
            "search_used": "error",
            "error":       str(e)
        })


# ─── Layer 2: GPT Simplification ─────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
def simplify_verbatim(verbatim: str) -> str:
    """Strip brand/dose/date → extract core MedDRA-matchable concept."""
    try:
        messages = [
            SystemMessage(content=_SIMPLIFY_PROMPT),
            HumanMessage(content=verbatim),
        ]
        resp = llm.invoke(messages)
        return resp.content.strip().lower()
    except Exception:
        return verbatim.lower()


# ─── Layer 4: GPT Validation ──────────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
def validate_match(verbatim: str, simplified: str, decode: str) -> Dict:
    """Validates if MedDRA LLT match is medically appropriate."""
    try:
        prompt = (
            f"Verbatim:   {verbatim}\n"
            f"Simplified: {simplified}\n"
            f"Decode:     {decode}"
        )
        messages = [
            SystemMessage(content=_VALIDATE_PROMPT),
            HumanMessage(content=prompt),
        ]
        resp = llm.invoke(messages)
        raw  = re.sub(r"```json|```", "", resp.content.strip()).strip()

        json_start = raw.find("{")
        json_end   = raw.rfind("}")
        if json_start != -1 and json_end != -1:
            raw = raw[json_start: json_end + 1]

        result = json.loads(raw)
        return {
            "is_valid":              bool(result.get("is_valid", True)),
            "confidence_adjustment": float(result.get("confidence_adjustment", 0)),
            "validation_note":       str(result.get("validation_note", "")),
            "review_flag":           str(result.get("review_flag", "NEEDS_REVIEW")),
        }
    except Exception:
        return {
            "is_valid":              True,
            "confidence_adjustment": 0,
            "validation_note":       "Validation error — manual review recommended",
            "review_flag":           "NEEDS_REVIEW",
        }


# ─── Layer 1: Verbatim Extractor ──────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
def extract_verbatim(narrative: str) -> List[Dict]:
    """Extract verbatim terms + event types from narrative."""
    try:
        messages = [
            SystemMessage(content=_VERBATIM_PROMPT),
            HumanMessage(content=f"Patient Narrative:\n{narrative}"),
        ]
        resp = llm.invoke(messages)
        raw  = re.sub(r"```json|```", "", resp.content.strip()).strip()

        json_start = raw.find("[")
        json_end   = raw.rfind("]")
        if json_start != -1 and json_end != -1:
            raw = raw[json_start: json_end + 1]

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
    Full 4-layer pipeline with error handling + session caching.

    Layer 1 → GPT extracts verbatims + event types
    Layer 2 → GPT simplifies (strips brand/dose/date)
    Layer 3 → Fuzzy + BM25 + Pinecone Vector finds best LLT
    Layer 4 → GPT validates → OK | NEEDS_REVIEW | MANUAL_CODING_REQUIRED
    """
    # ── Cache check ───────────────────────────────────────────────
    cache_key = _narrative_hash(narrative)
    if cache_key in _result_cache:
        return _result_cache[cache_key]

    # ── Layer 1: Extract ──────────────────────────────────────────
    extracted = extract_verbatim(narrative)
    if not extracted:
        return []

    # ── Layer 2: Simplify ─────────────────────────────────────────
    for item in extracted:
        item["simplified"] = simplify_verbatim(item["verbatim"])

    # ── Layer 3: LLT matching via agent ───────────────────────────
    try:
        agent_input = (
            f"Match these terms to MedDRA LLTs using the simplified field:\n"
            f"{json.dumps(extracted, indent=2)}"
        )
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            output = agent.invoke({"messages": [HumanMessage(content=agent_input)]})

        raw = output["messages"][-1].content.strip()
        json_start = raw.find("[")
        json_end   = raw.rfind("]")
        if json_start != -1 and json_end != -1:
            raw = raw[json_start: json_end + 1]
        raw = re.sub(r"```json|```", "", raw).strip()
        coded = json.loads(raw)

    except Exception as e:
        print(f"Agent error: {e}")
        return []

    if not coded:
        return []

    # ── Layer 4: Validate each match ──────────────────────────────
    results = []
    for item in coded:
        if not isinstance(item, dict):
            continue
        try:
            verbatim   = item.get("verbatim", "")
            simplified = item.get("simplified", "")
            decode     = item.get("decode", "")
            confidence = float(item.get("confidence", 50.0))

            validation       = validate_match(verbatim, simplified, decode)
            final_confidence = max(0.0, min(100.0,
                confidence + validation["confidence_adjustment"]
            ))
            review_flag = validation["review_flag"] or _get_review_flag(
                final_confidence, validation["is_valid"]
            )

            results.append(LLTResult(
                verbatim        = verbatim,
                simplified      = simplified,
                event_type      = item.get("event_type", "other"),
                decode          = decode,
                llt_code        = int(item.get("llt_code", 0)),
                pt_code         = int(item.get("pt_code", 0)),
                search_tier     = item.get("search_tier", "fuzzy_fallback"),
                confidence      = round(final_confidence, 1),
                is_valid        = validation["is_valid"],
                validation_note = validation["validation_note"],
                review_flag     = review_flag,
            ))
        except Exception:
            continue

    # ── Cache and return ──────────────────────────────────────────
    _result_cache[cache_key] = results
    return results


# ─── Lookup Helper ────────────────────────────────────────────────────────────
def lookup_llt(search_value: str) -> pd.DataFrame:
    """Search by LLT Code, PT Code, or partial Decode name."""
    try:
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
    except Exception:
        return pd.DataFrame(columns=[COL_LLT_CODE, COL_DECODE, COL_PT_CODE])
