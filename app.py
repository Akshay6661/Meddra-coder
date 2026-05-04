"""
app.py — Pharmacovigilance MedDRA Coding App
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from pipeline import init_pipeline, run_pipeline, lookup_llt, LLTResult

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedDRA Coding Agent",
    page_icon="💊",
    layout="wide",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background-color: #0f1117; }

    .header-block {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
    }
    .header-block h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        color: #e2e8f0;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .header-block p {
        color: #718096;
        margin: 0;
        font-size: 0.9rem;
    }

    .result-table {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
    }

    .tier-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
    }
    .tier-fuzzy  { background: #1a4731; color: #68d391; }
    .tier-bm25   { background: #744210; color: #f6ad55; }
    .tier-vector { background: #1a365d; color: #63b3ed; }
    .tier-fallback { background: #2d1515; color: #fc8181; }

    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .metric-card .val {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #63b3ed;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #718096;
        margin-top: 0.2rem;
    }

    .stTextArea textarea {
        font-family: 'IBM Plex Sans', sans-serif;
        background: #1a1f2e !important;
        border: 1px solid #2d3748 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: #2b6cb0;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.9rem;
        font-weight: 600;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #3182ce; }

    div[data-testid="stDataFrame"] {
        border: 1px solid #2d3748;
        border-radius: 8px;
        overflow: hidden;
    }

    .lookup-result {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Init pipeline (cached so it only runs once) ──────────────────────────────
@st.cache_resource(show_spinner="Loading MedDRA database & models...")
def load_pipeline():
    init_pipeline(
        excel_path=st.session_state.get("excel_path", "llt_dl.xlsx"),
        api_key=st.session_state["api_key"],
    )


# ─── Sidebar — Config & Lookup ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    api_key = st.text_input(
        "Euron API Key",
        type="password",
        placeholder="euri-xxxxxxxxxxxx",
    )
    excel_path = st.text_input(
        "LLT Excel Path",
        value="llt_dl.xlsx",
        placeholder="path/to/llt_dl.xlsx",
    )

    if api_key:
        st.session_state["api_key"]    = api_key
        st.session_state["excel_path"] = excel_path

    st.divider()

    # ── LLT Lookup ────────────────────────────────────────────────
    st.markdown("### 🔍 LLT Lookup")
    st.caption("Search by LLT code, parent code, or name")
    lookup_query = st.text_input(
        "Search",
        placeholder="e.g. 10019211 or Headache",
        label_visibility="collapsed",
    )

    if lookup_query and "api_key" in st.session_state:
        try:
            df_lookup = lookup_llt(lookup_query)
            if df_lookup.empty:
                st.warning("No results found.")
            else:
                st.dataframe(
                    df_lookup,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "LLT_CODE":    st.column_config.NumberColumn("LLT Code",    format="%d"),
                        "PARENT_CODE": st.column_config.NumberColumn("Parent Code", format="%d"),
                        "LLT_NAME":    st.column_config.TextColumn("LLT Name"),
                    }
                )
        except Exception as e:
            st.error(f"Load pipeline first: {e}")


# ─── Main Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
    <h1>💊 MedDRA Coding Agent</h1>
    <p>Pharmacovigilance · Verbatim Extraction · 3-Tier Search for Code</p>
</div>
""", unsafe_allow_html=True)


# ─── Guard: need API key first ────────────────────────────────────────────────
if "api_key" not in st.session_state:
    st.info("👈 Enter your Euron API key in the sidebar to get started.")
    st.stop()

# Load pipeline
load_pipeline()

# ─── Narrative Input ──────────────────────────────────────────────────────────
narrative = st.text_area(
    "Patient Narrative",
    placeholder="Paste patient narrative here...\n\ne.g. After taking the injection, my temple started paining badly. I was seeing double and couldn't catch my breath.",
    height=160,
    label_visibility="visible",
)

col_btn, col_clear = st.columns([3, 1])
with col_btn:
    run = st.button("⚡ Run MedDRA Coding", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.session_state.pop("results", None)
        st.rerun()

st.divider()

# ─── Run pipeline ────────────────────────────────────────────────────────────
if run:
    if not narrative.strip():
        st.warning("Please enter a patient narrative.")
    else:
        with st.spinner("Extracting verbatims & coding to MedDRA LLTs..."):
            results: list[LLTResult] = run_pipeline(narrative)
        st.session_state["results"]   = results
        st.session_state["narrative"] = narrative

# ─── Display Results ─────────────────────────────────────────────────────────
if "results" in st.session_state:
    results  = st.session_state["results"]
    narrative_display = st.session_state.get("narrative", "")

    if not results:
        st.error("No adverse event terms found in the narrative.")
    else:
        # ── Metrics row ───────────────────────────────────────────
        tiers = [r.search_tier for r in results]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="val">{len(results)}</div>
                <div class="label">Terms Coded</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="val">{tiers.count('fuzzy')}</div>
                <div class="label">🟢 Fuzzy Matches</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="val">{tiers.count('bm25')}</div>
                <div class="label">🟡 BM25 Matches</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card">
                <div class="val">{tiers.count('vector')}</div>
                <div class="label">🔵 Vector Matches</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Results table ─────────────────────────────────────────
        df = pd.DataFrame([r.model_dump() for r in results])

        # Rename columns for display
        df = df.rename(columns={
            "verbatim":    "Verbatim",
            "llt_name":    "LLT Name",
            "llt_code":    "LLT Code",
            "parent_code": "Parent Code",
            "search_tier": "Search Tier",
            "confidence":  "Confidence",
        })

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "LLT Code":    st.column_config.NumberColumn("LLT Code",    format="%d"),
                "Parent Code": st.column_config.NumberColumn("Parent Code", format="%d"),
                "Confidence":  st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100, format="%.1f"
                ),
                "Search Tier": st.column_config.TextColumn("Search Tier"),
            }
        )

        # ── Download button ───────────────────────────────────────
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=df.to_csv(index=False),
            file_name="meddra_coded_results.csv",
            mime="text/csv",
        )
