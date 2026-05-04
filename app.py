"""
app.py — Pharmacovigilance MedDRA Coding App
Dataset: MedDRA_LLT_PT_v28.1.xlsx  (LLT Code | Decode | PT Code)
Run: streamlit run app.py
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
    }
    .header-block p {
        color: #718096;
        margin: 0;
        font-size: 0.9rem;
    }

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
        border-radius: 8px !important;
    }
    .stButton > button {
        background: #2b6cb0;
        color: white;
        border: none;
        border-radius: 8px;
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
</style>
""", unsafe_allow_html=True)


# ─── Load pipeline (cached — runs once per session) ───────────────────────────
@st.cache_resource(show_spinner="⏳ Loading MedDRA v28.1 database & models...")
def load_pipeline(_api_key: str):
    """Underscore prefix on arg tells Streamlit not to hash it."""
    init_pipeline(
        api_key=_api_key,
        excel_path="MedDRA_LLT_PT_v28.1.xlsx",
    )


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Read from Streamlit secrets first (cloud deploy), fallback to input
    api_key = st.secrets.get("EURON_API_KEY", "") or st.text_input(
        "Euron API Key",
        type="password",
        placeholder="euri-xxxxxxxxxxxx",
    )

    st.caption("Dataset: MedDRA_LLT_PT_v28.1.xlsx")
    st.caption("Model: openai/gpt-oss-120b")

    st.divider()

    # ── LLT / PT Lookup ───────────────────────────────────────────
    st.markdown("### 🔍 MedDRA Lookup")
    st.caption("Search by LLT Code, PT Code, or Decode name")

    lookup_query = st.text_input(
        "Search",
        placeholder="e.g. 10019211 or Headache",
        label_visibility="collapsed",
    )

    if lookup_query and api_key:
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
                        "LLT Code": st.column_config.NumberColumn("LLT Code", format="%d"),
                        "PT Code":  st.column_config.NumberColumn("PT Code",  format="%d"),
                        "Decode":   st.column_config.TextColumn("Decode"),
                    }
                )
        except Exception as e:
            st.error(f"Pipeline not loaded yet: {e}")


# ─── Main Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
    <h1>💊 MedDRA Coding Agent</h1>
    <p>Pharmacovigilance · MedDRA v28.1 · Verbatim → LLT Code + PT Code · 3-Tier Hybrid Search</p>
</div>
""", unsafe_allow_html=True)

# ─── Guard ───────────────────────────────────────────────────────────────────
if not api_key:
    st.info("👈 Enter your Euron API key in the sidebar to get started.")
    st.stop()

# Load pipeline
load_pipeline(api_key)

# ─── Narrative Input ──────────────────────────────────────────────────────────
narrative = st.text_area(
    "Patient Narrative",
    placeholder=(
        "Paste patient narrative here...\n\n"
        "e.g. After taking the injection my temple started paining badly. "
        "I was seeing double and couldn't catch my breath."
    ),
    height=160,
)

col_run, col_clear = st.columns([3, 1])
with col_run:
    run = st.button("⚡ Run MedDRA Coding", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.session_state.pop("results", None)
        st.rerun()

st.divider()

# ─── Run Pipeline ────────────────────────────────────────────────────────────
if run:
    if not narrative.strip():
        st.warning("Please enter a patient narrative.")
    else:
        with st.spinner("Extracting verbatims & coding to MedDRA..."):
            results: list[LLTResult] = run_pipeline(narrative)
        st.session_state["results"]   = results
        st.session_state["narrative"] = narrative

# ─── Display Results ─────────────────────────────────────────────────────────
if "results" in st.session_state:
    results = st.session_state["results"]

    if not results:
        st.error("❌ No adverse event terms found in the narrative.")
    else:
        # ── Metrics ───────────────────────────────────────────────
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
                <div class="label">🟢 Fuzzy</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="val">{tiers.count('bm25')}</div>
                <div class="label">🟡 BM25</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card">
                <div class="val">{tiers.count('vector')}</div>
                <div class="label">🔵 Vector</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Results table ─────────────────────────────────────────
        df = pd.DataFrame([r.model_dump() for r in results])
        df = df.rename(columns={
            "verbatim":    "Verbatim",
            "decode":      "Decode",
            "llt_code":    "LLT Code",
            "pt_code":     "PT Code",
            "search_tier": "Search Tier",
            "confidence":  "Confidence",
        })

        # Reorder columns
        df = df[["Verbatim", "Decode", "LLT Code", "PT Code", "Search Tier", "Confidence"]]

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "LLT Code":   st.column_config.NumberColumn("LLT Code",  format="%d"),
                "PT Code":    st.column_config.NumberColumn("PT Code",   format="%d"),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100, format="%.1f"
                ),
                "Search Tier": st.column_config.TextColumn("Search Tier"),
            }
        )

        # ── Download ──────────────────────────────────────────────
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=df.to_csv(index=False),
            file_name="meddra_coded_results.csv",
            mime="text/csv",
        )
