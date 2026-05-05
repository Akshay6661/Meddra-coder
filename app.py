"""
app.py — MedDRA Coding Agent (Production — Medical Reviewer Grade)
Run: streamlit run app.py
"""

import os
import streamlit as st

# ── Set secrets before any other imports ─────────────────────────
os.environ["HUGGING_FACE_HUB_TOKEN"] = st.secrets.get("HF_TOKEN", "")
os.environ["HF_TOKEN"]               = st.secrets.get("HF_TOKEN", "")

import pandas as pd
from pipeline import init_pipeline, run_pipeline, lookup_llt, LLTResult

# ─── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="MedDRA Coding Agent",
    page_icon="💊",
    layout="wide",
)

# ─── CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #2d3748;
    }
    .header h1 {
        font-family: 'IBM Plex Mono', monospace;
        color: #e2e8f0;
        font-size: 1.7rem;
        margin: 0 0 0.3rem 0;
    }
    .header p { color: #90cdf4; margin: 0; font-size: 0.85rem; }

    .metric-card {
        background: #1a202c;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card .val {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #63b3ed;
    }
    .metric-card .label { font-size: 0.75rem; color: #718096; margin-top: 4px; }

    .info-box {
        background: #1a2744;
        border-left: 4px solid #4299e1;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        color: #bee3f8;
    }

    .stButton > button {
        background: #2b6cb0;
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { background: #3182ce; }

    div[data-testid="stDataFrame"] {
        border: 1px solid #2d3748;
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ─── Event type config ────────────────────────────────────────────
EVENT_CONFIG = {
    "adverse_event":    {"icon": "🔴", "label": "Adverse Event"},
    "device_issue":     {"icon": "🔧", "label": "Device Issue"},
    "medication_error": {"icon": "🟡", "label": "Medication Error"},
    "product_quality":  {"icon": "🟠", "label": "Product Quality"},
    "lack_of_efficacy": {"icon": "🔵", "label": "Lack of Efficacy"},
    "other":            {"icon": "⚪", "label": "Other"},
}




# ─── Load pipeline ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(_api_key: str, _pinecone_key: str):
    init_pipeline(
        api_key=_api_key,
        pinecone_api_key=_pinecone_key,
        excel_path="MedDRA_LLT_PT_v28.1.xlsx",
    )

# ✅ Read both keys from secrets FIRST
api_key      = st.secrets["EURON_API_KEY"]
pinecone_key = st.secrets["PINECONE_API_KEY"]

# ✅ Then load pipeline
with st.spinner("Loading MedDRA database..."):
    load_pipeline(api_key, pinecone_key)


# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    api_key = st.secrets.get("EURON_API_KEY", "") or st.text_input(
        "Euron API Key", type="password", placeholder="euri-xxxxxxxxxxxx"
    )

    st.markdown("""
    <div style='font-size:0.75rem; color:#718096; margin-top:0.5rem;'>
        <b>Dataset:</b> MedDRA v28.1<br>
        <b>Model:</b> openai/gpt-oss-120b<br>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── MedDRA Lookup ─────────────────────────────────────────────
    st.markdown("### 🔍 MedDRA Lookup")
    st.caption("Search by LLT Code, PT Code, or Decode name")
    lookup_q = st.text_input(
        "lookup", placeholder="e.g. 10019211 or Headache",
        label_visibility="collapsed"
    )
    if lookup_q and api_key:
        try:
            df_l = lookup_llt(lookup_q)
            if df_l.empty:
                st.warning("No results found.")
            else:
                st.dataframe(df_l, use_container_width=True, hide_index=True,
                    column_config={
                        "LLT Code": st.column_config.NumberColumn(format="%d"),
                        "PT Code":  st.column_config.NumberColumn(format="%d"),
                    }
                )
        except Exception as e:
            st.error(str(e))

    st.divider()

    # ── Legend ────────────────────────────────────────────────────
    st.markdown("### 📋 Event Type Legend")
    for key, val in EVENT_CONFIG.items():
        st.markdown(f"{val['icon']} **{val['label']}**")


# ─── Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>💊 MedDRA Coding Agent</h1>
    <p>Pharmacovigilance · Medical Reviewer Grade · Consistent Output</p>
</div>
""", unsafe_allow_html=True)

if not api_key:
    st.info("👈 Enter your Euron API key in the sidebar to get started.")
    st.stop()

# Load pipeline
with st.spinner("Loading MedDRA database..."):
    load_pipeline(api_key)

# ─── Info box ────────────────────────────────────────────────────
st.markdown("""
<div class="info-box">
    ℹ️ Only for testing purpose
</div>
""", unsafe_allow_html=True)

# ─── Input ───────────────────────────────────────────────────────
narrative = st.text_area(
    "Patient Narrative / Case Description",
    placeholder=(
        "Paste the full case narrative here...\n\n"
        "e.g. Patient reported difficulty injecting the prescribed dose. "
        "The pen device was returned. Patient was given a replacement."
    ),
    height=200,
)

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    run = st.button("⚡ Run MedDRA Coding", use_container_width=True)
with col2:
    if st.button("🗑️ Clear Results", use_container_width=True):
        st.session_state.pop("results", None)
        st.session_state.pop("narrative_used", None)
        st.rerun()
with col3:
    if st.button("📋 Copy Narrative", use_container_width=True):
        st.toast("Use Ctrl+A in the text box to select all", icon="📋")

st.divider()

# ─── Run ─────────────────────────────────────────────────────────
if run:
    if not narrative.strip():
        st.warning("Please enter a patient narrative.")
    else:
        with st.spinner("🔍 Extracting events → Simplifying → Searching MedDRA..."):
            results = run_pipeline(narrative)
        st.session_state["results"]       = results
        st.session_state["narrative_used"] = narrative

# ─── Results ─────────────────────────────────────────────────────
if "results" in st.session_state:
    results  = st.session_state["results"]
    nar_used = st.session_state.get("narrative_used", "")

    if not results:
        st.error("❌ No reportable events found in the narrative.")
        st.stop()

    # ── Metrics ───────────────────────────────────────────────────
    tiers      = [r.search_tier for r in results]
    event_types = [r.event_type for r in results]

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (len(results),                    "Total Events"),
        (event_types.count("adverse_event"),    "🔴 Adverse Events"),
        (event_types.count("medication_error"), "🟡 Med Errors"),
        (event_types.count("device_issue"),     "🔧 Device Issues"),
        (event_types.count("product_quality") +
         event_types.count("lack_of_efficacy") +
         event_types.count("other"),            "⚪ Other"),
    ]
    for col, (val, label) in zip([c1,c2,c3,c4,c5], metrics):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="val">{val}</div>
                <div class="label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Results table ─────────────────────────────────────────────
    rows = []
    for r in results:
        cfg = EVENT_CONFIG.get(r.event_type, {"icon": "⚪", "label": r.event_type})
        rows.append({
            "Verbatim":     r.verbatim,
            "Simplified":   r.simplified,
            "Event Type":   f"{cfg['icon']} {cfg['label']}",
            "Decode":       r.decode,
            "LLT Code":     r.llt_code,
            "PT Code":      r.pt_code,
            "Search Tier":  r.search_tier,
            "Confidence":   r.confidence,
        })

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "LLT Code":   st.column_config.NumberColumn("LLT Code",  format="%d"),
            "PT Code":    st.column_config.NumberColumn("PT Code",   format="%d"),
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Verbatim":   st.column_config.TextColumn("Verbatim",   width="large"),
            "Simplified": st.column_config.TextColumn("Simplified", width="medium"),
            "Decode":     st.column_config.TextColumn("Decode",     width="medium"),
        }
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Expander: full case summary ───────────────────────────────
    with st.expander("📄 View Full Case Summary"):
        st.markdown("**Narrative Used:**")
        st.text(nar_used)
        st.markdown("**Coded Events (JSON):**")
        st.json([r.model_dump() for r in results])

    # ── Download ──────────────────────────────────────────────────
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="⬇️ Download as CSV",
            data=df.to_csv(index=False),
            file_name="meddra_coded_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_dl2:
        import json as _json
        st.download_button(
            label="⬇️ Download as JSON",
            data=_json.dumps([r.model_dump() for r in results], indent=2),
            file_name="meddra_coded_results.json",
            mime="application/json",
            use_container_width=True,
        )
