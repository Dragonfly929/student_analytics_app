import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Student Placement Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --accent: #58a6ff;
    --accent2: #3fb950;
    --accent3: #f78166;
    --accent4: #d2a8ff;
    --text: #e6edf3;
    --muted: #8b949e;
}

html, body, .stApp { background-color: var(--bg) !important; color: var(--text) !important; }

.stApp { font-family: 'DM Sans', sans-serif; }

h1, h2, h3, .section-title { font-family: 'Space Mono', monospace !important; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

.stButton>button {
    background: var(--accent) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stButton>button:hover { opacity: 0.85 !important; }

.stSelectbox>div>div, .stMultiSelect>div>div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
}

.metric-label {
    font-size: 0.8rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

.info-box {
    background: rgba(88,166,255,0.08);
    border: 1px solid rgba(88,166,255,0.25);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}

.success-box {
    background: rgba(63,185,80,0.08);
    border: 1px solid rgba(63,185,80,0.25);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}

.warn-box {
    background: rgba(247,129,102,0.08);
    border: 1px solid rgba(247,129,102,0.25);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}

.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px !important; }

.sidebar-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    color: var(--text);
    margin-bottom: 0.3rem;
}

.sidebar-sub {
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 1.5rem;
}

.nav-item {
    padding: 0.6rem 1rem;
    border-radius: 6px;
    margin-bottom: 0.3rem;
    cursor: pointer;
    font-weight: 500;
}

.stRadio label { color: var(--text) !important; }

[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--muted) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

hr { border-color: var(--border) !important; }

.stAlert { border-radius: 8px !important; }

div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--surface) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "data_source" not in st.session_state:
    st.session_state.data_source = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🎓 Student Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Placement & Career Success · 2026</div>', unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📂 Data Loading",
         "📊 Descriptive Analysis",
         "🔍 Diagnostic Analysis",
         "🤖 Predictive Analysis",
         "🧭 Prescriptive Analysis"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    if st.session_state.df is not None:
        df_info = st.session_state.df
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Dataset loaded</div>
            <div class="metric-value">{len(df_info):,}</div>
            <div class="metric-label">rows · {len(df_info.columns)} columns</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Source: {st.session_state.data_source}")

# ── Route pages ──────────────────────────────────────────────────────────────
if page == "📂 Data Loading":
    from views.data_loading import show
    show()
elif page == "📊 Descriptive Analysis":
    from views.descriptive import show
    show()
elif page == "🔍 Diagnostic Analysis":
    from views.diagnostic import show
    show()
elif page == "🤖 Predictive Analysis":
    from views.predictive import show
    show()
elif page == "🧭 Prescriptive Analysis":
    from views.prescriptive import show
    show()
