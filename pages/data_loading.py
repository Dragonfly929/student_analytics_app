import streamlit as st
import pandas as pd
import os

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data.csv")

def load_df(path_or_buffer, source_name):
    df = pd.read_csv(path_or_buffer)
    st.session_state.df = df
    st.session_state.data_source = source_name
    return df

def show():
    st.title("📂 Data Loading")
    st.markdown("Choose how to load the dataset. The default dataset is the **Student Placement & Career Success 2026** CSV included with this application.")

    # ── Source selection ──────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-header">Option A · Default dataset</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            Use the bundled <b>student_placement_career_success_dataset_2026.csv</b><br>
            <span style="color:var(--muted);">20,000 rows · 20 columns · no missing values</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("▶ Load default dataset", use_container_width=True):
            df = load_df(DEFAULT_PATH, "Default (bundled CSV)")
            st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")

    with col2:
        st.markdown('<div class="section-header">Option B · Upload your own CSV</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            Upload a modified or newer version of the dataset.<br>
            <span style="color:var(--muted);">Must be a CSV file with a compatible schema.</span>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded is not None:
            if st.button("▶ Load uploaded file", use_container_width=True):
                df = load_df(uploaded, f"Uploaded: {uploaded.name}")
                st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")

    # ── Preview ───────────────────────────────────────────────────────────────
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("---")
        st.markdown("### Dataset Preview")

        tab1, tab2, tab3 = st.tabs(["🗂 First rows", "📋 Schema", "⚠️ Data quality"])

        with tab1:
            n = st.slider("Rows to show", 5, 50, 10)
            st.dataframe(df.head(n), use_container_width=True)

        with tab2:
            schema = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.values,
                "Non-null": df.notnull().sum().values,
                "Unique": df.nunique().values,
                "Sample": [str(df[c].iloc[0]) for c in df.columns]
            })
            st.dataframe(schema, use_container_width=True, hide_index=True)

        with tab3:
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Total rows", f"{len(df):,}")
            col_b.metric("Total columns", len(df.columns))
            col_c.metric("Missing values", int(df.isnull().sum().sum()))
            col_d.metric("Duplicate rows", int(df.duplicated().sum()))

            nulls = df.isnull().sum()
            nulls = nulls[nulls > 0]
            if len(nulls) == 0:
                st.markdown('<div class="success-box">✅ No missing values detected.</div>', unsafe_allow_html=True)
            else:
                st.warning("Columns with missing values:")
                st.dataframe(nulls.reset_index().rename(columns={"index": "Column", 0: "Missing"}))

            dups = df.duplicated().sum()
            if dups == 0:
                st.markdown('<div class="success-box">✅ No duplicate rows detected.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="warn-box">⚠️ {dups} duplicate rows found.</div>', unsafe_allow_html=True)
    else:
        st.markdown("---")
        st.markdown('<div class="info-box">👆 Load a dataset above to begin analysis.</div>', unsafe_allow_html=True)
