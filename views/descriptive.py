import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOT_THEME = "plotly_dark"
COLOR_SEQ = px.colors.qualitative.Bold

def guard():
    if st.session_state.df is None:
        st.warning("⚠️ Please load a dataset first on the **Data Loading** page.")
        st.stop()
    return st.session_state.df

def show():
    df = guard()
    st.title("📊 Descriptive Analysis")
    st.markdown("Summary statistics, distributions, and univariate/bivariate/multivariate exploration of the student dataset.")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # ── 1. Summary statistics ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">1 · Summary Statistics</div>', unsafe_allow_html=True)

    desc = df[num_cols].describe().T
    desc["skewness"] = df[num_cols].skew()
    desc["kurtosis"] = df[num_cols].kurt()
    st.dataframe(desc.style.format("{:.2f}"), width="stretch")

    # Top-level KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    placed = df["Placement_Status"].sum() if "Placement_Status" in df.columns else 0
    pct = placed / len(df) * 100
    avg_sal = df["Salary_LPA"].mean() if "Salary_LPA" in df.columns else 0
    avg_cgpa = df["CGPA"].mean() if "CGPA" in df.columns else 0
    c1.metric("Total Students", f"{len(df):,}")
    c2.metric("Placed", f"{int(placed):,}")
    c3.metric("Placement Rate", f"{pct:.1f}%")
    c4.metric("Avg Salary (LPA)", f"{avg_sal:.2f}")
    c5.metric("Avg CGPA", f"{avg_cgpa:.2f}")

    # ── 2. Univariate ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">2 · Univariate Analysis</div>', unsafe_allow_html=True)
    col_choice = st.selectbox("Select a variable", num_cols + cat_cols)

    if col_choice in num_cols:
        col_left, col_right = st.columns(2)
        with col_left:
            fig = px.histogram(df, x=col_choice, nbins=50,
                               title=f"Distribution of {col_choice}",
                               color_discrete_sequence=["#58a6ff"],
                               template=PLOT_THEME)
            fig.update_layout(bargap=0.05, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
        with col_right:
            fig2 = px.box(df, y=col_choice, title=f"Boxplot · {col_choice}",
                          color_discrete_sequence=["#3fb950"], template=PLOT_THEME)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, width="stretch")

        # Stats table
        s = df[col_choice]
        stats = {
            "Mean": s.mean(), "Median": s.median(), "Mode": float(s.mode()[0]),
            "Std Dev": s.std(), "Variance": s.var(),
            "Min": s.min(), "Max": s.max(),
            "Skewness": s.skew(), "Kurtosis": s.kurt(),
            "IQR": s.quantile(0.75) - s.quantile(0.25)
        }
        st.dataframe(pd.DataFrame(stats, index=["Value"]).T.rename(columns={"Value": col_choice}),
                     width="stretch")
    else:
        vc = df[col_choice].value_counts().reset_index()
        vc.columns = [col_choice, "count"]
        fig = px.bar(vc, x=col_choice, y="count",
                     title=f"Frequency · {col_choice}",
                     color=col_choice, color_discrete_sequence=COLOR_SEQ,
                     template=PLOT_THEME)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, width="stretch")

        fig2 = px.pie(vc, names=col_choice, values="count",
                      title=f"Share · {col_choice}",
                      color_discrete_sequence=COLOR_SEQ, template=PLOT_THEME)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, width="stretch")

    # ── 3. Bivariate ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">3 · Bivariate Analysis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        x_var = st.selectbox("X variable", num_cols, index=0, key="biv_x")
    with c2:
        y_var = st.selectbox("Y variable", num_cols, index=min(1, len(num_cols)-1), key="biv_y")

    color_var = st.selectbox("Color by (optional)", ["None"] + cat_cols + ["Placement_Status"], key="biv_col")
    cv = None if color_var == "None" else color_var

    fig = px.scatter(df.sample(min(2000, len(df)), random_state=42),
                     x=x_var, y=y_var, color=cv,
                     opacity=0.55, trendline="ols",
                     title=f"{x_var} vs {y_var}",
                     color_discrete_sequence=COLOR_SEQ, template=PLOT_THEME)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

    # ── 4. Multivariate ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">4 · Multivariate · Pair Plot (sample)</div>', unsafe_allow_html=True)
    default_vars = ["CGPA", "Aptitude_Test_Score", "Mock_Interview_Score",
                    "Resume_Score", "Salary_LPA"]
    chosen = st.multiselect("Select variables (3-6 recommended)", num_cols,
                            default=[v for v in default_vars if v in num_cols])
    if len(chosen) >= 2:
        sample = df[chosen + (["Placement_Status"] if "Placement_Status" in df.columns else [])].sample(
            min(1500, len(df)), random_state=1)
        if "Placement_Status" in sample.columns:
            sample["Placed"] = sample["Placement_Status"].map({1: "Placed", 0: "Not Placed"})
            fig = px.scatter_matrix(sample, dimensions=chosen, color="Placed",
                                    color_discrete_sequence=["#58a6ff", "#f78166"],
                                    opacity=0.4, template=PLOT_THEME)
        else:
            fig = px.scatter_matrix(sample, dimensions=chosen, opacity=0.4, template=PLOT_THEME)
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    # ── 5. Categorical breakdown ───────────────────────────────────────────────
    st.markdown('<div class="section-header">5 · Categorical Breakdown</div>', unsafe_allow_html=True)
    if cat_cols:
        cat_sel = st.selectbox("Category", cat_cols)
        num_sel = st.selectbox("Numeric metric", num_cols, index=num_cols.index("Salary_LPA") if "Salary_LPA" in num_cols else 0)
        fig = px.box(df, x=cat_sel, y=num_sel, color=cat_sel,
                     color_discrete_sequence=COLOR_SEQ,
                     title=f"{num_sel} by {cat_sel}", template=PLOT_THEME)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, width="stretch")
