import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

PLOT_THEME = "plotly_dark"
COLOR_SEQ = px.colors.qualitative.Bold

def guard():
    if st.session_state.df is None:
        st.warning("⚠️ Please load a dataset first on the **Data Loading** page.")
        st.stop()
    return st.session_state.df

def detect_outliers_iqr(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < lb) | (series > ub), lb, ub

def detect_outliers_sigma(series, k=3):
    m, s = series.mean(), series.std()
    return (series < m - k * s) | (series > m + k * s), m - k * s, m + k * s

def show():
    df = guard()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.title("🔍 Diagnostic Analysis")
    st.markdown("Correlation matrices, statistical tests, and anomaly detection to understand *why* patterns exist in the data.")

    # ── 1. Correlation matrix ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">1 · Correlation Matrix (Pearson)</div>', unsafe_allow_html=True)

    corr = df[num_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        hoverongaps=False
    ))
    fig.update_layout(title="Pearson Correlation Matrix",
                      template=PLOT_THEME, height=600,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

    # Top correlations with target
    if "Placement_Status" in df.columns:
        target_corr = corr["Placement_Status"].drop("Placement_Status").sort_values(key=abs, ascending=False)
        st.markdown("**Top correlations with Placement_Status:**")
        fig2 = px.bar(
            x=target_corr.values, y=target_corr.index,
            orientation="h",
            color=target_corr.values,
            color_continuous_scale="RdBu",
            range_color=[-1, 1],
            title="Feature Correlation with Placement Status",
            template=PLOT_THEME
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           yaxis_title="", xaxis_title="Pearson r", height=400)
        st.plotly_chart(fig2, width="stretch")

    # ── 2. Pairwise correlation explorer ──────────────────────────────────────
    st.markdown('<div class="section-header">2 · Pairwise Correlation Explorer</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        va = st.selectbox("Variable A", num_cols, index=0)
    with c2:
        vb = st.selectbox("Variable B", num_cols, index=min(4, len(num_cols)-1))

    r, pval = stats.pearsonr(df[va], df[vb])
    rho, pval_sp = stats.spearmanr(df[va], df[vb])

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Pearson r", f"{r:.4f}")
    col_m2.metric("p-value (Pearson)", f"{pval:.2e}")
    col_m3.metric("Spearman ρ", f"{rho:.4f}")
    col_m4.metric("p-value (Spearman)", f"{pval_sp:.2e}")

    sig = pval < 0.05
    if sig:
        st.markdown(f'<div class="success-box">✅ Statistically significant correlation at α=0.05 (p={pval:.4f})</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn-box">❌ Not significant at α=0.05 (p={pval:.4f})</div>', unsafe_allow_html=True)

    fig3 = px.scatter(df.sample(min(2000, len(df)), random_state=7),
                      x=va, y=vb, opacity=0.4,
                      trendline="ols",
                      color_discrete_sequence=["#58a6ff"],
                      template=PLOT_THEME,
                      title=f"{va} vs {vb}  (r={r:.3f})")
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, width="stretch")

    # ── 3. Statistical tests ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">3 · Statistical Tests</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["T-test (placed vs not)", "Normality (Shapiro-Wilk)", "ANOVA by College Tier"])

    with tab1:
        if "Salary_LPA" in df.columns:
            sel_num = st.selectbox("Numeric variable", num_cols,
                                   index=num_cols.index("CGPA") if "CGPA" in num_cols else 0,
                                   key="ttest_col")
            median_sal = df["Salary_LPA"].median()
            df_ttest = df.copy()
            df_ttest["Salary_Group"] = (df_ttest["Salary_LPA"] >= median_sal).map(
                {True: "High Salary", False: "Low Salary"})
            g1 = df_ttest[df_ttest["Salary_LPA"] >= median_sal][sel_num].dropna()
            g0 = df_ttest[df_ttest["Salary_LPA"] < median_sal][sel_num].dropna()
            t, p = stats.ttest_ind(g1, g0)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("t-statistic", f"{t:.4f}")
            col_b.metric("p-value", f"{p:.2e}")
            col_c.metric("Significant (a=0.05)", "Yes" if p < 0.05 else "No")
            fig_t = px.violin(df_ttest, x="Salary_Group", y=sel_num,
                              box=True, color="Salary_Group",
                              color_discrete_sequence=["#f78166", "#3fb950"],
                              template=PLOT_THEME,
                              title=f"{sel_num} — High vs Low Salary group")
            fig_t.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_t, width="stretch")

    with tab2:
        sel_norm = st.selectbox("Variable", num_cols, key="norm_col")
        sample_sw = df[sel_norm].dropna().sample(min(5000, len(df)), random_state=1)
        stat_sw, p_sw = stats.shapiro(sample_sw)
        st.metric("Shapiro-Wilk W", f"{stat_sw:.4f}")
        st.metric("p-value", f"{p_sw:.4e}")
        if p_sw > 0.05:
            st.markdown('<div class="success-box">✅ Cannot reject normality at α=0.05</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warn-box">⚠️ Reject normality at α=0.05 — distribution is non-normal</div>', unsafe_allow_html=True)

        # QQ plot
        (osm, osr), (slope, intercept, r) = stats.probplot(df[sel_norm].dropna())
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                                    marker=dict(color="#58a6ff", size=3, opacity=0.4),
                                    name="Data"))
        line_x = np.array([min(osm), max(osm)])
        fig_qq.add_trace(go.Scatter(x=line_x, y=slope * line_x + intercept,
                                    mode="lines", line=dict(color="#f78166", width=2),
                                    name="Normal reference"))
        fig_qq.update_layout(title=f"Q-Q Plot · {sel_norm}", template=PLOT_THEME,
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles")
        st.plotly_chart(fig_qq, width="stretch")

    with tab3:
        if "College_Tier" in df.columns:
            anova_var = st.selectbox("Metric", num_cols,
                                     index=num_cols.index("Salary_LPA") if "Salary_LPA" in num_cols else 0,
                                     key="anova_col")
            groups = [grp[anova_var].dropna() for _, grp in df.groupby("College_Tier")]
            f_stat, p_anova = stats.f_oneway(*groups)
            col_a, col_b = st.columns(2)
            col_a.metric("F-statistic", f"{f_stat:.4f}")
            col_b.metric("p-value", f"{p_anova:.4e}")
            if p_anova < 0.05:
                st.markdown('<div class="success-box">✅ Significant difference between college tiers (α=0.05)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warn-box">❌ No significant difference detected</div>', unsafe_allow_html=True)

            fig_an = px.box(df, x="College_Tier", y=anova_var, color="College_Tier",
                            color_discrete_sequence=COLOR_SEQ, template=PLOT_THEME,
                            title=f"{anova_var} by College Tier")
            fig_an.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_an, width="stretch")

    # ── 4. Anomaly detection ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">4 · Anomaly Detection</div>', unsafe_allow_html=True)

    anom_col = st.selectbox("Variable for anomaly detection", num_cols, key="anom_col")
    method = st.radio("Method", ["IQR (interquartile range)", "3-Sigma rule"], horizontal=True)

    if method == "IQR (interquartile range)":
        mask, lb, ub = detect_outliers_iqr(df[anom_col])
        method_label = "IQR"
    else:
        mask, lb, ub = detect_outliers_sigma(df[anom_col])
        method_label = "3-Sigma"

    n_outliers = mask.sum()
    pct_out = n_outliers / len(df) * 100

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Outliers detected", f"{n_outliers:,}")
    col_b.metric("Percentage", f"{pct_out:.2f}%")
    col_c.metric("Method", method_label)

    fig_anom = go.Figure()
    normal_data = df[~mask][anom_col]
    outlier_data = df[mask][anom_col]

    fig_anom.add_trace(go.Scatter(
        y=df[anom_col].values,
        mode="markers",
        marker=dict(color=np.where(mask, "#f78166", "#58a6ff"), size=3, opacity=0.6),
        name="Data points"
    ))
    fig_anom.add_hline(y=lb, line_dash="dash", line_color="#d2a8ff",
                       annotation_text=f"Lower bound ({lb:.2f})")
    fig_anom.add_hline(y=ub, line_dash="dash", line_color="#d2a8ff",
                       annotation_text=f"Upper bound ({ub:.2f})")
    fig_anom.update_layout(title=f"Anomaly Detection · {anom_col} ({method_label})",
                            template=PLOT_THEME,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            xaxis_title="Row index", yaxis_title=anom_col)
    st.plotly_chart(fig_anom, width="stretch")

    if n_outliers > 0:
        with st.expander(f"🔎 Show {min(n_outliers, 50)} outlier rows"):
            st.dataframe(df[mask].head(50), width="stretch")

    # Multi-column anomaly summary
    st.markdown("**Anomaly count per numeric column (IQR method):**")
    anom_summary = []
    for col in num_cols:
        m, lo, hi = detect_outliers_iqr(df[col])
        anom_summary.append({"Column": col, "Outliers": int(m.sum()), "Pct": round(m.mean()*100, 2),
                              "Lower bound": round(lo, 3), "Upper bound": round(hi, 3)})
    anom_df = pd.DataFrame(anom_summary).sort_values("Outliers", ascending=False)
    st.dataframe(anom_df, width="stretch", hide_index=True)
