import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

PLOT_THEME = "plotly_dark"

def guard():
    if st.session_state.df is None:
        st.warning("⚠️ Please load a dataset first on the **Data Loading** page.")
        st.stop()
    return st.session_state.df

def preprocess(df):
    df2 = df.copy()
    le = LabelEncoder()
    for col in df2.select_dtypes(exclude=np.number).columns:
        df2[col] = le.fit_transform(df2[col].astype(str))
    return df2

FEATURES = ["CGPA", "Mock_Interview_Score", "Resume_Score", "Aptitude_Test_Score",
            "Communication_Skills", "Internships", "Projects_Count", "Certifications",
            "DSA_Problems_Solved", "LeetCode_Rating", "GitHub_Contributions",
            "Hackathons_Participated", "AI_ML_Skill_Level", "System_Design_Knowledge"]

SAL_FEATURES = ["CGPA", "Mock_Interview_Score", "Resume_Score", "Aptitude_Test_Score",
                "LeetCode_Rating", "Internships", "Projects_Count",
                "AI_ML_Skill_Level", "System_Design_Knowledge"]

def get_models(df):
    key = "presc_models"
    if key not in st.session_state:
        with st.spinner("Building models for simulation…"):
            df2 = preprocess(df)
            median_sal = df["Salary_LPA"].median()
            df2["High_Salary"] = (df["Salary_LPA"] >= median_sal).astype(int)

            feat = [f for f in FEATURES if f in df2.columns]
            feat_sal = [f for f in SAL_FEATURES if f in df2.columns]

            X = df2[feat].fillna(df2[feat].median())
            y_cls = df2["High_Salary"]
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X_s, y_cls)

            X_sal = df2[feat_sal].fillna(df2[feat_sal].median())
            y_sal = df2["Salary_LPA"]
            scaler_sal = StandardScaler()
            X_sal_s = scaler_sal.fit_transform(X_sal)
            reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            reg.fit(X_sal_s, y_sal)

            st.session_state[key] = (clf, scaler, feat, reg, scaler_sal, feat_sal, median_sal)

    return st.session_state[key]

def predict_profile(clf, scaler, feat, reg, scaler_sal, feat_sal, profile):
    row = np.array([[profile.get(f, 0) for f in feat]])
    row_s = scaler.transform(row)
    prob = clf.predict_proba(row_s)[0][1]
    row_sal = np.array([[profile.get(f, 0) for f in feat_sal]])
    row_sal_s = scaler_sal.transform(row_sal)
    sal = reg.predict(row_sal_s)[0]
    return prob, sal

def show():
    df = guard()
    st.title("🧭 Prescriptive Analysis")
    st.markdown("""
    Prescriptive analysis answers **"What should we do?"** — simulating scenarios,
    comparing strategies, and recommending actions to maximize student salary outcomes.
    """)

    clf, scaler, feat, reg, scaler_sal, feat_sal, median_sal = get_models(df)

    tab1, tab2, tab3 = st.tabs(["🎯 Student Profile Optimizer",
                                  "📈 What-If Scenario Simulator",
                                  "🏫 Institutional ROI Analysis"])

    # ═══ TAB 1: Student Profile Optimizer ════════════════════════
    with tab1:
        st.markdown('<div class="section-header">Recommend skill improvements for a specific student</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            cgpa = st.slider("CGPA", 4.0, 10.0, 7.5, 0.1)
            mock = st.slider("Mock Interview Score", 0, 100, 55)
            resume = st.slider("Resume Score", 0, 100, 55)
            apt = st.slider("Aptitude Test Score", 0, 100, 60)
            comm = st.slider("Communication Skills", 0, 100, 60)
        with col2:
            intern = st.number_input("Internships", 0, 10, 1)
            proj = st.number_input("Projects Count", 0, 20, 3)
            cert = st.number_input("Certifications", 0, 15, 3)
            dsa = st.number_input("DSA Problems", 0, 1000, 150, 10)
        with col3:
            lc = st.number_input("LeetCode Rating", 0, 3500, 1600, 50)
            gh = st.number_input("GitHub Contributions", 0, 1000, 100, 10)
            hack = st.number_input("Hackathons", 0, 20, 2)
            ai_ml = st.slider("AI/ML Skill Level", 1, 10, 5)
            sys_d = st.slider("System Design", 1, 10, 5)

        base = {"CGPA": cgpa, "Mock_Interview_Score": mock, "Resume_Score": resume,
                "Aptitude_Test_Score": apt, "Communication_Skills": comm,
                "Internships": intern, "Projects_Count": proj, "Certifications": cert,
                "DSA_Problems_Solved": dsa, "LeetCode_Rating": lc,
                "GitHub_Contributions": gh, "Hackathons_Participated": hack,
                "AI_ML_Skill_Level": ai_ml, "System_Design_Knowledge": sys_d}

        prob_base, sal_base = predict_profile(clf, scaler, feat, reg, scaler_sal, feat_sal, base)

        m1, m2, m3 = st.columns(3)
        m1.metric("High-Salary Probability", f"{prob_base:.2%}")
        m2.metric("Est. Salary", f"{sal_base:.2f} LPA")
        m3.metric("Category", f"{'HIGH ≥' if prob_base >= 0.5 else 'LOW <'} {median_sal:.1f} LPA")

        st.markdown("#### Impact of +20% improvement per skill")
        improvements = []
        for feature in feat:
            val = base.get(feature, 0)
            improved = base.copy()
            if feature == "CGPA":
                improved[feature] = min(10.0, val * 1.2)
            else:
                improved[feature] = min(val * 1.2, float(df[feature].max()) if feature in df.columns else val * 1.5)

            prob_i, sal_i = predict_profile(clf, scaler, feat, reg, scaler_sal, feat_sal, improved)
            improvements.append({
                "Feature": feature,
                "ΔHigh-Salary Prob (pp)": round((prob_i - prob_base) * 100, 2),
                "ΔSalary (LPA)": round(sal_i - sal_base, 3),
            })

        imp_df = pd.DataFrame(improvements).sort_values("ΔHigh-Salary Prob (pp)", ascending=False)

        colors = ["#3fb950" if v >= 0 else "#f78166" for v in imp_df["ΔHigh-Salary Prob (pp)"]]
        fig_imp = go.Figure(go.Bar(x=imp_df["Feature"], y=imp_df["ΔHigh-Salary Prob (pp)"],
                                   marker_color=colors))
        fig_imp.update_layout(title="Placement Category Improvement per Skill (+20%)",
                              template=PLOT_THEME, xaxis_tickangle=-35,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              yaxis_title="Δ Probability (pp)")
        st.plotly_chart(fig_imp, width="stretch")

        top3 = imp_df.head(3)
        for _, row_r in top3.iterrows():
            if row_r["ΔHigh-Salary Prob (pp)"] > 0:
                st.markdown(f'<div class="success-box">⬆️ <b>{row_r["Feature"]}</b> → +{row_r["ΔHigh-Salary Prob (pp)"]:.1f} pp high-salary probability · +{row_r["ΔSalary (LPA)"]:.2f} LPA salary</div>', unsafe_allow_html=True)

        with st.expander("📋 Full improvement table"):
            st.dataframe(imp_df, width="stretch", hide_index=True)

    # ═══ TAB 2: What-If Scenario Simulator ══════════════════════
    with tab2:
        st.markdown('<div class="section-header">Compare multiple student development strategies</div>', unsafe_allow_html=True)

        n_sc = st.slider("Number of scenarios", 2, 4, 3)
        defaults = [
            {"label": "🎓 Academic Focus", "CGPA": 9.2, "Mock_Interview_Score": 55, "Resume_Score": 58,
             "Aptitude_Test_Score": 65, "Communication_Skills": 60, "Internships": 1,
             "Projects_Count": 2, "Certifications": 4, "DSA_Problems_Solved": 180,
             "LeetCode_Rating": 1580, "GitHub_Contributions": 70, "Hackathons_Participated": 1,
             "AI_ML_Skill_Level": 5, "System_Design_Knowledge": 4},
            {"label": "💼 Industry Ready", "CGPA": 7.5, "Mock_Interview_Score": 87, "Resume_Score": 88,
             "Aptitude_Test_Score": 82, "Communication_Skills": 88, "Internships": 4,
             "Projects_Count": 9, "Certifications": 9, "DSA_Problems_Solved": 420,
             "LeetCode_Rating": 1950, "GitHub_Contributions": 350, "Hackathons_Participated": 6,
             "AI_ML_Skill_Level": 8, "System_Design_Knowledge": 9},
            {"label": "⚖️ Balanced", "CGPA": 8.3, "Mock_Interview_Score": 72, "Resume_Score": 74,
             "Aptitude_Test_Score": 73, "Communication_Skills": 74, "Internships": 2,
             "Projects_Count": 5, "Certifications": 6, "DSA_Problems_Solved": 300,
             "LeetCode_Rating": 1750, "GitHub_Contributions": 200, "Hackathons_Participated": 3,
             "AI_ML_Skill_Level": 7, "System_Design_Knowledge": 6},
            {"label": "🤖 AI/ML Specialist", "CGPA": 8.0, "Mock_Interview_Score": 68, "Resume_Score": 70,
             "Aptitude_Test_Score": 78, "Communication_Skills": 65, "Internships": 2,
             "Projects_Count": 12, "Certifications": 11, "DSA_Problems_Solved": 360,
             "LeetCode_Rating": 1860, "GitHub_Contributions": 550, "Hackathons_Participated": 8,
             "AI_ML_Skill_Level": 10, "System_Design_Knowledge": 9},
        ]

        results = []
        cols_sc = st.columns(n_sc)
        for i, col in enumerate(cols_sc):
            d = defaults[i] if i < len(defaults) else defaults[-1]
            with col:
                label = st.text_input("Name", d["label"], key=f"sc_name_{i}")
                cgpa_s = st.slider("CGPA", 4.0, 10.0, float(d["CGPA"]), 0.1, key=f"sc_cgpa_{i}")
                mock_s = st.slider("Mock Score", 0, 100, d["Mock_Interview_Score"], key=f"sc_mock_{i}")
                intern_s = st.number_input("Internships", 0, 10, d["Internships"], key=f"sc_int_{i}")
                proj_s = st.number_input("Projects", 0, 20, d["Projects_Count"], key=f"sc_proj_{i}")
                lc_s = st.number_input("LeetCode", 0, 3500, d["LeetCode_Rating"], 50, key=f"sc_lc_{i}")
                ai_s = st.slider("AI/ML Level", 1, 10, d["AI_ML_Skill_Level"], key=f"sc_ai_{i}")

                profile_s = {**d, "CGPA": cgpa_s, "Mock_Interview_Score": mock_s,
                             "Internships": intern_s, "Projects_Count": proj_s,
                             "LeetCode_Rating": lc_s, "AI_ML_Skill_Level": ai_s}
                prob_s, sal_s = predict_profile(clf, scaler, feat, reg, scaler_sal, feat_sal, profile_s)
                results.append({
                    "Scenario": label,
                    "High-Salary Prob (%)": round(prob_s * 100, 1),
                    "Predicted Category": f"HIGH ≥{median_sal:.1f}" if prob_s >= 0.5 else f"LOW <{median_sal:.1f}",
                    "Est. Salary (LPA)": round(sal_s, 2)
                })

        res_df = pd.DataFrame(results)
        st.markdown("---")
        st.dataframe(res_df, width="stretch", hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.bar(res_df, x="Scenario", y="High-Salary Prob (%)",
                          color="High-Salary Prob (%)", color_continuous_scale="Blues",
                          title="High-Salary Probability by Scenario", template=PLOT_THEME)
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig1, width="stretch")
        with c2:
            fig2 = px.bar(res_df, x="Scenario", y="Est. Salary (LPA)",
                          color="Est. Salary (LPA)", color_continuous_scale="Greens",
                          title="Expected Salary by Scenario", template=PLOT_THEME)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig2, width="stretch")

        best = res_df.loc[res_df["Est. Salary (LPA)"].idxmax()]
        st.markdown(f'<div class="success-box">🏆 Best outcome: <b>{best["Scenario"]}</b> → <b>{best["Est. Salary (LPA)"]:.2f} LPA</b> · High-salary prob <b>{best["High-Salary Prob (%)"]:.1f}%</b></div>', unsafe_allow_html=True)

    # ═══ TAB 3: Institutional ROI ════════════════════════════════
    with tab3:
        st.markdown('<div class="section-header">Which skill investment yields the most salary improvement institution-wide?</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            improve_feat = st.selectbox("Skill to boost college-wide", feat)
        with col_b:
            pct_boost = st.slider("Improvement % applied to all students", 5, 50, 20)

        if st.button("🔬 Run simulation"):
            with st.spinner("Simulating…"):
                df2 = preprocess(df)
                X_sal_base = df2[feat_sal].fillna(df2[feat_sal].median())
                X_sal_base_s = scaler_sal.transform(X_sal_base)
                base_sals = reg.predict(X_sal_base_s)
                avg_base = base_sals.mean()

                X_boosted = X_sal_base.copy()
                if improve_feat in X_boosted.columns:
                    bf = 1 + pct_boost / 100
                    max_v = float(df[improve_feat].max()) if improve_feat in df.columns else X_boosted[improve_feat].max() * 1.5
                    if improve_feat == "CGPA":
                        X_boosted[improve_feat] = (X_boosted[improve_feat] * bf).clip(upper=10.0)
                    else:
                        X_boosted[improve_feat] = (X_boosted[improve_feat] * bf).clip(upper=max_v)

                X_boosted_s = scaler_sal.transform(X_boosted)
                boost_sals = reg.predict(X_boosted_s)
                avg_boost = boost_sals.mean()

            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Salary Before", f"{avg_base:.2f} LPA")
            m2.metric("Avg Salary After", f"{avg_boost:.2f} LPA")
            m3.metric("Average Gain", f"+{avg_boost - avg_base:.3f} LPA")

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=base_sals, name="Before", opacity=0.6,
                                             marker_color="#f78166", nbinsx=50))
            fig_dist.add_trace(go.Histogram(x=boost_sals, name="After", opacity=0.6,
                                             marker_color="#3fb950", nbinsx=50))
            fig_dist.update_layout(barmode="overlay",
                                   title=f"Salary Distribution Shift: {improve_feat} +{pct_boost}%",
                                   template=PLOT_THEME,
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   xaxis_title="Predicted Salary (LPA)", yaxis_title="Students")
            st.plotly_chart(fig_dist, width="stretch")

            # All-feature ROI
            roi_data = []
            for f_roi in feat:
                if f_roi not in feat_sal:
                    continue
                X_b = X_sal_base.copy()
                bf = 1 + pct_boost / 100
                max_v = float(df[f_roi].max()) if f_roi in df.columns else X_b[f_roi].max() * 1.5
                if f_roi == "CGPA":
                    X_b[f_roi] = (X_b[f_roi] * bf).clip(upper=10.0)
                else:
                    X_b[f_roi] = (X_b[f_roi] * bf).clip(upper=max_v)
                X_b_s = scaler_sal.transform(X_b)
                sals_b = reg.predict(X_b_s)
                roi_data.append({"Feature": f_roi,
                                  "Avg Salary Gain (LPA)": round(sals_b.mean() - avg_base, 4),
                                  "Students benefiting": int((sals_b > base_sals).sum())})

            roi_df = pd.DataFrame(roi_data).sort_values("Avg Salary Gain (LPA)", ascending=False)

            fig_roi = px.bar(roi_df, x="Feature", y="Avg Salary Gain (LPA)",
                             color="Avg Salary Gain (LPA)", color_continuous_scale="Viridis",
                             title=f"Salary Gain per Skill Investment ({pct_boost}% boost)",
                             template=PLOT_THEME)
            fig_roi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   xaxis_tickangle=-30, showlegend=False)
            st.plotly_chart(fig_roi, width="stretch")

            top_inst = roi_df.iloc[0]
            st.markdown(f'<div class="success-box">🏆 Best ROI: invest in <b>{top_inst["Feature"]}</b> → average salary gain of <b>+{top_inst["Avg Salary Gain (LPA)"]:.4f} LPA</b> per student</div>', unsafe_allow_html=True)

            with st.expander("📋 Full ROI table"):
                st.dataframe(roi_df, width="stretch", hide_index=True)
