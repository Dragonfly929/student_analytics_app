import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, r2_score, mean_absolute_error, mean_squared_error
)
import warnings
warnings.filterwarnings("ignore")

PLOT_THEME = "plotly_dark"
COLOR_SEQ = px.colors.qualitative.Bold

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

def show():
    df = guard()

    # Derive binary target: High Salary = above median
    median_sal = df["Salary_LPA"].median()
    df = df.copy()
    df["High_Salary"] = (df["Salary_LPA"] >= median_sal).astype(int)

    st.title("🤖 Predictive Analysis")
    st.markdown(f"""
    Train **classification** and **regression** models, evaluate performance, and apply to new inputs.

    > **Dataset note:** All 20,000 students in this dataset are placed.
    > For classification, we predict **High vs Low salary** (threshold: median ≥ {median_sal:.2f} LPA).
    > Regression predicts the exact salary value.
    """)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    tab_cls, tab_reg = st.tabs(["🎯 Classification · High vs Low Salary", "💰 Regression · Salary Prediction"])

    # ═══════════════════════════════════════════════════════════════
    # CLASSIFICATION
    # ═══════════════════════════════════════════════════════════════
    with tab_cls:
        st.markdown(f'<div class="section-header">Predict: High Salary (≥{median_sal:.2f} LPA) vs Low Salary</div>', unsafe_allow_html=True)

        all_cols = [c for c in df.columns if c not in ["High_Salary", "Salary_LPA"]]
        default_feats = ["CGPA", "Mock_Interview_Score", "Resume_Score",
                         "Aptitude_Test_Score", "Communication_Skills",
                         "Internships", "Projects_Count", "Certifications",
                         "DSA_Problems_Solved", "LeetCode_Rating",
                         "GitHub_Contributions", "Hackathons_Participated",
                         "AI_ML_Skill_Level", "System_Design_Knowledge"]
        feat_cls = st.multiselect(
            "Features",
            [c for c in all_cols if c in df.select_dtypes(include=np.number).columns],
            default=[c for c in default_feats if c in df.columns],
            key="cls_feats"
        )

        model_name = st.selectbox("Algorithm", ["Random Forest", "Logistic Regression"], key="cls_algo")
        test_size = st.slider("Test set %", 10, 40, 20, key="cls_ts") / 100

        if st.button("🚀 Train classifier", key="train_cls"):
            with st.spinner("Training…"):
                df2 = preprocess(df)
                X = df2[feat_cls].fillna(df2[feat_cls].median())
                y = df2["High_Salary"]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y)

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                if model_name == "Random Forest":
                    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
                else:
                    clf = LogisticRegression(max_iter=1000, random_state=42)

                clf.fit(X_train_s, y_train)
                y_pred = clf.predict(X_test_s)
                y_prob = clf.predict_proba(X_test_s)[:, 1]

                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_prob)
                cv_scores = cross_val_score(clf, scaler.transform(X.fillna(X.median())), y, cv=5)

                st.session_state["clf_model"] = clf
                st.session_state["clf_scaler"] = scaler
                st.session_state["clf_features"] = feat_cls
                st.session_state["clf_median_sal"] = median_sal

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc:.4f}")
            m2.metric("AUC-ROC", f"{auc:.4f}")
            m3.metric("CV Accuracy (5-fold)", f"{cv_scores.mean():.4f}")
            m4.metric("CV Std", f"{cv_scores.std():.4f}")

            col_l, col_r = st.columns(2)
            with col_l:
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = go.Figure(go.Heatmap(
                    z=cm, x=["Low Salary", "High Salary"], y=["Low Salary", "High Salary"],
                    colorscale="Blues", text=cm, texttemplate="%{text}", showscale=False
                ))
                fig_cm.update_layout(title="Confusion Matrix", template=PLOT_THEME,
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                     xaxis_title="Predicted", yaxis_title="Actual", height=350)
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_r:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                             name=f"AUC={auc:.3f}",
                                             line=dict(color="#58a6ff", width=2)))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                             line=dict(dash="dash", color="#8b949e"), name="Random"))
                fig_roc.update_layout(title="ROC Curve", template=PLOT_THEME,
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      xaxis_title="FPR", yaxis_title="TPR", height=350)
                st.plotly_chart(fig_roc, use_container_width=True)

            if model_name == "Random Forest":
                fi = pd.DataFrame({"Feature": feat_cls, "Importance": clf.feature_importances_})
                fi = fi.sort_values("Importance", ascending=True)
                fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                                color="Importance", color_continuous_scale="Blues",
                                title="Feature Importances", template=PLOT_THEME)
                fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                     height=420, showlegend=False)
                st.plotly_chart(fig_fi, use_container_width=True)

            st.text(classification_report(y_test, y_pred, target_names=["Low Salary", "High Salary"]))

        # Apply to new student
        if "clf_model" in st.session_state and "clf_scaler" in st.session_state and "clf_features" in st.session_state and "clf_median_sal" in st.session_state:
            st.markdown('<div class="section-header">🔮 Predict for New Student</div>', unsafe_allow_html=True)
            clf = st.session_state["clf_model"]
            scaler = st.session_state["clf_scaler"]
            features = st.session_state["clf_features"]
            med = st.session_state["clf_median_sal"]

            input_vals = {}
            cols_form = st.columns(3)
            for i, feat in enumerate(features):
                with cols_form[i % 3]:
                    mn = float(df[feat].min())
                    mx = float(df[feat].max())
                    mid = float(df[feat].mean())
                    input_vals[feat] = st.number_input(feat, min_value=mn, max_value=mx,
                                                       value=round(mid, 2), key=f"cls_inp_{feat}")

            if st.button("📊 Predict Salary Category", key="btn_predict_cls"):
                row = np.array([[input_vals[f] for f in features]])
                row_s = scaler.transform(row)
                pred = clf.predict(row_s)[0]
                prob = clf.predict_proba(row_s)[0]
                label = f"HIGH salary (≥{med:.2f} LPA)" if pred == 1 else f"LOW salary (<{med:.2f} LPA)"
                color = "success-box" if pred == 1 else "warn-box"
                st.markdown(f'<div class="{color}"><b>{"✅" if pred==1 else "⚠️"} Predicted: {label}</b><br>Confidence: {max(prob):.2%}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # REGRESSION
    # ═══════════════════════════════════════════════════════════════
    with tab_reg:
        st.markdown('<div class="section-header">Predict Exact Salary (LPA)</div>', unsafe_allow_html=True)

        all_reg_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in ["Salary_LPA", "High_Salary"]]
        feat_reg = st.multiselect(
            "Features for regression",
            all_reg_cols,
            default=[c for c in ["CGPA", "Mock_Interview_Score", "Resume_Score",
                                  "Aptitude_Test_Score", "LeetCode_Rating",
                                  "Internships", "Projects_Count",
                                  "AI_ML_Skill_Level", "System_Design_Knowledge"] if c in all_reg_cols],
            key="reg_feats"
        )

        reg_model_name = st.selectbox("Algorithm", ["Random Forest", "Linear Regression"], key="reg_algo")
        reg_test_size = st.slider("Test set %", 10, 40, 20, key="reg_ts") / 100

        if st.button("🚀 Train regressor", key="train_reg"):
            with st.spinner("Training…"):
                X = df[feat_reg].fillna(df[feat_reg].median())
                y = df["Salary_LPA"]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=reg_test_size, random_state=42)
                scaler_r = StandardScaler()
                X_train_s = scaler_r.fit_transform(X_train)
                X_test_s = scaler_r.transform(X_test)

                if reg_model_name == "Random Forest":
                    reg = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
                else:
                    reg = LinearRegression()

                reg.fit(X_train_s, y_train)
                y_pred_r = reg.predict(X_test_s)

                r2 = r2_score(y_test, y_pred_r)
                mae = mean_absolute_error(y_test, y_pred_r)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_r))

                st.session_state["reg_model"] = reg
                st.session_state["reg_scaler"] = scaler_r
                st.session_state["reg_features"] = feat_reg
                st.session_state["reg_df"] = df

            m1, m2, m3 = st.columns(3)
            m1.metric("R² Score", f"{r2:.4f}")
            m2.metric("MAE (LPA)", f"{mae:.4f}")
            m3.metric("RMSE (LPA)", f"{rmse:.4f}")

            col_l, col_r = st.columns(2)
            with col_l:
                fig_avp = go.Figure()
                fig_avp.add_trace(go.Scatter(x=y_test.values, y=y_pred_r, mode="markers",
                                             marker=dict(color="#58a6ff", size=3, opacity=0.35), name="Predictions"))
                mn_v, mx_v = min(y_test.min(), y_pred_r.min()), max(y_test.max(), y_pred_r.max())
                fig_avp.add_trace(go.Scatter(x=[mn_v, mx_v], y=[mn_v, mx_v], mode="lines",
                                             line=dict(color="#f78166", dash="dash"), name="Perfect fit"))
                fig_avp.update_layout(title="Actual vs Predicted Salary", template=PLOT_THEME,
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      xaxis_title="Actual (LPA)", yaxis_title="Predicted (LPA)")
                st.plotly_chart(fig_avp, use_container_width=True)

            with col_r:
                residuals = y_test.values - y_pred_r
                fig_res = px.histogram(x=residuals, nbins=50, title="Residuals Distribution",
                                       color_discrete_sequence=["#3fb950"], template=PLOT_THEME)
                fig_res.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      xaxis_title="Residual", yaxis_title="Count")
                st.plotly_chart(fig_res, use_container_width=True)

            if reg_model_name == "Random Forest":
                fi_r = pd.DataFrame({"Feature": feat_reg, "Importance": reg.feature_importances_})
                fi_r = fi_r.sort_values("Importance", ascending=True)
                fig_fi_r = px.bar(fi_r, x="Importance", y="Feature", orientation="h",
                                  color="Importance", color_continuous_scale="Greens",
                                  title="Feature Importances (Salary)", template=PLOT_THEME)
                fig_fi_r.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       height=380, showlegend=False)
                st.plotly_chart(fig_fi_r, use_container_width=True)

        if "reg_model" in st.session_state and "reg_scaler" in st.session_state:
            st.markdown('<div class="section-header">🔮 Predict Salary for New Student</div>', unsafe_allow_html=True)
            reg = st.session_state["reg_model"]
            scaler_r = st.session_state["reg_scaler"]
            features_r = st.session_state["reg_features"]
            df_ref = st.session_state["reg_df"]

            input_r = {}
            cols_r = st.columns(3)
            for i, feat in enumerate(features_r):
                with cols_r[i % 3]:
                    mn = float(df_ref[feat].min())
                    mx = float(df_ref[feat].max())
                    mid = float(df_ref[feat].mean())
                    input_r[feat] = st.number_input(feat, min_value=mn, max_value=mx,
                                                    value=round(mid, 2), key=f"reg_inp_{feat}")

            if st.button("📊 Predict Salary", key="btn_predict_reg"):
                row_r = np.array([[input_r[f] for f in features_r]])
                row_r_s = scaler_r.transform(row_r)
                sal_pred = reg.predict(row_r_s)[0]
                st.markdown(f'<div class="success-box"><b>💰 Predicted Salary: {sal_pred:.2f} LPA</b></div>', unsafe_allow_html=True)
