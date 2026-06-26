import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
 
# ML Libraries
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, r2_score, accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.decomposition import PCA
 
# Page Config
st.set_page_config(
    page_title="Digital Addiction Pattern Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
#  Custom CSS 
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0e1117; }
 
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
 
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d1b69 100%);
        border: 1px solid #4a90e2;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(74,144,226,0.2);
    }
    [data-testid="metric-container"] label {
        color: #a0c4ff !important;
        font-size: 0.85rem !important;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #ffffff !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
 
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #4a90e2, #7b68ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 1rem 0 0.5rem 0;
    }
 
    /* Insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-left: 4px solid #4a90e2;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #c8d8e8;
        font-size: 0.9rem;
    }
 
    /* Risk badge */
    .risk-high {
        background: #c0392b; color: white;
        padding: 3px 10px; border-radius: 20px; font-weight: 600;
    }
    .risk-medium {
        background: #e67e22; color: white;
        padding: 3px 10px; border-radius: 20px; font-weight: 600;
    }
    .risk-low {
        background: #27ae60; color: white;
        padding: 3px 10px; border-radius: 20px; font-weight: 600;
    }
 
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #0e1117;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1a1a2e;
        border-radius: 8px 8px 0 0;
        color: #a0c4ff;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4a90e2, #7b68ee) !important;
        color: white !important;
    }
 
    /* Divider */
    hr { border-color: #2d3a4a; }
</style>
""", unsafe_allow_html=True)
 
# Load & Preprocess Data 
@st.cache_data
def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
    else:
        # Default sample data matching the uploaded dataset structure
        data = {
            "Hours Studied": [7,4,8,5,7,3,6,8,5,7,4,6,7,5,8],
            "Previous Scores": [99,82,51,52,75,45,80,77,70,85,65,78,82,60,72],
            "Extracurricular Activities": [1,0,1,1,0,0,1,0,0,1,0,1,0,1,0],
            "Sleep Hours": [9,4,7,5,8,4,6,7,5,8,4,7,6,5,9],
            "Productivity Score": [91,65,45,36,66,27,68,63,50,75,40,70,65,44,80],
            "Screen Time (hrs)": [2.7,7.5,4.7,5.5,4.2,7.6,4.3,5.1,6.0,3.8,7.0,4.8,5.2,6.5,3.5],
            "App Usage Count": [7,17,9,12,10,20,9,12,14,8,16,10,11,15,8],
        }
        df = pd.DataFrame(data)
    # Drop unnamed index column if present
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df

 
@st.cache_data
def engineer_features(df):
    df = df.copy()
    # Addiction Risk Score (higher screen time + more apps = more risk)
    df["Addiction Risk Score"] = (
        (df["Screen Time (hrs)"] / df["Screen Time (hrs)"].max()) * 40 +
        (df["App Usage Count"] / df["App Usage Count"].max()) * 30 +
        ((1 - df["Sleep Hours"] / df["Sleep Hours"].max()) * 20) +
        ((1 - df["Hours Studied"] / df["Hours Studied"].max()) * 10)
    ).round(2)
 
    # Risk Category
    def risk_cat(score):
        if score >= 60: return "High Risk"
        elif score >= 35: return "Medium Risk"
        else: return "Low Risk"
    df["Risk Category"] = df["Addiction Risk Score"].apply(risk_cat)
 
    # Digital Wellbeing Index (higher = healthier)
    df["Digital Wellbeing Index"] = (
        (df["Sleep Hours"] / df["Sleep Hours"].max()) * 35 +
        (df["Hours Studied"] / df["Hours Studied"].max()) * 30 +
        (df["Productivity Score"] / df["Productivity Score"].max()) * 35
    ).round(2)
 
    # Screen Time Category
    df["Screen Time Category"] = pd.cut(
        df["Screen Time (hrs)"],
        bins=[0, 3, 5, 7, 24],
        labels=["Low (<3h)", "Moderate (3-5h)", "High (5-7h)", "Very High (>7h)"]
    )
    return df
 
# Sidebar
with st.sidebar:
    st.markdown("##  Digital Addiction\n### Pattern Analyzer")
    st.markdown("---")
 
    uploaded = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
    df_raw = load_data(uploaded)
    df = engineer_features(df_raw)
 
    st.markdown("###  Analysis Settings")
    n_clusters = st.slider("KMeans Clusters", 2, 5, 3)
    test_size = st.slider("Test Split (%)", 10, 40, 20) / 100
    show_raw = st.checkbox("Show Raw Data Table", False)
 
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.markdown(f"- **Rows:** {len(df)}")
    st.markdown(f"- **Features:** {len(df.columns)}")
    high_risk = (df["Risk Category"] == "High Risk").sum()
    st.markdown(f"- **High Risk Users:** {high_risk}")
 
    st.markdown("---")
    st.caption("Built with Streamlit · scikit-learn · Plotly")
 
# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 1.5rem 0 1rem 0;'>
    <h1 style='background: linear-gradient(90deg, #4a90e2, #7b68ee, #e040fb);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               font-size: 2.4rem; font-weight: 800; margin-bottom: 0;'>
         Digital Addiction Pattern Analysis
    </h1>
    <p style='color: #7f8c8d; font-size: 1rem; margin-top: 0.3rem;'>
        ML-Powered Insights · Clustering · Predictive Modeling · Risk Assessment
    </p>
</div>
""", unsafe_allow_html=True)
 
st.markdown("---")
 
# KPI Metrics
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric(" Total Users", len(df))
with c2:
    st.metric(" Avg Screen Time", f"{df['Screen Time (hrs)'].mean():.1f} hrs")
with c3:
    st.metric(" Avg Productivity", f"{df['Productivity Score'].mean():.0f}")
with c4:
    st.metric(" Avg Sleep", f"{df['Sleep Hours'].mean():.1f} hrs")
with c5:
    st.metric(" High Risk %", f"{(df['Risk Category']=='High Risk').mean()*100:.0f}%")
 
st.markdown("---")
 
# Tabs 
tabs = st.tabs([
    " Overview",
    " Correlation Analysis",
    " ML Clustering",
    " Predictive Models",
    " Risk Assessment",
    " Predict New User"
])
 
# 
# TAB 1 — OVERVIEW
# 
with tabs[0]:
    st.markdown('<p class="section-header">📊 Dataset Overview & Distributions</p>', unsafe_allow_html=True)
 
    if show_raw:
        st.dataframe(df.style.background_gradient(cmap="Blues"), use_container_width=True)
 
    # Distribution plots
    num_cols = ["Screen Time (hrs)", "App Usage Count", "Sleep Hours",
                "Hours Studied", "Productivity Score", "Addiction Risk Score"]
 
    fig = make_subplots(rows=2, cols=3, subplot_titles=num_cols)
    colors = ["#4a90e2", "#e040fb", "#00bcd4", "#ff9800", "#66bb6a", "#ef5350"]
    for i, col in enumerate(num_cols):
        r, c = divmod(i, 3)
        fig.add_trace(
            go.Histogram(x=df[col], name=col, marker_color=colors[i],
                         opacity=0.8, showlegend=False),
            row=r+1, col=c+1
        )
    fig.update_layout(
        template="plotly_dark", height=450,
        title_text="Feature Distributions",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117"
    )
    st.plotly_chart(fig, use_container_width=True)
 
    # Screen Time vs Productivity scatter
    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.scatter(
            df, x="Screen Time (hrs)", y="Productivity Score",
            color="Risk Category", size="App Usage Count",
            color_discrete_map={"High Risk":"#ef5350","Medium Risk":"#ff9800","Low Risk":"#66bb6a"},
            title="Screen Time vs Productivity Score",
            template="plotly_dark", hover_data=["Sleep Hours","Hours Studied"]
        )
        fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22")
        st.plotly_chart(fig2, use_container_width=True)
 
    with col2:
        fig3 = px.pie(
            df, names="Risk Category",
            color="Risk Category",
            color_discrete_map={"High Risk":"#ef5350","Medium Risk":"#ff9800","Low Risk":"#66bb6a"},
            title="Addiction Risk Distribution",
            template="plotly_dark", hole=0.45
        )
        fig3.update_layout(paper_bgcolor="#0e1117")
        st.plotly_chart(fig3, use_container_width=True)
 
    # Screen time category bar
    cat_counts = df["Screen Time Category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    fig4 = px.bar(
        cat_counts, x="Category", y="Count",
        color="Category",
        color_discrete_sequence=["#66bb6a","#ff9800","#ef5350","#b71c1c"],
        title="Screen Time Category Distribution",
        template="plotly_dark"
    )
    fig4.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22", showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)
 
 

# TAB 2 — CORRELATION ANALYSIS

with tabs[1]:
    st.markdown('<p class="section-header"> Correlation & Relationship Analysis</p>', unsafe_allow_html=True)
 
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
 
    fig_heat = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Heatmap",
        template="plotly_dark"
    )
    fig_heat.update_layout(paper_bgcolor="#0e1117", height=500)
    st.plotly_chart(fig_heat, use_container_width=True)
 
    # Key correlations with Productivity Score
    st.markdown("####  Correlations with Productivity Score")
    prod_corr = corr["Productivity Score"].drop("Productivity Score").sort_values()
    colors_bar = ["#ef5350" if v < 0 else "#66bb6a" for v in prod_corr.values]
    fig_bar = go.Figure(go.Bar(
        x=prod_corr.values, y=prod_corr.index,
        orientation="h", marker_color=colors_bar,
    ))
    fig_bar.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
        title="Impact on Productivity Score",
        xaxis_title="Pearson r", height=350
    )
    st.plotly_chart(fig_bar, use_container_width=True)
 
    # Scatter matrix
    st.markdown("#### 📐 Pair Plot — Key Variables")
    fig_pair = px.scatter_matrix(
        df,
        dimensions=["Screen Time (hrs)", "App Usage Count",
                    "Sleep Hours", "Productivity Score"],
        color="Risk Category",
        color_discrete_map={"High Risk":"#ef5350","Medium Risk":"#ff9800","Low Risk":"#66bb6a"},
        template="plotly_dark"
    )
    fig_pair.update_traces(marker=dict(size=5, opacity=0.7))
    fig_pair.update_layout(paper_bgcolor="#0e1117", height=500)
    st.plotly_chart(fig_pair, use_container_width=True)
 
 

# TAB 3 — ML CLUSTERING

with tabs[2]:
    st.markdown('<p class="section-header"> Unsupervised Learning — KMeans Clustering</p>', unsafe_allow_html=True)
 
    # Features for clustering
    cluster_features = ["Screen Time (hrs)", "App Usage Count",
                        "Sleep Hours", "Hours Studied", "Productivity Score"]
    X_cluster = df[cluster_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
 
    # Elbow method
    inertias = []
    k_range = range(2, 7)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
 
    col1, col2 = st.columns([1, 2])
    with col1:
        fig_elbow = go.Figure(go.Scatter(
            x=list(k_range), y=inertias,
            mode="lines+markers", line=dict(color="#4a90e2", width=2),
            marker=dict(size=8, color="#e040fb")
        ))
        fig_elbow.update_layout(
            title="Elbow Method", xaxis_title="K", yaxis_title="Inertia",
            template="plotly_dark", paper_bgcolor="#0e1117",
            plot_bgcolor="#161b22", height=300
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
 
    # Run KMeans with selected k
    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = km_final.fit_predict(X_scaled).astype(str)
 
    # PCA for 2D visualization
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X_scaled)
    df["PCA1"] = pca_coords[:, 0]
    df["PCA2"] = pca_coords[:, 1]
 
    with col2:
        fig_pca = px.scatter(
            df, x="PCA1", y="PCA2", color="Cluster",
            symbol="Risk Category", size="Addiction Risk Score",
            title=f"KMeans Clusters (k={n_clusters}) — PCA Projection",
            template="plotly_dark",
            hover_data=["Screen Time (hrs)", "Sleep Hours", "Productivity Score"]
        )
        fig_pca.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22", height=300)
        st.plotly_chart(fig_pca, use_container_width=True)
 
    # Cluster profiles
    st.markdown("####  Cluster Profiles")
    cluster_profile = df.groupby("Cluster")[cluster_features + ["Addiction Risk Score"]].mean().round(2)
    st.dataframe(
        cluster_profile.style.background_gradient(cmap="YlOrRd", axis=0),
        use_container_width=True
    )
 
    # Radar chart per cluster
    st.markdown("####  Cluster Radar Chart")
    radar_features = ["Screen Time (hrs)", "App Usage Count", "Sleep Hours",
                      "Hours Studied", "Productivity Score"]
    cluster_means = df.groupby("Cluster")[radar_features].mean()
    # normalize 0-1
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-9)
 
    radar_colors = ["#4a90e2","#e040fb","#00bcd4","#ff9800","#66bb6a"]
    fig_radar = go.Figure()
    for i, (idx, row) in enumerate(cluster_means_norm.iterrows()):
        vals = list(row.values) + [row.values[0]]
        cats = radar_features + [radar_features[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            name=f"Cluster {idx}",
            line_color=radar_colors[i % len(radar_colors)], opacity=0.7
        ))
    fig_radar.update_layout(
        polar=dict(bgcolor="#161b22"),
        template="plotly_dark", paper_bgcolor="#0e1117",
        title="Cluster Behavioral Profiles (Normalized)", height=400
    )
    st.plotly_chart(fig_radar, use_container_width=True)
 
    st.markdown("""
    <div class="insight-box">
     <b>Insight:</b> Clusters reveal distinct behavioral archetypes —
    high screen time + low sleep clusters correlate with significantly reduced
    productivity and elevated addiction risk scores. Use cluster labels for
    targeted digital wellness interventions.
    </div>
    """, unsafe_allow_html=True)
 
 

# TAB 4 — PREDICTIVE MODELS

with tabs[3]:
    st.markdown('<p class="section-header">🎯 Predictive Modeling</p>', unsafe_allow_html=True)
 
    feat_cols = ["Screen Time (hrs)", "App Usage Count", "Sleep Hours",
                 "Hours Studied", "Extracurricular Activities", "Previous Scores"]
 
    # ── Model A: Predict Productivity Score (Regression) ──
    st.markdown("### 📈 Model A: Predict Productivity Score (Random Forest Regression)")
    X = df[feat_cols]
    y_reg = df["Productivity Score"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_reg, test_size=test_size, random_state=42)
 
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)
    y_pred_rf = rf.predict(X_te)
 
    # Linear Regression for comparison
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    y_pred_lr = lr.predict(X_te)
 
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RF — R² Score", f"{r2_score(y_te, y_pred_rf):.3f}")
    col2.metric("RF — MAE", f"{mean_absolute_error(y_te, y_pred_rf):.2f}")
    col3.metric("LR — R² Score", f"{r2_score(y_te, y_pred_lr):.3f}")
    col4.metric("LR — MAE", f"{mean_absolute_error(y_te, y_pred_lr):.2f}")
 
    # Actual vs Predicted
    pred_df = pd.DataFrame({
        "Index": range(len(y_te)),
        "Actual": y_te.values,
        "RF Predicted": y_pred_rf.round(1),
        "LR Predicted": y_pred_lr.round(1)
    })
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=pred_df["Index"], y=pred_df["Actual"],
                                   mode="lines+markers", name="Actual", line=dict(color="#66bb6a")))
    fig_pred.add_trace(go.Scatter(x=pred_df["Index"], y=pred_df["RF Predicted"],
                                   mode="lines+markers", name="RF Predicted", line=dict(color="#4a90e2", dash="dash")))
    fig_pred.add_trace(go.Scatter(x=pred_df["Index"], y=pred_df["LR Predicted"],
                                   mode="lines+markers", name="LR Predicted", line=dict(color="#e040fb", dash="dot")))
    fig_pred.update_layout(
        title="Actual vs Predicted — Productivity Score",
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
        height=320, xaxis_title="Test Sample", yaxis_title="Productivity Score"
    )
    st.plotly_chart(fig_pred, use_container_width=True)
 
    # Feature Importance
    st.markdown("####  Feature Importance (Random Forest)")
    fi_df = pd.DataFrame({
        "Feature": feat_cols,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=True)
 
    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="Blues",
                    template="plotly_dark", title="Feature Importance for Productivity Prediction")
    fig_fi.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22", height=300)
    st.plotly_chart(fig_fi, use_container_width=True)
 
    # ── Model B: Risk Category Classification ──
    st.markdown("---")
    st.markdown("###  Model B: Risk Category Classification (Gradient Boosting)")
 
    le = LabelEncoder()
    y_cls = le.fit_transform(df["Risk Category"])
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y_cls, test_size=test_size, random_state=42)
 
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_tr2, y_tr2)
    y_pred_cls = gb.predict(X_te2)
 
    acc = accuracy_score(y_te2, y_pred_cls)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(" Classification Accuracy", f"{acc*100:.1f}%")
        # Confusion matrix
        cm = confusion_matrix(y_te2, y_pred_cls)
        fig_cm = px.imshow(
            cm, 
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion Matrix",
            template="plotly_dark"
        )
        fig_cm.update_layout(paper_bgcolor="#0e1117", height=300)
        st.plotly_chart(fig_cm, use_container_width=True)
 
    with col2:
        # GB feature importance
        gb_fi = pd.DataFrame({
            "Feature": feat_cols,
            "Importance": gb.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig_gb_fi = px.bar(
            gb_fi, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Purples",
            title="Feature Importance for Risk Classification",
            template="plotly_dark"
        )
        fig_gb_fi.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22", height=300)
        st.plotly_chart(fig_gb_fi, use_container_width=True)
 
 
#
# TAB 5 — RISK ASSESSMENT
# 
with tabs[4]:
    st.markdown('<p class="section-header">⚠️ Digital Addiction Risk Assessment</p>', unsafe_allow_html=True)
 
    # Risk score distribution by category
    col1, col2 = st.columns(2)
    with col1:
        fig_box = px.box(
            df, x="Risk Category", y="Addiction Risk Score",
            color="Risk Category",
            color_discrete_map={"High Risk":"#ef5350","Medium Risk":"#ff9800","Low Risk":"#66bb6a"},
            title="Addiction Risk Score by Category",
            template="plotly_dark"
        )
        fig_box.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22")
        st.plotly_chart(fig_box, use_container_width=True)
 
    with col2:
        fig_violin = px.violin(
            df, x="Risk Category", y="Productivity Score",
            color="Risk Category",
            color_discrete_map={"High Risk":"#ef5350","Medium Risk":"#ff9800","Low Risk":"#66bb6a"},
            title="Productivity by Risk Category",
            template="plotly_dark", box=True
        )
        fig_violin.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22")
        st.plotly_chart(fig_violin, use_container_width=True)
 
    # Screen time vs sleep colored by risk
    fig_2d = px.scatter(
        df, x="Screen Time (hrs)", y="Sleep Hours",
        color="Risk Category", size="Addiction Risk Score",
        color_discrete_map={"High Risk":"#ef5350","Medium Risk":"#ff9800","Low Risk":"#66bb6a"},
        title="Screen Time vs Sleep Hours — Risk Zones",
        template="plotly_dark",
        hover_data=["Productivity Score", "App Usage Count"]
    )
    # add quadrant lines
    fig_2d.add_vline(x=df["Screen Time (hrs)"].mean(), line_dash="dash",
                     line_color="white", opacity=0.3)
    fig_2d.add_hline(y=df["Sleep Hours"].mean(), line_dash="dash",
                     line_color="white", opacity=0.3)
    fig_2d.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22", height=400)
    st.plotly_chart(fig_2d, use_container_width=True)
 
    # Risk Summary Table
    st.markdown("#### 📋 Risk Category Summary")
    risk_summary = df.groupby("Risk Category").agg(
        Users=("Addiction Risk Score", "count"),
        Avg_Screen_Time=("Screen Time (hrs)", "mean"),
        Avg_App_Usage=("App Usage Count", "mean"),
        Avg_Sleep=("Sleep Hours", "mean"),
        Avg_Productivity=("Productivity Score", "mean"),
        Avg_Risk_Score=("Addiction Risk Score", "mean")
    ).round(2)
    st.dataframe(risk_summary.style.background_gradient(cmap="RdYlGn_r"), use_container_width=True)
 
    # Wellbeing index
    st.markdown("####  Digital Wellbeing Index vs Addiction Risk Score")
    fig_wellbeing = px.scatter(
        df, x="Addiction Risk Score", y="Digital Wellbeing Index",
        color="Risk Category", size="Screen Time (hrs)",
        color_discrete_map={"High Risk":"#ef5350","Medium Risk":"#ff9800","Low Risk":"#66bb6a"},
        title="Addiction Risk vs Wellbeing Index",
        template="plotly_dark",
        trendline="ols"
    )
    fig_wellbeing.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22")
    st.plotly_chart(fig_wellbeing, use_container_width=True)
 
 
# 
# TAB 6 — PREDICT NEW USER
# 
with tabs[5]:
    st.markdown('<p class="section-header"> Predict New User Profile</p>', unsafe_allow_html=True)
    st.markdown("Enter a new user's data to predict their **Productivity Score** and **Addiction Risk**.")
 
    col1, col2, col3 = st.columns(3)
    with col1:
        inp_screen = st.slider(" Screen Time (hrs/day)", 1.0, 12.0, 5.0, 0.1)
        inp_app = st.slider(" App Usage Count", 1, 30, 12)
    with col2:
        inp_sleep = st.slider(" Sleep Hours", 3, 12, 7)
        inp_study = st.slider(" Hours Studied", 1, 10, 5)
    with col3:
        inp_extra = st.selectbox("🏃 Extracurricular Activities", [0, 1],
                                  format_func=lambda x: "Yes" if x else "No")
        inp_prev = st.slider(" Previous Scores", 30, 100, 70)
 
    if st.button(" Predict Profile", use_container_width=True):
        # Re-train on full data
        X_full = df[feat_cols]
        y_rf_full = df["Productivity Score"]
        rf_full = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_full.fit(X_full, y_rf_full)
 
        le_full = LabelEncoder()
        y_cls_full = le_full.fit_transform(df["Risk Category"])
        gb_full = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_full.fit(X_full, y_cls_full)
 
        new_user = np.array([[inp_screen, inp_app, inp_sleep,
                               inp_study, inp_extra, inp_prev]])
        pred_prod = rf_full.predict(new_user)[0]
        pred_risk_idx = gb_full.predict(new_user)[0]
        pred_risk = le_full.inverse_transform([pred_risk_idx])[0]
        pred_prob = gb_full.predict_proba(new_user)[0]
 
        # Compute derived scores
        pred_addiction_score = (
            (inp_screen / 12) * 40 +
            (inp_app / 30) * 30 +
            ((1 - inp_sleep / 12) * 20) +
            ((1 - inp_study / 10) * 10)
        )
 
        badge_color = {"High Risk":"risk-high","Medium Risk":"risk-medium","Low Risk":"risk-low"}[pred_risk]
 
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(" Predicted Productivity", f"{pred_prod:.0f} / 100")
        col2.metric(" Addiction Risk Score", f"{pred_addiction_score:.1f}")
        col3.metric(" Screen Time", f"{inp_screen:.1f} hrs")
        col4.metric(" Sleep Quality", f"{'Good ✅' if inp_sleep >= 7 else 'Poor'}")
 
        st.markdown(f"""
        <div class="insight-box" style="margin-top:1rem;">
            <b>Risk Category:</b> <span class="{badge_color}">{pred_risk}</span><br><br>
            <b>Classification Probabilities:</b><br>
            {"  ".join([f"<b>{le_full.classes_[i]}</b>: {p*100:.1f}%" for i, p in enumerate(pred_prob)])}
        </div>
        """, unsafe_allow_html=True)
 
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_prod,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Predicted Productivity Score", 'font': {'size': 18, 'color': 'white'}},
            delta={'reference': df["Productivity Score"].mean(), 'increasing': {'color': "#66bb6a"}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white'},
                'bar': {'color': "#4a90e2"},
                'bgcolor': "#161b22",
                'borderwidth': 2,
                'bordercolor': "#4a90e2",
                'steps': [
                    {'range': [0, 40], 'color': '#ef5350'},
                    {'range': [40, 70], 'color': '#ff9800'},
                    {'range': [70, 100], 'color': '#66bb6a'}
                ],
                'threshold': {
                    'line': {'color': "#e040fb", 'width': 4},
                    'thickness': 0.75,
                    'value': df["Productivity Score"].mean()
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor= "#0e1117", font=dict(color='white'),
            height=320
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
 
        # Personalized recommendations
        st.markdown("### Personalized Recommendations")
        recs = []
        if inp_screen > 6:
            recs.append(" **Reduce Screen Time** — You exceed the healthy 6hr threshold. Try app timers or digital detox hours.")
        if inp_sleep < 7:
            recs.append(" **Improve Sleep** — Less than 7 hours is linked to 30%+ productivity drop. Set a consistent sleep schedule.")
        if inp_app > 15:
            recs.append(" **Cut App Count** — High app usage fragments focus. Uninstall non-essential apps.")
        if inp_study < 4:
            recs.append(" **Increase Study Time** — Structured learning time is the strongest productivity predictor.")
        if inp_extra == 0:
            recs.append(" **Join Activities** — Extracurricular involvement correlates with better focus and mental health.")
        if not recs:
            recs.append(" **Great Habits!** Your digital lifestyle is well-balanced. Keep it up!")
 
        for r in recs:
            st.markdown(f'<div class="insight-box">{r}</div>', unsafe_allow_html=True)
 
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#556b7d; font-size:0.82rem; padding: 0.5rem 0;'>
    Digital Addiction Pattern Analysis · Powered by Streamlit, scikit-learn &amp; Plotly<br>
    Models: KMeans Clustering · Random Forest Regression · Gradient Boosting Classification · PCA
</div>
""", unsafe_allow_html=True)
