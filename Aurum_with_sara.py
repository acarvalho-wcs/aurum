
import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from scipy.stats import chi2_contingency

st.set_page_config(page_title="Aurum Dashboard", layout="wide")
st.title("🐾 Aurum - Wildlife Crime Analysis Dashboard")
st.markdown("Select an analysis from the sidebar to begin.")

st.sidebar.markdown("## 📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file (.xlsx):", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip().str.replace('\xa0', '', regex=True)

        if 'Year' in df.columns:
            df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})').astype(float)

        def expand_multi_species_rows(df):
            expanded_rows = []
            for _, row in df.iterrows():
                matches = re.findall(r'(\d+)\s*([A-Z]{2,})', str(row.get('N seized specimens', '')))
                if matches:
                    for qty, species in matches:
                        new_row = row.copy()
                        new_row['N_seized'] = float(qty)
                        new_row['Species'] = species
                        expanded_rows.append(new_row)
                else:
                    expanded_rows.append(row)
            return pd.DataFrame(expanded_rows)

        df = expand_multi_species_rows(df).reset_index(drop=True)

        if 'Case #' in df.columns and 'Species' in df.columns:
            species_per_case = df.groupby('Case #')['Species'].nunique()
            df['Logistic Convergence'] = df['Case #'].map(lambda x: "Yes" if species_per_case.get(x, 0) > 1 else "No")

        def normalize_text(text):
            if not isinstance(text, str):
                text = str(text)
            text = text.strip().lower()
            text = unicodedata.normalize("NFKD", text)
            text = re.sub(r'\s+', ' ', text)
            return text

        def infer_stage(row):
            seizure = normalize_text(row.get("Seizure Status", ""))
            transit = normalize_text(row.get("Transit Feature", ""))
            logistic = row.get("Logistic Convergence", "No")
            if any(k in seizure for k in ["planned", "trap", "attempt"]):
                return "Preparation"
            elif "captivity" in transit or "breeding" in transit:
                return "Captivity"
            elif any(k in transit for k in ["airport", "border", "highway", "port"]):
                return "Transport"
            elif logistic == "Yes":
                return "Logistic Consolidation"
            else:
                return "Unclassified"

        df["Inferred Stage"] = df.apply(infer_stage, axis=1)

        st.success("✅ File uploaded and cleaned successfully!")

        species_options = sorted(df['Species'].dropna().unique())
        selected_species = st.sidebar.multiselect("Select one or more species:", species_options)

        if selected_species:
            df_selected = df[df['Species'].isin(selected_species)]

            if st.sidebar.checkbox("📊 Show Data Visualization", value=False):
                st.markdown("## 📊 Data Visualization")
                if st.sidebar.checkbox("Preview data"):
                    st.dataframe(df_selected.head())

                chart_type = st.sidebar.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Pie"])
                x = st.sidebar.selectbox("X Axis", df_selected.columns)
                y = st.sidebar.selectbox("Y Axis", df_selected.columns)

                if chart_type == "Bar":
                    fig = px.bar(df_selected, x=x, y=y, color='Species')
                elif chart_type == "Line":
                    fig = px.line(df_selected, x=x, y=y, color='Species')
                elif chart_type == "Scatter":
                    fig = px.scatter(df_selected, x=x, y=y, color='Species')
                elif chart_type == "Pie":
                    fig = px.pie(df_selected, names=x, values=y)
                st.plotly_chart(fig)

            if st.sidebar.checkbox("📈 Show Trend Analysis", value=False):
                st.markdown("## 📈 Trend Analysis")

                def trend_component(df, year_col='Year', count_col='N_seized', breakpoint=2015):
                    df_pre = df[df[year_col] <= breakpoint]
                    df_post = df[df[year_col] > breakpoint]

                    if len(df_pre) < 2 or len(df_post) < 2:
                        return 0.0, "Insufficient data"

                    X_pre = sm.add_constant(df_pre[[year_col]])
                    y_pre = df_pre[count_col]
                    slope_pre = sm.OLS(y_pre, X_pre).fit().params[year_col]

                    X_post = sm.add_constant(df_post[[year_col]])
                    y_post = df_post[count_col]
                    slope_post = sm.OLS(y_post, X_post).fit().params[year_col]

                    tcs = (slope_post - slope_pre) / (abs(slope_pre) + 1)
                    return tcs, f"TCS = {tcs:.2f}"

                tcs, log = trend_component(df_selected)
                st.markdown(f"**{log}**")

                show_cusum = st.checkbox("Show CUSUM & Cumulative Mean", value=False)
                if show_cusum:
                    fig2, ax2 = plt.subplots(figsize=(8, 3))
                    for sp in selected_species:
                        subset = df_selected[df_selected['Species'] == sp].sort_values("Year")
                        y = subset['N_seized']
                        ax2.plot(subset['Year'], y.expanding().mean(), label=f"{sp} Mean")
                        ax2.plot(subset['Year'], (y - y.mean()).cumsum(), linestyle="--", label=f"{sp} CUSUM")
                    ax2.legend()
                    st.pyplot(fig2)

            if st.sidebar.checkbox("🧬 Show Species Co-occurrence", value=False):
                st.markdown("## 🧬 Species Co-occurrence")

                def general_species_cooccurrence(df, species_list, case_col='Case #'):
                    presence = pd.DataFrame(index=df[case_col].unique())
                    for sp in species_list:
                        sp_df = df[df['Species'] == sp][[case_col]]
                        sp_df['present'] = 1
                        presence[sp] = presence.index.isin(sp_df[case_col])
                    results = []
                    for sp1, sp2 in combinations(species_list, 2):
                        table = pd.crosstab(presence[sp1], presence[sp2])
                        if table.shape == (2, 2):
                            chi2, p, _, _ = chi2_contingency(table)
                            results.append((sp1, sp2, chi2, p))
                    return results

                results = general_species_cooccurrence(df_selected, selected_species)
                for sp1, sp2, chi2, p in results:
                    st.markdown(f"**{sp1} × {sp2}** — Chi² = `{chi2:.2f}` | p = `{p:.4f}`")

            if st.sidebar.checkbox("🚨 Show Anomaly Detection", value=False):
                st.markdown("## 🚨 Anomaly Detection")
                numeric_cols = df_selected.select_dtypes(include=np.number).columns.tolist()
                features = st.multiselect("Select features", numeric_cols, default=numeric_cols[:2])
                if features:
                    X = StandardScaler().fit_transform(df_selected[features])
                    models = {
                        "IForest": IsolationForest().fit_predict(X),
                        "LOF": LocalOutlierFactor().fit_predict(X),
                        "Z-Score": np.where(np.abs(X).max(axis=1) > 3, -1, 1)
                    }
                    votes = pd.DataFrame(models)
                    votes["Case #"] = df_selected["Case #"].values
                    votes["Outlier Score"] = (votes == -1).sum(axis=1)
                    st.dataframe(votes.sort_values(by="Outlier Score", ascending=False).head(10))

            if st.sidebar.checkbox("🕸️ Show Network Analysis", value=False):
                st.markdown("## 🕸️ Network Analysis")
                features = st.multiselect("Group by features", df_selected.columns, default=["Case #"])
                G = nx.Graph()
                for _, group in df_selected.groupby(features):
                    species = group["Species"].unique()
                    for a, b in combinations(species, 2):
                        G.add_edge(a, b)
                pos = nx.spring_layout(G)
                edge_trace = go.Scatter(x=[], y=[], mode='lines', line=dict(width=0.5, color='#888'))
                node_trace = go.Scatter(x=[], y=[], mode='markers+text', text=[], textposition='top center')
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_trace.x += (x0, x1, None)
                    edge_trace.y += (y0, y1, None)
                for node in G.nodes():
                    x, y = pos[node]
                    node_trace.x += (x,)
                    node_trace.y += (y,)
                    node_trace.text += (node,)
                fig = go.Figure(data=[edge_trace, node_trace])
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
