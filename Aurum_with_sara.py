import streamlit as st
import pandas as pd
import re
import unicodedata
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from scipy.stats import chi2_contingency
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Configuração da página
st.set_page_config(page_title="Aurum Dashboard", layout="wide")

# Título e logotipo
st.title("🐾 Aurum - Wildlife Crime Analysis Dashboard")
st.markdown("Select an analysis from the sidebar to begin.")

# Upload do arquivo
st.sidebar.markdown("## 📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file (.xlsx):", type=["xlsx"])

df = None
df_selected = None
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
                            # [REMOVIDO AUTOMATICAMENTE] else sem contexto
                            expanded_rows.append(row)
                            return pd.DataFrame(expanded_rows)

                            df = expand_multi_species_rows(df).reset_index(drop=True)


                            # Aplicar valores numéricos aos países se o arquivo estiver disponível
                            import os
                            country_score_path = "country_offenders_values.csv"
                            if os.path.exists(country_score_path):
                                df_country_score = pd.read_csv(country_score_path, encoding="ISO-8859-1")
                                country_map = dict(zip(df_country_score["Country"].str.strip(), df_country_score["Value"]))

                                def score_countries(cell_value, country_map):
                                    if not isinstance(cell_value, str):
                                        return 0
                                        countries = [c.strip() for c in cell_value.split("+")]
                                        return sum(country_map.get(c, 0) for c in countries)

                                        if "Country of offenders" in df.columns:
                                            df["Offender_value"] = df["Country of offenders"].apply(lambda x: score_countries(x, country_map))
                                            st.markdown("✅ `Offender_value` column added using country_offenders_values.csv")
                                            # [REMOVIDO AUTOMATICAMENTE] else sem contexto
                                            st.warning("⚠️ File country_offenders_values.csv not found. Offender scoring skipped.")


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

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
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
def trend_component(df, year_col='Year', count_col='N_seized', breakpoint=2015):
    df_pre = df[df[year_col] <= breakpoint]
    df_post = df[df[year_col] > breakpoint]
    if len(df_pre) < 2 or len(df_post) < 2:
        return 0.0, "Insufficient data for segmented regression"
    X_pre = sm.add_constant(df_pre[[year_col]])
    y_pre = df_pre[count_col]
    model_pre = sm.OLS(y_pre, X_pre).fit()
    slope_pre = model_pre.params[year_col]
    X_post = sm.add_constant(df_post[[year_col]])
    y_post = df_post[count_col]
    model_post = sm.OLS(y_post, X_post).fit()
    slope_post = model_post.params[year_col]
    tcs = (slope_post - slope_pre) / (abs(slope_pre) + 1)
    log = f"TCS = {tcs:.2f}"
    return tcs, log
        def general_species_cooccurrence(df, species_list, case_col='Case #'):
        presence = pd.DataFrame()
        presence[case_col] = df[case_col].unique()
        presence.set_index(case_col, inplace=True)

        for sp in species_list:
        sp_df = df[df['Species'] == sp][[case_col]]
        sp_df['present'] = 1
        grouped = sp_df.groupby(case_col)['present'].max()
        presence[sp] = grouped

        presence.fillna(0, inplace=True)
        presence = presence.astype(int)

        results = []
        for sp_a, sp_b in combinations(species_list, 2):
        table = pd.crosstab(presence[sp_a], presence[sp_b])
        if table.shape == (2, 2):
        chi2, p, _, _ = chi2_contingency(table)
        results.append((sp_a, sp_b, chi2, p, table))
        return results

        co_results = general_species_cooccurrence(df_selected, selected_species)

        if co_results:
        st.markdown("### 📊 Co-occurrence Results")
        for sp_a, sp_b, chi2, p, table in co_results:
        st.markdown(f"**{sp_a} × {sp_b}**")
        st.dataframe(table)
        st.markdown(f"Chi² = `{chi2:.2f}` | p = `{p:.4f}`")
        st.markdown("---")
        # [REMOVIDO AUTOMATICAMENTE] else sem contexto
        st.info("No co-occurrence data available for selected species.")

        show_anomaly = st.sidebar.checkbox("🚨 Show Anomaly Detection", value=False)
        if show_anomaly:
        st.markdown("## 🚨 Anomaly Detection")

        numeric_cols = [col for col in df_selected.columns if pd.api.types.is_numeric_dtype(df_selected[col])]
        selected_features = st.multiselect("Select numeric features for anomaly detection:", numeric_cols, default=["N_seized", "Year"])

        if selected_features:
        X = StandardScaler().fit_transform(df_selected[selected_features])

        models = {
        "Isolation Forest": IsolationForest(random_state=42).fit_predict(X),
        "LOF": LocalOutlierFactor().fit_predict(X),
        "DBSCAN": DBSCAN(eps=1.2, min_samples=2).fit_predict(X),
        "Z-Score": np.where(np.any(np.abs(X) > 3, axis=1), -1, 1)
        }

        try:
        cov = np.cov(X, rowvar=False)
        inv_cov = np.linalg.inv(cov)
        mean = np.mean(X, axis=0)
        diff = X - mean
        md = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
        threshold_md = np.percentile(md, 97.5)
        models["Mahalanobis"] = np.where(md > threshold_md, -1, 1)
        except np.linalg.LinAlgError:
        models["Mahalanobis"] = np.ones(len(X))

        vote_df = pd.DataFrame(models)
        vote_df["Outlier Votes"] = (vote_df == -1).sum(axis=1)
        vote_df["Case #"] = df_selected["Case #"].values

        consensus_ratio = (vote_df["Outlier Votes"] > 2).sum() / len(vote_df)
        st.markdown(f"**Consensus Outlier Ratio:** `{consensus_ratio:.2%}`")

        st.markdown("### 📋 Most anomalous cases")
        top_outliers = vote_df.sort_values(by="Outlier Votes", ascending=False).head(10)
        st.dataframe(top_outliers.set_index("Case #"))

        show_network = st.sidebar.checkbox("🕸️ Show Network Analysis", value=False)
        if show_network:
        st.markdown("## 🕸️ Network Analysis")

        import networkx as nx
        import plotly.graph_objects as go

        st.markdown("This network will be generated based on selected features.")

        available_features = [col for col in df_selected.columns if col not in ['N_seized'] and df_selected[col].nunique() > 1]
        selected_network_features = st.multiselect("Select features to define network connections:", available_features, default=["Case #"])

        if selected_network_features:
        G = nx.Graph()

        for key, group in df_selected.groupby(selected_network_features):
        species_in_group = group['Species'].unique()
        for sp1, sp2 in combinations(species_in_group, 2):
        if G.has_edge(sp1, sp2):
        G[sp1][sp2]['weight'] += 1
        # [REMOVIDO AUTOMATICAMENTE] else sem contexto
        G.add_edge(sp1, sp2, weight=1)

        if G.number_of_edges() == 0:
        st.info("No edges were generated with the selected features.")
        # [REMOVIDO AUTOMATICAMENTE] else sem contexto
        pos = nx.spring_layout(G, seed=42)

        edge_x = []
        edge_y = []
        for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node} ({G.degree[node]} connections)")

        node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        hoverinfo='text',
        marker=dict(
        showscale=False,
        color='lightblue',
        size=[8 + 2*G.degree[node] for node in G.nodes()],
        line_width=1))

        fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
        title='Dynamic Species Co-occurrence Network',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40)))

        st.plotly_chart(fig, use_container_width=True)
        # [REMOVIDO AUTOMATICAMENTE] else sem contexto
        st.info("Please select at least one feature to generate the network.")

        # [REMOVIDO AUTOMATICAMENTE] else sem contexto
        st.warning("⚠️ Please select at least one species to explore the data.")

        except Exception as e:
        st.error(f"❌ Error reading file: {e}")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
