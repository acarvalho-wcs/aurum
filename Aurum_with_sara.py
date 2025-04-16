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

        st.sidebar.markdown("---")
        st.sidebar.markdown("## 🧬 Select Species")
        species_options = sorted(df['Species'].dropna().unique())
        selected_species = st.sidebar.multiselect("Select one or more species:", species_options)

        if selected_species:
            df_selected = df[df['Species'].isin(selected_species)]

            # Nova versão dinâmica de Network Analysis
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
                            else:
                                G.add_edge(sp1, sp2, weight=1)

                    if G.number_of_edges() == 0:
                        st.info("No edges were generated with the selected features.")
                    else:
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
                else:
                    st.info("Please select at least one feature to generate the network.")

        else:
            st.warning("⚠️ Please select at least one species to explore the data.")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
