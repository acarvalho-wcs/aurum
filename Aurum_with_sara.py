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
import os
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go

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
        else:
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

            show_viz = st.sidebar.checkbox("📊 Show Data Visualization", value=False)
            if show_viz:
                st.markdown("## 📊 Data Visualization")
                if st.sidebar.checkbox("Preview data"):
                    st.write("### Preview of cleaned data:")
                    st.dataframe(df_selected.head())

                chart_type = st.sidebar.selectbox("Select chart type:", ["Bar", "Line", "Scatter", "Pie"])
                x_axis = st.sidebar.selectbox("X-axis:", df_selected.columns, index=0)
                y_axis = st.sidebar.selectbox("Y-axis:", df_selected.columns, index=1)

                st.markdown("### Custom Chart")
                if chart_type == "Bar":
                    fig = px.bar(df_selected, x=x_axis, y=y_axis, color='Species')
                elif chart_type == "Line":
                    fig = px.line(df_selected, x=x_axis, y=y_axis, color='Species')
                elif chart_type == "Scatter":
                    fig = px.scatter(df_selected, x=x_axis, y=y_axis, color='Species')
                elif chart_type == "Pie":
                    fig = px.pie(df_selected, names=x_axis, values=y_axis)
                st.plotly_chart(fig)

            show_trend = st.sidebar.checkbox("📈 Show Trend Analysis", value=False)
            if show_trend:
                st.markdown("## 📈 Trend Analysis")
                breakpoint_year = st.number_input("Breakpoint year (split the trend):", 1990, 2030, value=2015)

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

                tcs, tcs_log = trend_component(df_selected, breakpoint=breakpoint_year)
                st.markdown(f"**Trend Coordination Score (TCS):** `{tcs:.2f}`")
                st.info(tcs_log)

                st.markdown("### 📉 Trend Plot")
                fig, ax = plt.subplots(figsize=(8, 5))
                for species in selected_species:
                    subset = df_selected[df_selected['Species'] == species]
                    ax.scatter(subset['Year'], subset['N_seized'], label=species, alpha=0.6)
                    df_pre = subset[subset['Year'] <= breakpoint_year]
                    df_post = subset[subset['Year'] > breakpoint_year]
                    if len(df_pre) > 1:
                        model_pre = sm.OLS(df_pre['N_seized'], sm.add_constant(df_pre['Year'])).fit()
                        ax.plot(df_pre['Year'], model_pre.predict(sm.add_constant(df_pre['Year'])), linestyle='--')
                    if len(df_post) > 1:
                        model_post = sm.OLS(df_post['N_seized'], sm.add_constant(df_post['Year'])).fit()
                        ax.plot(df_post['Year'], model_post.predict(sm.add_constant(df_post['Year'])), linestyle='-.')
                ax.axvline(breakpoint_year, color='red', linestyle=':', label=f"Breakpoint = {breakpoint_year}")
                ax.set_title("Seizure Trend by Species")
                ax.set_xlabel("Year")
                ax.set_ylabel("Individuals Seized")
                ax.legend()
                st.pyplot(fig)

                cusum_enabled = st.checkbox("➕ Include CUSUM Analysis", value=False)
                if cusum_enabled:
                    st.markdown("### 🔎 CUSUM Analysis with Anomaly Detection")
                    def calculate_cusum_series(values):
                        mean = np.mean(values)
                        cusum_pos, cusum_neg = [0], [0]
                        for val in values[1:]:
                            s_pos = max(0, cusum_pos[-1] + (val - mean))
                            s_neg = min(0, cusum_neg[-1] + (val - mean))
                            cusum_pos.append(s_pos)
                            cusum_neg.append(s_neg)
                        return np.array(cusum_pos), np.array(cusum_neg)

                    fig, ax = plt.subplots(figsize=(10, 5))
                    for species in selected_species:
                        subset = df_selected[df_selected['Species'] == species]
                        yearly = subset.groupby("Year")["N_seized"].sum().sort_index()
                        years = yearly.index
                        values = yearly.values
                        cusum_pos, cusum_neg = calculate_cusum_series(values)
                        threshold = np.std(values) * 2
                        anomalies = np.where((cusum_pos > threshold) | (cusum_neg < -threshold))[0]
                        ax.plot(years, values, marker='o', color='black', label="Trend" if species == selected_species[0] else "")
                        ax.plot(years, cusum_pos, linestyle='--', color='green', label="CUSUM+" if species == selected_species[0] else "")
                        ax.plot(years, cusum_neg, linestyle='--', color='orange', label="CUSUM–" if species == selected_species[0] else "")
                        if len(anomalies) > 0:
                            ax.scatter(years[anomalies], values[anomalies], color='red', marker='x', s=80, label="Anomaly" if species == selected_species[0] else "")
                    ax.set_title("Trend & CUSUM with Anomalies")
                    ax.set_xlabel("Year")
                    ax.set_ylabel("Seized Specimens")
                    ax.legend()
                    st.pyplot(fig)
