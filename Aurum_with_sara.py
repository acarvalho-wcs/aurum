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

        st.success("✅ File uploaded and cleaned successfully!")

        st.sidebar.markdown("---")
        st.sidebar.markdown("## 🧬 Select Species")
        species_options = sorted(df['Species'].dropna().unique())
        selected_species = st.sidebar.multiselect("Select one or more species:", species_options)

        if selected_species:
            df_selected = df[df['Species'].isin(selected_species)]

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

        else:
            st.warning("⚠️ Please select at least one species to explore the data.")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
