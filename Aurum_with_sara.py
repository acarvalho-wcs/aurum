import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import re
import unicodedata
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
from itertools import combinations
from scipy.stats import chi2_contingency
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import bcrypt
import os
import requests
import extruct
from w3lib.html import get_base_url
from bs4 import BeautifulSoup
from transformers import pipeline

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Aurum Dashboard", layout="centered")
st.title("Aurum - Wildlife Trafficking Analytics")

# Upload do arquivo
st.sidebar.markdown("## Welcome to Aurum")
st.sidebar.markdown("Log in below to unlock multi-user tools.")
show_about = st.sidebar.button("**About Aurum**")
if show_about:
    st.markdown("## About Aurum")
    st.markdown("""
**Aurum** is a modular and interactive toolkit designed to support the detection and analysis of **wildlife trafficking** and organized environmental crime. Developed by the Wildlife Conservation Society (WCS) – Brazil, it empowers analysts, researchers, and enforcement professionals with data-driven insights through a user-friendly interface.

The platform enables the upload and processing of case-level data and offers a suite of analytical tools, including:

- **Trend Analysis**: Explore temporal patterns using segmented regression (TCS) to measure directional changes in trends before and after a chosen breakpoint year. Additionally, detect significant deviations from historical averages with CUSUM.
- **Species Co-occurrence**: Identify statistically significant co-trafficking relationships between species using chi-square tests and network-based representations.
- **Anomaly Detection**: Apply multiple methods (Isolation Forest, LOF, DBSCAN, Mahalanobis distance, Z-Score) to identify outlier cases based on numerical features.
- **Criminal Network Analysis**: Visualize co-occurrence networks to reveal potential connections and logistical consolidation among species and locations.
- **Interactive Visualization**: Generate customized plots and dashboards based on uploaded data and selected variables.
""")

st.sidebar.markdown("## 📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("**Upload your Excel file (.xlsx).**", type=["xlsx"])

st.sidebar.markdown("**Download Template**")
with open("Aurum_template.xlsx", "rb") as f:
    st.sidebar.download_button(
        label="Download a data template for wildlife trafficking analysis in Aurum",
        data=f,
        file_name="aurum_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
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
                matches = re.findall(r'(\d+)\s*([A-Z]{2,}\d{0,3})', str(row.get('N seized specimens', '')))
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
        else:
            st.warning("⚠️ File country_offenders_values.csv not found. Offender scoring skipped.")


        if 'Case #' in df.columns and 'Species' in df.columns:
            species_per_case = df.groupby('Case #')['Species'].nunique()
            df['Logistic Convergence'] = df['Case #'].map(lambda x: "1" if species_per_case.get(x, 0) > 1 else "0")

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
            logistic = row.get("Logistic Convergence", "0")
            if any(k in seizure for k in ["planned", "trap", "attempt"]):
                return "Preparation"
            elif "captivity" in transit or "breeding" in transit:
                return "Captivity"
            elif any(k in transit for k in ["airport", "border", "highway", "port"]):
                return "Transport"
            elif logistic == "1":
                return "Logistic Consolidation"
            else:
                return "Unclassified"

        df["Inferred Stage"] = df.apply(infer_stage, axis=1)

        st.success("✅ File uploaded and cleaned successfully!")

        st.sidebar.markdown("## Select Species")
        species_options = sorted(df['Species'].dropna().unique())
        selected_species = st.sidebar.multiselect("Select one or more species:", species_options)

        if selected_species:
            df_selected = df[df['Species'].isin(selected_species)]

            show_viz = st.sidebar.checkbox("Data Visualization", value=False)
            if show_viz:
                st.markdown("## Data Visualization")
                if st.sidebar.checkbox("Preview data"):
                    st.write("### Preview of cleaned data:")
                    st.dataframe(df_selected.head())

                chart_type = st.sidebar.selectbox("Select chart type:", ["Bar", "Line", "Scatter", "Pie"])
                x_axis = st.sidebar.selectbox("X-axis:", df_selected.columns, index=0)
                y_axis = st.sidebar.selectbox("Y-axis:", df_selected.columns, index=1)

                import plotly.express as px
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

            
            show_trend = st.sidebar.checkbox("Trend Analysis", value=False)
            if show_trend:
                st.markdown("## Trend Analysis")

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

                with st.expander("ℹ️ Learn more about this analysis"):
                    st.markdown("""
                    ### About Trend Analysis

                    The *Trend Analysis* section helps identify shifts in wildlife seizure patterns over time for the selected species.

                    - The analysis uses segmented linear regressions based on a user-defined **breakpoint year**.
                    - For each species, a regression is computed before and after the breakpoint to estimate the slope (i.e., the trend) of increase or decrease.
                    - These slopes are used to calculate the **Trend Coordination Score (TCS)**, which measures the relative change between the two periods:
                      - `TCS > 0` indicates an increase in trend after the breakpoint.
                      - `TCS < 0` indicates a decrease.
                      - `TCS ≈ 0` suggests stability.

                    - The score is normalized to reduce instability when the pre-breakpoint slope is close to zero. While TCS has no strict bounds, in practice it typically falls between −1 and +1. 
                    - Extreme values may indicate sharp shifts in trend intensity or imbalances in the temporal distribution of data. Although wildlife trafficking patterns are rarely linear in reality, this method adopts linear segments as a practical approximation to detect directional shifts. 
                    - It does not assume true linear behavior, but rather uses regression slopes as a comparative metric across time intervals. The analysis requires at least two observations on each side of the breakpoint to produce meaningful estimates. 
                    - The score can be sensitive to outliers or sparsely populated time ranges, and should be interpreted in light of the broader case context.
                    - The section also generates a plot showing data points and trend lines for each species, making it easier to visualize changes over time.
                    - Find more details in the ReadMe file and/or in Carvalho (2025).
                    """)

                st.markdown("### Trend Plot")
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

                with st.expander("📉 Show regression details by species"):
                    for species in selected_species:
                        subset = df_selected[df_selected['Species'] == species]
                        df_pre = subset[subset['Year'] <= breakpoint_year]
                        df_post = subset[subset['Year'] > breakpoint_year]

                        st.markdown(f"#### {species}")

                        if len(df_pre) > 1:
                            X_pre = sm.add_constant(df_pre['Year'])
                            y_pre = df_pre['N_seized']
                            model_pre = sm.OLS(y_pre, X_pre).fit()
                            slope_pre = model_pre.params['Year']
                            r2_pre = model_pre.rsquared
                            pval_pre = model_pre.pvalues['Year']
                            st.markdown(f"- Pre-breakpoint slope: β = `{slope_pre:.2f}`")
                            st.markdown(f"- R² = `{r2_pre:.2f}`")
                            st.markdown(f"- p-value = `{pval_pre:.4f}`")
                        else:
                            st.info("Not enough data before breakpoint.")

                        if len(df_post) > 1:
                            X_post = sm.add_constant(df_post['Year'])
                            y_post = df_post['N_seized']
                            model_post = sm.OLS(y_post, X_post).fit()
                            slope_post = model_post.params['Year']
                            r2_post = model_post.rsquared
                            pval_post = model_post.pvalues['Year']
                            st.markdown(f"- Post-breakpoint slope: β = `{slope_post:.2f}`")
                            st.markdown(f"- R² = `{r2_post:.2f}`")
                            st.markdown(f"- p-value = `{pval_post:.4f}`")
                        else:
                            st.info("Not enough data after breakpoint.")

                # Optional CUSUM
                if st.checkbox("Activate CUSUM analysis"):
                    st.subheader("CUSUM - Trend Change Detection")

                    cusum_option = st.radio(
                        "Choose the metric to analyze:",
                        ["Total individuals seized per year", "Average individuals per seizure (per year)"]
                    )

                    if cusum_option == "Total individuals seized per year":
                        df_cusum = df_selected.groupby("Year")["N_seized"].sum().reset_index()
                        col_data = "N_seized"
                        col_time = "Year"
                    else:
                        df_cusum = df_selected.groupby("Year")["N_seized"].mean().reset_index()
                        col_data = "N_seized"
                        col_time = "Year"

                    def plot_cusum_trend(df, col_data, col_time, species_name="Selected Species"):
                        df_sorted = df.sort_values(by=col_time).reset_index(drop=True)
                        years = df_sorted[col_time]
                        values = df_sorted[col_data]

                        mean_val = values.mean()
                        std_dev = values.std()

                        cusum_pos = [0]
                        cusum_neg = [0]

                        for i in range(1, len(values)):
                            delta = values.iloc[i] - mean_val
                            cusum_pos.append(max(0, cusum_pos[-1] + delta))
                            cusum_neg.append(min(0, cusum_neg[-1] + delta))

                        fig, ax = plt.subplots(figsize=(10, 6))

                        ax.plot(years, values, color='black', marker='o', label='Trend')
                        ax.plot(years, cusum_pos, color='green', linestyle='--', label='CUSUM+')
                        ax.plot(years, cusum_neg, color='orange', linestyle='--', label='CUSUM-')

                        # Highlight years with significant deviation
                        highlight_years = [
                            i for i, val in enumerate(values)
                            if abs(val - mean_val) > 1.5 * std_dev
                        ]

                        ax.scatter(
                            [years.iloc[i] for i in highlight_years],
                            [values.iloc[i] for i in highlight_years],
                            color='red', marker='x', s=100, label='Significant Deviation'
                        )

                        ax.set_title(f"{species_name} - Trend & CUSUM", fontsize=14)
                        ax.set_xlabel("Year")
                        ax.set_ylabel("Seized Specimens")
                        ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray')
                        ax.legend()
                        st.pyplot(fig)

                        # Interpretation
                        st.subheader("Automated Interpretation")
                        cusum_range = max(cusum_pos) - min(cusum_neg)

                        if cusum_range > 2 * std_dev:
                            # Detect all years with significant deviation from the mean
                            change_years = [
                                years.iloc[i]
                                for i, val in enumerate(values)
                                if abs(val - mean_val) > 1.5 * std_dev
                            ]

                            if change_years:
                                formatted_years = " and ".join(str(y) for y in change_years)
                                st.markdown(f"Significant trend changes detected in: **{formatted_years}** (based on deviations from the historical average).")
                            else:
                                st.markdown("CUSUM suggests change, but no single year shows strong deviation from the mean.")
                        else:
                            st.markdown("✅ No significant trend change detected.")


                        # Explanation toggle
                        with st.expander("ℹ️ Learn more about this analysis"):
                            st.markdown("""
                            ### About CUSUM Analysis

                            The *CUSUM Analysis* section is designed to detect significant changes in the temporal trend of wildlife seizures by evaluating how yearly values deviate from the historical average.

                            - The method is based on **Cumulative Sum (CUSUM)** analysis, which tracks the cumulative deviation of observed values from their overall mean.
                            - Two cumulative paths are calculated:
                              - **CUSUM+** accumulates positive deviations (above the mean).
                              - **CUSUM−** accumulates negative deviations (below the mean).
                            - This dual-track approach highlights the **direction and magnitude of long-term deviations**, making it easier to identify sustained changes.

                            - Unlike methods that directly model trends (e.g., segmented regression), CUSUM reacts **only when there is consistent deviation**, amplifying the signal of real change over time.
                            - The method does **not identify the exact year of change by itself**, but instead signals that a shift in the distribution has occurred—often **triggered by a single or small set of high-impact events**.
                            - To estimate the timing more precisely, the analysis **identifies years where the seizure counts deviate sharply from the historical mean** (greater than 1.5 standard deviations).
                            - These years are reported as **likely points of trend change**.

                            - CUSUM is especially useful when changes are not gradual or linear, and when a single anomalous year can drive broader shifts.
                            - The results should be interpreted in light of species-specific context and enforcement history, as well as any known conservation events, policy changes, or trafficking routes.

                            - The section also generates a plot combining:
                              - Observed seizure counts (black line),
                              - CUSUM+ and CUSUM− paths (green and orange dashed lines),
                              - Highlighted years with significant deviations (when present).

                            - For more details, refer to the ReadMe file and/or Carvalho (2025).
                            """)

                    plot_cusum_trend(df_cusum, col_data=col_data, col_time=col_time)

            show_cooc = st.sidebar.checkbox("Species Co-occurrence", value=False)
            if show_cooc:
                st.markdown("## Species Co-occurrence Analysis")

                with st.expander("ℹ️ Learn more about this analysis"):
                    st.markdown("""
                    ### About Species Co-occurrence

                    The Species Co-occurrence section identifies pairs of species that tend to be trafficked together within the same cases, revealing potential patterns of coordinated extraction, transport, or market demand.

                    This analysis uses a binary presence-absence matrix for each selected species across all case IDs. For every species pair, a **chi-square test of independence** is performed to evaluate whether the observed co-occurrence is statistically significant beyond what would be expected by chance.

                    - A **2×2 contingency table** is generated for each pair, indicating joint presence or absence across cases.
                    - The **Chi² statistic** quantifies the degree of association: higher values suggest stronger deviation from independence (i.e., a stronger link between the species).
                    - The associated **p-value** indicates whether this deviation is statistically significant. A p-value below 0.05 typically means the co-occurrence is unlikely to be due to chance.

                    **Interpretation**:
                    - **High Chi² values** signal that the two species co-occur more (or less) than expected — implying possible ecological overlap, shared trafficking routes, or joint market targeting.
                    - **Low Chi² values** suggest weak or no association, even if species occasionally appear together.

                    This method is particularly useful for identifying species that may be captured, transported, or traded together due to logistical, ecological, or commercial drivers.

                    The results are displayed in an interactive table showing co-occurrence counts, the Chi² statistic, and p-values for each species pair — helping analysts prioritize combinations for deeper investigation.
                    """)

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

                def interpret_cooccurrence(table, chi2, p):
                    a = table.iloc[0, 0]
                    b = table.iloc[0, 1]
                    c = table.iloc[1, 0]
                    d = table.iloc[1, 1]

                    threshold = 0.05

                    if p >= threshold:
                        st.info("No statistically significant association between these species was found (p ≥ 0.05).")
                        return

                    if d == 0:
                        st.warning("⚠️ These species were never trafficked together. This pattern suggests **mutual exclusivity**, possibly due to distinct trafficking chains or ecological separation.")
                    elif b + c == 0:
                        st.success("✅ These species always appear together. This indicates a **perfect positive association**, potentially reflecting joint capture, transport, or market demand.")
                    elif d > b + c:
                        st.success("🔗 These species frequently appear together and are **positively associated** in trafficking records. The co-occurrence is unlikely to be due to chance.")
                    elif d < min(b, c):
                        st.error("❌ These species are almost always recorded **separately**, suggesting a **strong negative association** or operational separation in trafficking routes.")
                    else:
                        st.info("ℹ️ A statistically significant association was detected. While co-occurrence exists, it is not dominant — suggesting **partial overlap** in trafficking patterns.")

                co_results = general_species_cooccurrence(df_selected, selected_species)

                if co_results:
                    st.markdown("### Co-occurrence Results")
                    for sp_a, sp_b, chi2, p, table in co_results:
                        st.markdown(f"**{sp_a} × {sp_b}**")
                        st.dataframe(table)
                        st.markdown(f"Chi² = `{chi2:.2f}` | p = `{p:.4f}`")
                        interpret_cooccurrence(table, chi2, p)
                        st.markdown("---")
                else:
                    st.info("No co-occurrence data available for selected species.")

                st.markdown("### Co-trafficked Species Cases")

                grouped = df_selected.groupby('Case #')
                multi_species_cases = grouped.filter(lambda x: x['Species'].nunique() > 1)

                if multi_species_cases.empty:
                    st.info("No multi-species trafficking cases found for the selected species.")
                else:
                    summary = multi_species_cases[['Case #', 'Country of offenders', 'Species', 'N_seized']].sort_values(by='Case #')
                    st.dataframe(summary)

            show_anomaly = st.sidebar.checkbox("Anomaly Detection", value=False)
            if show_anomaly:
                st.markdown("## Anomaly Detection")

                with st.expander("ℹ️ Learn more about this analysis"):
                    st.markdown("""
                    ### About Anomaly Detection

                    The Anomaly Detection section helps identify wildlife trafficking cases that deviate significantly from typical patterns based on selected numerical features.

                    The analysis applies multiple outlier detection algorithms to highlight cases that may involve unusual species combinations, unusually large quantities of individuals, recurring offender countries, or rare time periods.

                    By default, the following methods are applied in parallel:

                    - **Isolation Forest**: an ensemble method that detects anomalies by isolating data points in a randomly partitioned feature space.
                    - **Local Outlier Factor (LOF)**: detects anomalies based on the local density of data points. Cases that lie in areas of significantly lower density than their neighbors are flagged.
                    - **DBSCAN**: a clustering algorithm that marks low-density or unclustered points as outliers.
                    - **Z-Score**: a statistical approach that flags values deviating more than 3 standard deviations from the mean in any feature.
                    - **Mahalanobis Distance**: a multivariate distance measure that accounts for correlations between features to identify statistically distant points.

                    Each method produces a binary vote (outlier or not). The final score reflects how many methods agree in classifying a case as anomalous — forming a **consensus-based ranking**.

                    A high consensus score suggests stronger evidence of atypical behavior, but anomalies do not necessarily imply criminality. They may reflect rare events, data entry issues, or unique but legitimate circumstances.

                    This module is most effective when the user selects a meaningful combination of numerical features, such as year, number of individuals, or offender-related values. The output highlights the top-ranked outliers and their anomaly vote count, supporting investigation and prioritization.
                    """)

                numeric_cols = [col for col in df_selected.columns if pd.api.types.is_numeric_dtype(df_selected[col])]
                selected_features = st.multiselect("Select numeric features for anomaly detection:", numeric_cols, default=["N_seized", "Year", "Offender_value"])

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

                    st.markdown("### Most anomalous cases")
                    top_outliers = vote_df.sort_values(by="Outlier Votes", ascending=False).head(10)
                    st.dataframe(top_outliers.set_index("Case #"))

            show_network = st.sidebar.checkbox("Network Analysis", value=False)
            if show_network:
                st.markdown("## Network Analysis")

                import networkx as nx
                import plotly.graph_objects as go

                st.markdown("This network connects cases that share attributes like species or offender country.")

                default_features = ["Species", "Country of offenders"]
                network_features = st.multiselect(
                    "Select features to compare across cases:", 
                    options=[col for col in df_selected.columns if col != "Case #"],
                    default=default_features
                )

                if network_features:
                    # Prepare feature sets for each Case #
                    case_feature_sets = (
                        df_selected
                        .groupby("Case #")[network_features]
                        .agg(lambda x: set(x.dropna()))
                        .apply(lambda row: set().union(*row), axis=1)
                    )

                    G = nx.Graph()

                    # Create nodes
                    for case_id in case_feature_sets.index:
                        G.add_node(case_id)

                    # Create edges between cases that share features
                    case_ids = list(case_feature_sets.index)
                    for i in range(len(case_ids)):
                        for j in range(i + 1, len(case_ids)):
                            shared = case_feature_sets[case_ids[i]].intersection(case_feature_sets[case_ids[j]])
                            if shared:
                                G.add_edge(case_ids[i], case_ids[j], weight=len(shared))

                    if G.number_of_edges() == 0:
                        st.info("No connections were found between cases using the selected features.")
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
                            line=dict(width=1, color='#888'),
                            hoverinfo='text',
                            mode='lines',
                            text=[f"Shared features: {G[u][v]['weight']}" for u, v in G.edges()]
                        )

                        node_x = []
                        node_y = []
                        node_text = []
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            degree = G.degree[node]
                            node_text.append(f"Case #: {node} ({degree} connections)")

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
                                line_width=1
                            )
                        )

                        fig = go.Figure(
                            data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title='Case Connectivity Network',
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40)
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("### Network Metrics")

                        num_nodes = G.number_of_nodes()
                        num_edges = G.number_of_edges()
                        density = nx.density(G)
                        components = nx.number_connected_components(G)
                        degrees = dict(G.degree())
                        avg_degree = sum(degrees.values()) / num_nodes if num_nodes else 0
                        top_central = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]

                        st.write(f"- **Nodes:** {num_nodes}")
                        st.write(f"- **Edges:** {num_edges}")
                        st.write(f"- **Density:** `{density:.3f}`")
                        st.write(f"- **Connected components:** {components}")
                        st.write(f"- **Average degree:** `{avg_degree:.2f}`")

                        st.markdown("**Top central cases by degree:**")
                        for case_id, degree in top_central:
                            st.markdown(f"- Case `{case_id}`: {degree} connections")
                        
                        with st.expander("ℹ️ Learn more about this analysis"):
                            st.markdown("""
                            ### About Case Network Analysis

                            This section visualizes a network of wildlife trafficking cases based on shared attributes such as species, offender countries, or other relevant fields.

                            - **Each node in the network represents a unique case** (`Case #`).
                            - **An edge between two cases indicates that they share one or more selected attributes**, such as:
                                - The same species involved,
                                - The same offender country,
                                - Other user-selected fields (e.g., seizure location, transport method).

                            - The more attributes two cases have in common, the **stronger their connection** (i.e., higher edge weight).
                            - **Edge weight** represents the number of shared elements between the two cases, and is displayed interactively when hovering over connections.

                            - This type of network helps to:
                                - **Identify clusters of related cases**, which may signal recurrent patterns, shared trafficking routes, or organizational links.
                                - **Visualize potential case consolidation** (e.g., repeated behavior by the same actors or coordinated multi-species trafficking).
                                - Reveal connections that may not be obvious in tabular data.

                            - Node size reflects the number of connections (degree), helping to identify central or highly connected cases.
                            - The analysis is dynamic: users can choose which attributes to include, allowing flexible exploration of the data.

                            - For example:
                                - If two cases both involve *Panthera onca* and occurred with offenders from Brazil, a connection is drawn.
                                - If a third case shares only the species but not the country, it will also connect, but with a lower weight.

                            - For more information on network methods in environmental crime analysis, refer to the ReadMe file and Carvalho (2025).
                            """)
                else:
                    st.info("Please select at least one feature to define connections between cases.")
                    
        else:
            st.warning("⚠️ Please select at least one species to explore the data.")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def place_logo_bottom_right(image_path, width=100):
    img_base64 = get_base64_image(image_path)
    st.markdown(
        f"""
        <style>
        .custom-logo {{
            position: fixed;
            bottom: 40px;
            right: 10px;
            z-index: 9999;
        }}
        </style>
        <div class="custom-logo">
            <img src="data:image/png;base64,{img_base64}" width="{width}"/>
        </div>
        """,
        unsafe_allow_html=True
    )

# Chamada da função para exibir a logo
place_logo_bottom_right("wcs.jpg")

# --- AUTENTICAÇÃO E CONEXÃO COM GOOGLE SHEETS ---
SHEET_ID = "1HVYbot3Z9OBccBw7jKNw5acodwiQpfXgavDTIptSKic"
USERS_SHEET = "Users"
REQUESTS_SHEET = "Access Requests"

scope = ["https://www.googleapis.com/auth/spreadsheets"]
credentials = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
client = gspread.authorize(credentials)
sheets = client.open_by_key(SHEET_ID)

users_ws = sheets.worksheet(USERS_SHEET)
requests_ws = sheets.worksheet(REQUESTS_SHEET)
users_df = pd.DataFrame(users_ws.get_all_records())

st.sidebar.markdown("## Export Options")
export_xlsx = st.sidebar.button("Export Cleaned data.xlsx")
export_html = st.sidebar.button("Export Analysis Report (.html)")

if export_xlsx and df_selected is not None:
    from io import BytesIO
    towrite = BytesIO()
    df_selected.to_excel(towrite, index=False, engine='openpyxl')
    towrite.seek(0)
    st.download_button(
        label="Download Cleaned Excel File",
        data=towrite,
        file_name="aurum_cleaned_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
if export_html and df_selected is not None:
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_sections = []
    html_sections.append(f"<h1>Aurum Wildlife Trafficking Report</h1>")
    html_sections.append(f"<p><strong>Generated:</strong> {now}</p>")
    html_sections.append(f"<p><strong>Selected Species:</strong> {', '.join(selected_species)}</p>")

    # Tabela de dados
    html_sections.append("<h2>Data Sample</h2>")
    html_sections.append(df_selected.head(10).to_html(index=False))

    # Resultados de tendência
    if show_trend:
        html_sections.append("<h2>Trend Analysis</h2>")
        html_sections.append(f"<p><strong>TCS:</strong> {tcs:.2f}</p>")

        # Salvar figura
        trend_buf = BytesIO()
        fig.savefig(trend_buf, format="png", bbox_inches="tight")
        trend_buf.seek(0)
        trend_base64 = base64.b64encode(trend_buf.read()).decode("utf-8")
        html_sections.append(f'<img src="data:image/png;base64,{trend_base64}" width="700">')

    # Coocorrência
    if show_cooc and co_results:
        html_sections.append("<h2>Species Co-occurrence</h2>")
        for sp_a, sp_b, chi2, p, table in co_results:
            html_sections.append(f"<h4>{sp_a} × {sp_b}</h4>")
            html_sections.append(table.to_html())
            html_sections.append(f"<p>Chi² = {chi2:.2f} | p = {p:.4f}</p>")

    # Anomalias
    if show_anomaly and 'vote_df' in locals():
        html_sections.append("<h2>Anomaly Detection</h2>")
        html_sections.append(f"<p><strong>Consensus Outlier Ratio:</strong> {consensus_ratio:.2%}</p>")
        html_sections.append("<h4>Top Anomalies</h4>")
        html_sections.append(top_outliers.to_html(index=False))

    # Finaliza o HTML
    html_report = f"""
    <html>
    <head><meta charset='utf-8'><title>Aurum Report</title></head>
    <body>{''.join(html_sections)}</body>
    </html>
    """

    report_bytes = BytesIO()
    report_bytes.write(html_report.encode("utf-8"))
    report_bytes.seek(0)

    st.download_button(
        label="Download HTML Report",
        data=report_bytes,
        file_name="aurum_report.html",
        mime="text/html"
    )
    
# --- LOGIN ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔐 Login to Aurum - Under maintenance")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_button = st.sidebar.button("Login")

# Verify encrypted password
def verify_password(password, hashed):
    return password == hashed_pw

if login_button and username and password:
    user_row = users_df[users_df["Username"] == username]
    if not user_row.empty and str(user_row.iloc[0]["Approved"]).strip().lower() == "true":
        hashed_pw = user_row.iloc[0]["Password"].strip()
        
        if verify_password(password, hashed_pw):
            st.session_state["user"] = username
            st.session_state["is_admin"] = str(user_row.iloc[0]["Is_Admin"]).strip().lower() == "true"
            st.success(f"Logged in as {username}")
        else:
            st.error("Incorrect password.")
    else:
        st.error("User not approved or does not exist.")

# --- FORMULÁRIO DE ACESSO (REQUISIÇÃO) ---
# Inicializa estado
if "show_sidebar_request" not in st.session_state:
    st.session_state["show_sidebar_request"] = False

# Botão na sidebar
if st.sidebar.button("📩 Request Access"):
    st.session_state["show_sidebar_request"] = True

# Exibe o formulário de solicitação na sidebar se o botão foi clicado
if st.session_state["show_sidebar_request"]:
    with st.sidebar.form("sidebar_request_form"):
        new_username = st.text_input("Choose a username", key="sidebar_user")
        new_password = st.text_input("Choose a password", type="password", key="sidebar_pass")
        institution = st.text_input("Institution", key="sidebar_inst")
        email = st.text_input("E-mail", key="sidebar_email")
        reason = st.text_area("Why do you want access to Aurum?", key="sidebar_reason")
        submit_request = st.form_submit_button("Submit Request")

        if submit_request:
            if not new_username or not new_password or not reason:
                st.sidebar.warning("All fields are required.")
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                requests_ws.append_row([
                    timestamp,
                    new_username,
                    new_password,
                    institution,
                    email,
                    reason
                ])
                st.sidebar.success("✅ Request submitted!")
                st.session_state["show_sidebar_request"] = False

# --- PAINEL ADMINISTRATIVO ---
if st.session_state.get("is_admin"):
    st.markdown("## 🛡️ Admin Panel - Approve Access Requests")
    request_df = pd.DataFrame(requests_ws.get_all_records())
    if not request_df.empty:
        st.dataframe(request_df)

        with st.form("approve_form"):
            new_user = st.selectbox("Select username to approve:", request_df["Username"].unique())
            new_password = st.text_input("Set initial password", type="password")
            is_admin = st.checkbox("Grant admin access?")
            approve_button = st.form_submit_button("Approve User")

            if approve_button:
                if not new_user or not new_password:
                    st.warning("Username and password are required.")
                else:
                    hashed_pw = new_password
                    users_ws.append_row([new_user, hashed_pw, str(is_admin), "TRUE"])
                    st.success(f"✅ {new_user} has been approved and added to the system.")

# --- FORMULÁRIO ---
def get_worksheet(sheet_name="Aurum_data"):
    gc = gspread.authorize(credentials)
    sh = gc.open_by_key("1HVYbot3Z9OBccBw7jKNw5acodwiQpfXgavDTIptSKic")
    return sh.worksheet(sheet_name)

if "user" in st.session_state:
    st.markdown("## Submit New Case to Aurum")
    with st.form("aurum_form"):
        case_id = st.text_input("Case #")
        n_seized = st.text_input("N seized specimens (e.g. 2 lion + 1 chimpanze)")
        year = st.number_input("Year", step=1, format="%d")
        country = st.text_input("Country of offenders")
        seizure_status = st.text_input("Seizure status")
        transit = st.text_input("Transit feature")
        notes = st.text_area("Additional notes")

        submitted = st.form_submit_button("Submit Case")

        if submitted:
            new_row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                case_id,
                n_seized,
                year,
                country,
                seizure_status,
                transit,
                notes,
                st.session_state["user"]
            ]
            worksheet = get_worksheet()
            worksheet.append_row(new_row)
            st.success("✅ Case submitted to Aurum successfully!")

    st.subheader("Upload Multiple Cases (Batch Mode)")
    uploaded_file = st.file_uploader("Upload an Excel or CSV file with multiple cases", type=["xlsx", "csv"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                batch_data = pd.read_csv(uploaded_file)
            else:
                batch_data = pd.read_excel(uploaded_file)

            # Preenche colunas obrigatórias
            batch_data = batch_data.fillna("")
            batch_data["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            batch_data["Author"] = st.session_state["user"]

            # Reordena colunas se necessário
            ordered_cols = [
                "Timestamp", "Case #", "N seized specimens", "Year",
                "Country of offenders", "Seizure status", "Transit feature",
                "Notes", "Author"
            ]
            for col in ordered_cols:
                if col not in batch_data.columns:
                    batch_data[col] = ""
            batch_data = batch_data[ordered_cols]

            rows_to_append = batch_data.values.tolist()
            worksheet = get_worksheet()
            worksheet.append_rows(rows_to_append, value_input_option="USER_ENTERED")

            st.success("✅ Batch upload completed successfully!")

        except Exception as e:
            st.error(f"❌ Error during upload: {e}")

    st.markdown("## My Cases")
    worksheet = get_worksheet()
    try:
        records = worksheet.get_all_records()
        if not records:
            st.info("No data available at the moment.")
        else:
            data = pd.DataFrame(records)
            if st.session_state.get("is_admin"):
                st.dataframe(data)
            else:
                st.dataframe(data[data["Author"] == st.session_state["user"]])
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")

st.sidebar.markdown("## 🕸️ Aurum Scraper (VIDA-style)")
activate_scraper = st.sidebar.checkbox("🔎 Open Aurum Scraper")

if activate_scraper:
    st.header("🐾 Aurum Scraper – VIDA-style Metadata Extractor")
    st.markdown("This tool extracts structured metadata from a product listing (e.g. Mercado Livre) and classifies its content.")

    target_url = st.text_input("🔗 Paste a product URL (e.g. Mercado Livre listing):")
    run_scraper = st.button("🚀 Extract & Classify")

    if run_scraper and target_url:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(target_url, headers=headers, timeout=15)
            html = response.text
            base_url = get_base_url(html, response.url)

            metadata = extruct.extract(
                html,
                base_url=base_url,
                syntaxes=["json-ld", "microdata", "opengraph", "rdfa"],
                uniform=True
            )

            # Attempt to extract key fields from JSON-LD
            data = {}
            for block in metadata.get("json-ld", []):
                if "@type" in block and block["@type"] in ["Product", "Offer"]:
                    data.update(block)

            title = data.get("name", "N/A")
            description = data.get("description", "N/A")
            price = data.get("offers", {}).get("price") if isinstance(data.get("offers"), dict) else "N/A"

            st.subheader("🔍 Extracted Data")
            st.write(f"**Title:** {title}")
            st.write(f"**Description:** {description}")
            st.write(f"**Price:** {price}")

            # Zero-shot classification (title + description)
            classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            input_text = f"{title}. {description}"
            labels = ["live wild animal", "beekeeping product", "trap or lure", "toy or decorative item", "uncertain"]
            result = classifier(input_text, labels)
            top_label = result["labels"][0]

            st.subheader("🤖 Classification")
            st.markdown(f"**Predicted category:** `{top_label}`")

            # Show results as table
            result_df = pd.DataFrame([{
                "URL": target_url,
                "Title": title,
                "Description": description,
                "Price": price,
                "Classification": top_label
            }])
            st.dataframe(result_df)

            st.download_button(
                "📥 Download CSV",
                result_df.to_csv(index=False).encode("utf-8"),
                "aurum_scraper_result.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"❌ Failed to extract or classify: {e}")

st.sidebar.markdown("## 🕸️ Aurum Scraper (Keyword Search)")
activate_scraper = st.sidebar.checkbox("🔎 Run Aurum Scraper – Mercado Livre Search")

if activate_scraper:
    st.header("🐾 Aurum Scraper – Mercado Livre Search")
    st.markdown("Searches Mercado Livre by keyword, extracts product data, and classifies listings.")

    keyword = st.text_input("🔤 Enter a keyword (e.g. 'enxame'):", value="enxame")
    max_pages = st.slider("🔢 Number of pages to scan", min_value=1, max_value=10, value=3)
    run_scan = st.button("🚀 Start search and analysis")

    if run_scan and keyword:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        labels = ["live wild animal", "beekeeping product", "trap or lure", "toy or decorative item", "uncertain"]

        headers = {"User-Agent": "Mozilla/5.0"}
        results = []

        with st.spinner("🔍 Scanning Mercado Livre..."):
            for page in range(1, max_pages + 1):
                page_suffix = f"_Desde_{(page-1)*50+1}" if page > 1 else ""
                search_url = f"https://lista.mercadolivre.com.br/{keyword}{page_suffix}"

                try:
                    res = requests.get(search_url, headers=headers, timeout=10)
                    soup = BeautifulSoup(res.content, "html.parser")
                    product_links = [a["href"] for a in soup.select("a.ui-search-link") if a["href"].startswith("https")]

                    for url in product_links:
                        try:
                            prod_res = requests.get(url, headers=headers, timeout=10)
                            html = prod_res.text
                            base_url = get_base_url(html, url)
                            metadata = extruct.extract(
                                html,
                                base_url=base_url,
                                syntaxes=["json-ld"],
                                uniform=True
                            )

                            data = {}
                            for block in metadata.get("json-ld", []):
                                if "@type" in block and block["@type"] in ["Product", "Offer"]:
                                    data.update(block)

                            title = data.get("name", "N/A")
                            description = data.get("description", "N/A")
                            price = data.get("offers", {}).get("price") if isinstance(data.get("offers"), dict) else "N/A"
                            input_text = f"{title}. {description}"
                            prediction = classifier(input_text, labels)
                            top_label = prediction["labels"][0]

                            results.append({
                                "URL": url,
                                "Title": title,
                                "Description": description,
                                "Price": price,
                                "Classification": top_label
                            })

                        except Exception as e:
                            st.warning(f"❌ Could not process ad: {e}")
                except Exception as e:
                    st.warning(f"⚠️ Could not load page {page}: {e}")

        if results:
            df_scraped = pd.DataFrame(results)
            st.success(f"✅ {len(df_scraped)} listings processed from Mercado Livre.")
            st.dataframe(df_scraped)

            st.download_button(
                "📥 Download results (.csv)",
                df_scraped.to_csv(index=False).encode("utf-8"),
                "aurum_scraper_keyword_results.csv",
                "text/csv"
            )
        else:
            st.warning("No results found or all failed.")

st.sidebar.markdown("---")    
st.sidebar.markdown("**How to cite:** Carvalho, A. F. Detecting Organized Wildlife Crime with *Aurum*: A Toolkit for Wildlife Trafficking Analysis. Wildlife Conservation Society, 2025.")
