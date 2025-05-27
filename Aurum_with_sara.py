import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import re
import requests
import unicodedata
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from networkx.algorithms.community import greedy_modularity_communities
from io import BytesIO
import base64
from itertools import combinations
from scipy.stats import chi2_contingency
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
import urllib.parse
from uuid import uuid4
from datetime import datetime, timedelta
import pytz
from streamlit_shadcn_ui import tabs
from streamlit_shadcn_ui import button
import streamlit.components.v1 as components
brt = pytz.timezone("America/Sao_Paulo")

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Aurum Dashboard", layout="wide")
st.title("Aurum - Criminal Intelligence in Wildlife Trafficking")

# Upload do arquivo
from PIL import Image
logo = Image.open("logo.png")
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.markdown("## Welcome to Aurum")
st.sidebar.markdown("Log in below to unlock multi-user tools.")
# --- SOBRE O AURUM (ABOUT) ---
# Inicializa estado
if "show_sidebar_about" not in st.session_state:
    st.session_state["show_sidebar_about"] = False

# Botão fixo na sidebar
about_toggle = st.sidebar.button("**About Aurum**")

# Alterna a visibilidade da seção
if about_toggle:
    st.session_state["show_sidebar_about"] = not st.session_state["show_sidebar_about"]

# Exibe o conteúdo "About Aurum" se ativado
if st.session_state["show_sidebar_about"]:
    import base64

    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    img_base64 = get_base64_image("WCS-Brasil.jpeg")  # Certifique-se de que está no mesmo diretório

    col1, col2 = st.columns([4, 1])  # Proporção de espaço entre texto e imagem

    with col1:
        st.markdown("## About Aurum")
        st.markdown("""
**Aurum** is a modular and interactive platform for **criminal intelligence in wildlife trafficking**. Developed by the Wildlife Conservation Society (WCS) – Brazil, it empowers analysts, researchers, and enforcement professionals with data-driven insights through a user-friendly interface.

The platform enables the upload and processing of case-level data and provides a suite of analytical tools, including:

- **Interactive Visualization**: Build customized plots and dashboards based on selected variables to support real-time analysis and reporting.
- **Trend Analysis**: Explore directional changes in seizure patterns using segmented regression (TCS) and detect significant deviations from historical averages with cumulative sum control charts (CUSUM).
- **Species Co-occurrence**: Identify statistically significant co-trafficking relationships between species using chi-square tests and network-based visualizations.
- **Anomaly Detection**: Detect atypical or high-impact cases using multiple outlier detection methods (Isolation Forest, LOF, DBSCAN, Mahalanobis distance, and Z-Score).
- **Criminal Network Analysis**: Reveal connections between cases based on shared attributes such as species or offender countries to infer coordination and logistical convergence.
- **Geospatial Analysis**: Map the spatial distribution of trafficking activity using Kernel Density Estimation (KDE) and interactive heatmaps. Analysts can filter by species and time periods to detect spatial hotspots, regional trends, and high-risk corridors.

**Aurum** bridges conservation data and investigative workflows, offering a scalable and field-ready platform for intelligence-led responses to wildlife crime.
        """)

    with col2:
        st.image("WCS-Brasil.jpeg", width=120)

    st.markdown("---")
    
st.sidebar.markdown("## 📂 Upload IWT Data")
uploaded_file = st.sidebar.file_uploader("**Upload your Excel file (.xlsx) containing wildlife trafficking data.**", type=["xlsx"])

st.sidebar.markdown("**Download Template**")
with open("Aurum_template.xlsx", "rb") as f:
    st.sidebar.download_button(
        label="Download a data template for wildlife trafficking analysis in Aurum",
        data=f,
        file_name="aurum_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

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

# --- Função para acessar worksheet de dados principais ---
def get_worksheet(name="Aurum_data"):
    return sheets.worksheet(name)

# --- Mensagem inicial caso nenhum arquivo tenha sido enviado e usuário não esteja logado ---
if uploaded_file is None:
    st.markdown("""
    **Aurum** is a criminal intelligence platform developed to support the monitoring and investigation of **illegal wildlife trade (IWT)**.
    By integrating advanced statistical methods and interactive visualizations, Aurum enables researchers, enforcement agencies, and conservation organizations to identify operational patterns and support data-driven responses to IWT.

    Click the small arrow at the top-left corner (>) to open the sidebar and upload your XLSX data file.

    For the full Aurum experience, please request access or log in if you already have an account.  
    Click **About Aurum** to learn more about each analysis module.
    """)

# --- ALERTAS PÚBLICOS (visível para todos, inclusive sem login) ---
def display_public_alerts_section(sheet_id):
    import folium
    from folium.map import Icon
    from folium import Popup, Marker
    from folium.plugins import MarkerCluster
    import geopandas as gpd
    from streamlit.components.v1 import html
    import re

    def parse_italics(text):
        return re.sub(r'_([^_]+)_', r'<em>\1</em>', str(text))

    st.markdown("## 🌍 Alert Board")
    st.caption("These alerts are publicly available and updated by verified users of the Aurum system.")
    st.markdown("### Wildlife Trafficking Alerts")

    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    credentials = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
    client = gspread.authorize(credentials)
    sheets = client.open_by_key(sheet_id)

    try:
        df_alerts = pd.DataFrame(sheets.worksheet("Alerts").get_all_records())
        df_updates = pd.DataFrame(sheets.worksheet("Alert Updates").get_all_records())
        df_updates.columns = [col.strip() for col in df_updates.columns]

        if df_alerts.empty or "Public" not in df_alerts.columns:
            st.info("No public alerts available.")
            return

        df_alerts = df_alerts[df_alerts["Public"].astype(str).str.strip().str.upper() == "TRUE"]
        df_alerts = df_alerts.dropna(subset=["Latitude", "Longitude"])
        df_alerts = df_alerts[df_alerts["Latitude"].astype(str).str.strip() != ""]
        df_alerts = df_alerts[df_alerts["Longitude"].astype(str).str.strip() != ""]
        df_alerts["Latitude"] = pd.to_numeric(df_alerts["Latitude"], errors="coerce")
        df_alerts["Longitude"] = pd.to_numeric(df_alerts["Longitude"], errors="coerce")
        df_alerts = df_alerts.dropna(subset=["Latitude", "Longitude"])

        if df_alerts.empty:
            st.info("No georeferenced alerts to display on the map.")
            return

        gdf = gpd.GeoDataFrame(df_alerts, geometry=gpd.points_from_xy(df_alerts["Longitude"], df_alerts["Latitude"]), crs="EPSG:4326")
        m = folium.Map(location=[0, 0], zoom_start=2)
        marker_cluster = MarkerCluster().add_to(m)

        for _, row in df_alerts.iterrows():
            title = parse_italics(row["Title"])
            description = parse_italics(row["Description"])
            species = parse_italics(row.get("Species", "—"))
            source_link = row.get("Source Link")

            popup_parts = [
                f"<b>{title}</b><br>",
                f"<b>Risk:</b> {row['Risk Level']}<br>",
                f"<b>Category:</b> {row['Category']}<br>",
                f"<b>Species:</b> {species}<br>",
                f"<b>Country:</b> {row.get('Country', '—')}<br>",
                f"<b>Submitted by:</b> {row.get('Display As', 'Anonymous')}<br>",
                f"<b>Date:</b> {row['Created At']}<br>",
                f"<p style='margin-top:5px'><b>Description:</b><br>{description}</p>"
            ]

            if source_link:
                popup_parts.append(f"<p><a href='{source_link}' target='_blank'>🔗 Source Link</a></p>")

            updates = df_updates[df_updates["Alert ID"] == row["Alert ID"]].sort_values("Timestamp")
            if not updates.empty:
                updates_html = "<hr><b>Updates:</b><div style='max-height: 140px; overflow-y: auto; margin-top: 4px; padding-right: 6px;'><ul style='padding-left: 15px; font-size: 12px; margin-bottom: 0;'>"
                for _, upd in updates.iterrows():
                    updates_html += f"<li style='margin-bottom: 6px;'><i>{upd['Timestamp']}</i> – <b>{upd['User']}</b>: {upd['Update Text'].replace('\n', '<br>')}</li>"
                updates_html += "</ul></div>"
                popup_parts.append(updates_html)

            popup_html = "".join(popup_parts)

            color = {"High": "red", "Medium": "orange", "Low": "blue"}.get(row["Risk Level"], "gray")

            Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=Popup(popup_html, max_width=350),
                icon=Icon(color=color, icon="exclamation-sign")
            ).add_to(marker_cluster)

        legend_html = '''
            <div style="position: fixed; bottom: 20px; right: 20px; z-index:9999; background-color: white;
                        padding: 10px 14px; border-radius: 6px; font-size: 13px; box-shadow: 2px 2px 8px rgba(0,0,0,0.25);
                        line-height: 1.5;">
                <strong>Risk Level</strong><br>
                <div style="margin-left: 4px;">
                    <span style='color:red;'>●</span> High<br>
                    <span style='color:orange;'>●</span> Medium<br>
                    <span style='color:blue;'>●</span> Low
                </div>
                <div style="margin-top: 6px; font-size: 11px; color: gray;">
                    Generated with <strong>Aurum</strong>
                </div>
            </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        info_toggle_html = '''
            <div style="position: absolute; top: 90px; left: 10px; z-index: 9999;">
                <button onclick="var box = document.getElementById('info-box'); box.style.display = (box.style.display === 'none') ? 'block' : 'none';"
                    style="background-color: #4a90e2; color: white; border: none; padding: 8px 12px; border-radius: 50%; font-size: 16px; cursor: pointer;">
                    ℹ️
                </button>
                <div id="info-box" style="display: none; margin-top: 10px; background-color: #fefefe;
                    padding: 12px 16px; border-radius: 8px; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
                    font-size: 13px; max-width: 280px; line-height: 1.5;">
                    <b>Guidelines for submitting a Wildlife Trafficking Alert in Aurum</b><br><br>
                    <b>Species:</b> Use scientific names with underscores for italics (e.g., <em>_Panthera onca_</em>).<br>
                    <b>Description:</b> Add what happened, who was involved, when and where, and the information source.<br>
                    <b>Coordinates:</b> Provide values like <code>Latitude: -9.43453</code> and <code>Longitude: 3.53433</code>.<br>
                    <b>Country:</b> Country or precise location of the event.<br>
                    <b>Source Link:</b> (Optional) Add a link to a post, article, or evidence.
                </div>
            </div>
        '''
        m.get_root().html.add_child(folium.Element(info_toggle_html))

        html(m.get_root().render(), height=600)

    except Exception as e:
        st.error(f"❌ Failed to load public alerts: {e}")

if "user" in st.session_state:
    display_public_alerts_section(SHEET_ID)

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
                matches = re.findall(r'(\d+)\s*([A-Z][a-z]+(?:_[a-z]+)+)', str(row.get('N seized specimens', '')))
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

            # País de origem dos infratores
            if "Country of offenders" in df.columns:
                df["Offender_value"] = df["Country of offenders"].apply(lambda x: score_countries(x, country_map))

            # País de apreensão ou envio
            if "Country of seizure or shipment" in df.columns:
                df["Seizure_value"] = df["Country of seizure or shipment"].apply(lambda x: score_countries(x, country_map))

        else:
            st.warning("⚠️ File country_offenders_values.csv not found. Country scoring skipped.")

        # Marcação de convergência logística
        if 'Case #' in df.columns and 'Species' in df.columns:
            species_per_case = df.groupby('Case #')['Species'].nunique()
            df['Logistic Convergence'] = df['Case #'].map(lambda x: "1" if species_per_case.get(x, 0) > 1 else "0")

        def normalize_text(text):
            if not isinstance(text, str):
                text = str(text)
            text = text.strip().lower()
            text = unicodedata.normalize("NFKD", text)
            text = re.sub(r'\\s+', ' ', text)
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

                # Detecta as espécies e permite seleção de cor personalizada
                if "Species" in df_selected.columns:
                    unique_species = df_selected["Species"].dropna().unique()
                    species_colors = {}

                    st.markdown("### Species Color Customization")
                    with st.expander("Customize species colors"):
                        for sp in unique_species:
                            default_color = "#1f77b4" if "Lear" in sp else "#ff7f0e"
                            species_colors[sp] = st.color_picker(f"Color for {sp}", value=default_color, key=f"color_{sp}")

                st.markdown("### Custom Chart")
                if chart_type == "Bar":
                    fig = px.bar(df_selected, x=x_axis, y=y_axis, color="Species", color_discrete_map=species_colors)
                elif chart_type == "Line":
                    fig = px.line(df_selected, x=x_axis, y=y_axis, color="Species", color_discrete_map=species_colors)
                elif chart_type == "Scatter":
                    fig = px.scatter(df_selected, x=x_axis, y=y_axis, color="Species", color_discrete_map=species_colors)
                elif chart_type == "Pie":
                    fig = px.pie(df_selected, names=x_axis, values=y_axis)

                st.plotly_chart(fig, use_container_width=True)
            
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

                    This analysis uses a binary presence-absence matrix for each selected species across all case IDs. For every species pair, a **chi-square test of independence** (or Fisher's exact test when needed) is performed to evaluate whether the observed co-occurrence is statistically significant beyond what would be expected by chance.

                    - A **2×2 contingency table** is generated for each pair.
                    - The **Chi² statistic** or **Fisher’s exact p-value** indicates whether co-occurrence is stronger or weaker than random chance.
                    - The **Cramér's V** coefficient provides a measure of association strength (ranging from 0 to 1).

                    The results are displayed in a sortable table and detailed outputs for each pair, including visualizations of contingency tables and statistical interpretations.
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
                            obs = table.values
                            chi2, p_chi, _, expected = chi2_contingency(obs)
                            use_fisher = (expected < 5).any()
                            if use_fisher:
                                from scipy.stats import fisher_exact
                                _, p = fisher_exact(obs)
                                test_used = "Fisher"
                                chi2 = None
                                cramers_v = None
                            else:
                                n = obs.sum()
                                p = p_chi
                                test_used = "Chi²"
                                cramers_v = (chi2 / n) ** 0.5
                            results.append({
                                "A": sp_a, "B": sp_b,
                                "Chi2": chi2, "p": p,
                                "Table": table,
                                "Expected": expected,
                                "Test": test_used,
                                "Cramers_V": cramers_v
                            })
                    return sorted(results, key=lambda x: x["p"])

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

                    # Tabela resumo interativa
                    summary_df = pd.DataFrame([
                        {
                            "Species A": r["A"],
                            "Species B": r["B"],
                            "Test": r["Test"],
                            "Chi²": r["Chi2"] if r["Chi2"] is not None else "-",
                            "p-value": r["p"],
                            "Cramér's V": r["Cramers_V"] if r["Cramers_V"] is not None else "-"
                        }
                        for r in co_results
                    ])
                    st.dataframe(summary_df)

                    # Resultados detalhados
                    for r in co_results:
                        st.markdown(f"**{r['A']} × {r['B']}**")
                        st.markdown(f"Test used: `{r['Test']}`")
                        st.dataframe(r["Table"])
                        if r["Expected"] is not None:
                            with st.expander("Expected Frequencies"):
                                st.dataframe(pd.DataFrame(r["Expected"], 
                                                          index=r["Table"].index, 
                                                          columns=r["Table"].columns))
                        if r["Chi2"] is not None:
                            st.markdown(f"Chi² = `{r['Chi2']:.2f}`")
                        st.markdown(f"p = `{r['p']:.4f}`")
                        if r["Cramers_V"] is not None:
                            st.markdown(f"Cramér's V = `{r['Cramers_V']:.3f}`")
                        interpret_cooccurrence(r["Table"], r["Chi2"], r["p"])
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
                selected_features = st.multiselect("Select numeric features for anomaly detection:", numeric_cols, default=["N_seized", "Year", "Seizure_value", "Offender_value"])

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

                st.markdown("This network connects cases that share attributes like species, offender country, or others you select.")

                default_features = ["Species", "Country of offenders"]
                network_features = st.multiselect(
                    "Select features to compare across cases:", 
                    options=[col for col in df_selected.columns if col != "Case #"],
                    default=default_features
                )

                if network_features:
                    case_feature_sets = (
                        df_selected
                        .groupby("Case #")[network_features]
                        .agg(lambda x: set(x.dropna()))
                        .apply(lambda row: set().union(*row), axis=1)
                    )

                    G = nx.Graph()

                    for case_id in case_feature_sets.index:
                        G.add_node(case_id)

                    case_ids = list(case_feature_sets.index)
                    for i in range(len(case_ids)):
                        for j in range(i + 1, len(case_ids)):
                            shared = case_feature_sets[case_ids[i]].intersection(case_feature_sets[case_ids[j]])
                            if shared:
                                G.add_edge(case_ids[i], case_ids[j], weight=len(shared))

                    if G.number_of_edges() == 0:
                        st.info("No connections were found between cases using the selected features.")
                    else:
                        degree_centrality = nx.degree_centrality(G)
                        betweenness_centrality = nx.betweenness_centrality(G)
                        closeness_centrality = nx.closeness_centrality(G)
                        try:
                            eigenvector_centrality = nx.eigenvector_centrality(G)
                        except nx.NetworkXError:
                            eigenvector_centrality = {n: 0 for n in G.nodes()}

                        communities = list(greedy_modularity_communities(G))
                        community_map = {node: i for i, comm in enumerate(communities) for node in comm}

                        st.markdown("### Community Color Customization")
                        community_ids = sorted(set(community_map.values()))
                        color_map = {}

                        with st.expander("Customize community colors"):
                            for cid in community_ids:
                                default_color = "#1f77b4" if cid == 0 else "#ff7f0e"
                                color = st.color_picker(
                                    f"Color for Community {cid}",
                                    value=default_color,
                                    key=f"comm_color_{cid}"
                                )
                                color_map[cid] = color

                        pos = nx.kamada_kawai_layout(G)

                        edge_x, edge_y = [], []
                        for u, v in G.edges():
                            x0, y0 = pos[u]
                            x1, y1 = pos[v]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])

                        edge_trace = go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=1.2, color='#CCCCCC'),
                            hoverinfo='none',
                            mode='lines'
                        )

                        node_x, node_y, node_color_rgb, node_size, node_text = [], [], [], [], []
                        for node in G.nodes():
                            x, y = pos[node]
                            deg = G.degree[node]
                            com = community_map.get(node, 0)
                            node_x.append(x)
                            node_y.append(y)
                            node_color_rgb.append(color_map.get(com, "#999999"))
                            node_size.append(8 + deg * 1.2)
                            node_text.append(f"Case {node}<br>Degree: {deg}<br>Community: {com}")

                        node_trace = go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers',
                            hoverinfo='text',
                            text=node_text,
                            marker=dict(
                                color=node_color_rgb,
                                size=node_size,
                                line_width=0.8
                            )
                        )

                        fig = go.Figure(
                            data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title='Case Connectivity Network',
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40),
                                annotations=[
                                    dict(
                                        text="Node size = degree. Node color = community (customizable).",
                                        showarrow=False,
                                        xref="paper", yref="paper",
                                        x=0.005, y=-0.002
                                    )
                                ]
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("### Community Summary")
                        st.markdown(f"**Total communities detected:** `{len(communities)}`")
                        for i, comm in enumerate(communities):
                            st.markdown(f"- Community `{i}`: {len(comm)} cases")

                        st.markdown("### Network Metrics")
                        num_nodes = G.number_of_nodes()
                        num_edges = G.number_of_edges()
                        density = nx.density(G)
                        components = nx.number_connected_components(G)
                        degrees = dict(G.degree())
                        avg_degree = sum(degrees.values()) / num_nodes if num_nodes else 0
                        try:
                            diameter = nx.diameter(G)
                            avg_shortest_path = nx.average_shortest_path_length(G)
                        except nx.NetworkXError:
                            diameter = None
                            avg_shortest_path = None

                        modularity = nx.algorithms.community.quality.modularity(G, communities)

                        st.write(f"- **Nodes:** {num_nodes}")
                        st.write(f"- **Edges:** {num_edges}")
                        st.write(f"- **Density:** {density:.3f}")
                        st.write(f"- **Connected components:** {components}")
                        st.write(f"- **Average degree:** {avg_degree:.2f}")
                        if diameter is not None:
                            st.write(f"- **Network diameter:** {diameter}")
                            st.write(f"- **Average shortest path length:** {avg_shortest_path:.2f}")
                        st.write(f"- **Modularity (community structure):** {modularity:.3f}")

                        st.markdown("### Top Central Cases")

                        def show_top(dictionary, title):
                            top_items = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)[:5]
                            st.markdown(f"**{title}:**")
                            for case, value in top_items:
                                st.markdown(f"- Case `{case}`: `{value:.3f}`")

                        col1, col2 = st.columns(2)
                        with col1:
                            show_top(degree_centrality, "Top Degree Centrality")
                            show_top(closeness_centrality, "Top Closeness Centrality")
                        with col2:
                            show_top(betweenness_centrality, "Top Betweenness Centrality")
                            show_top(eigenvector_centrality, "Top Eigenvector Centrality")

                        with st.expander("ℹ️ Learn more about this analysis"):
                            st.markdown("""
                                ### About Case Network Analysis

                                This section visualizes a network of wildlife trafficking cases based on **shared attributes** such as species, offender countries, or other selected variables.

                                - **Each node represents a unique case** (`Case #`).
                                - **An edge connects two cases that share one or more selected attributes**, such as:
                                    - The same species involved,
                                    - The same offender country,
                                    - Or any other field selected in the sidebar (e.g., seizure location, transport method).
                                - **Node size** reflects the number of connections a case has (degree centrality).
                                - **Node color** reflects its **community**, detected automatically using modularity-based clustering.
                                - **Edge thickness** reflects the **strength of similarity**, i.e., how many attributes two cases have in common.

                                This type of network allows you to:

                                - **Identify clusters** of related cases, which may indicate organized groups or repeated patterns.
                                - **Spot intermediaries or bridges** — cases that connect otherwise distant groups (high betweenness centrality).
                                - **Detect influential hubs** — cases linked to others that are also highly connected (eigenvector centrality).
                                - **Assess the structure of trafficking dynamics**, including how centralized or fragmented the case network is.

                                You can select which attributes to include in the sidebar to dynamically reshape the network based on your analytical needs.

                                ---
                                **Example:**  
                                If two cases both involve *Panthera onca* and originate from Brazil, they will be connected.  
                                If a third case only shares the species but not the country, the connection will still be drawn, but with lower weight.

                                For deeper interpretation, use the metrics above and explore the top-ranking cases by different centralities.
                            """)

                else:
                    st.info("Please select at least one feature to define connections between cases.")
            
            show_geo = st.sidebar.checkbox("Geospatial Analysis", value=False)
            if show_geo:
                st.markdown("## Geospatial Analysis")

                st.markdown("This section maps the spatial distribution of wildlife trafficking cases using kernel density estimation (KDE) and interactive heatmaps.")

                if 'Latitude' not in df_selected.columns or 'Longitude' not in df_selected.columns:
                    st.warning("This analysis requires columns 'Latitude' and 'Longitude' in the dataset.")
                else:
                    df_geo = df_selected.dropna(subset=["Latitude", "Longitude"]).copy()
                    if df_geo.empty:
                        st.warning("No valid geolocation data found.")
                    else:
                        import geopandas as gpd
                        import numpy as np
                        import folium
                        from folium.plugins import HeatMap
                        import os
                        import tempfile

                        st.markdown("### Analysis Settings")
                        col1, col2 = st.columns(2)

                        with col1:
                            species_list = sorted(df_geo['Species'].dropna().unique().tolist())
                            selected_species = st.multiselect("Filter by species:", species_list, default=species_list, key="species_dashboard")
                            df_geo = df_geo[df_geo['Species'].isin(selected_species)]

                        with col2:
                            radius_val = st.slider("HeatMap radius (px)", 5, 50, 25)

                        st.markdown("### Temporal Filter")

                        if 'Year' in df_geo.columns:
                            min_year = int(df_geo['Year'].min())
                            max_year = int(df_geo['Year'].max())

                            col1, col2 = st.columns([1, 3])

                            with col1:
                                temporal_mode = st.radio("Mode", ["Year Range", "Single Year"], index=0)

                            with col2:
                                if temporal_mode == "Year Range":
                                    selected_years = st.slider(
                                        "",  # ocultar label para visual compacto
                                        min_value=min_year,
                                        max_value=max_year,
                                        value=(min_year, max_year),
                                        step=1,
                                        label_visibility="collapsed"
                                    )
                                    df_geo = df_geo[df_geo['Year'].between(selected_years[0], selected_years[1])]
                                    st.markdown(f"📆 **{selected_years[0]}** to **{selected_years[1]}**")

                                elif temporal_mode == "Single Year":
                                    unique_years = sorted(df_geo['Year'].dropna().unique().tolist())
                                    selected_year = st.select_slider(
                                        "", 
                                        options=unique_years,
                                        value=max_year,
                                        label_visibility="collapsed"
                                    )
                                    df_geo = df_geo[df_geo['Year'] == selected_year]
                                    st.markdown(f"📆 Year: **{selected_year}**")


                        # Mapa Interativo
                        df_geo_unique = df_geo.drop_duplicates(subset=["Case #", "Latitude", "Longitude"])
                        gdf = gpd.GeoDataFrame(
                            df_geo_unique,
                            geometry=gpd.points_from_xy(df_geo_unique["Longitude"], df_geo_unique["Latitude"]),
                            crs="EPSG:4326"
                        )

                        gdf_wgs = gdf.to_crs(epsg=4326)
                        bounds = gdf_wgs.total_bounds
                        center_lat = (bounds[1] + bounds[3]) / 2
                        center_lon = (bounds[0] + bounds[2]) / 2

                        m = folium.Map(location=[center_lat, center_lon], zoom_start=2)
                        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                        HeatMap(data=gdf_wgs[['Latitude', 'Longitude']].values, radius=radius_val).add_to(m)

                        # Legenda avançada com gradiente contínuo
                        legend_html = '''
                            <div style="
                                position: fixed;
                                bottom: 40px;
                                right: 20px;
                                z-index: 9999;
                                background-color: white;
                                padding: 10px;
                                border:2px solid gray;
                                border-radius:5px;
                                font-size:14px;
                                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
                                <b>HeatMap Intensity</b>
                                <div style="height: 10px; width: 120px;
                                    background: linear-gradient(to right, blue, cyan, lime, yellow, orange, red);
                                    margin: 5px 0;"></div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span>Low</span>
                                    <span>Medium</span>
                                    <span>High</span>
                                </div>
                                <div style="margin-top:6px; font-size:10px; color:gray;">Generated with Aurum</div>
                            </div>
                        '''
                        m.get_root().html.add_child(folium.Element(legend_html))

                        html_str = m.get_root().render()
                        st.components.v1.html(html_str, height=800)

                        full_map_path = os.path.join(tempfile.gettempdir(), "aurum_map.html")
                        m.save(full_map_path)

                        with open(full_map_path, "rb") as f:
                            btn_data = f.read()
                        st.download_button(
                            label="Download interactive map (.html)",
                            data=btn_data,
                            file_name="aurum_kde_map.html",
                            mime="text/html"
                        )

                        with st.expander("ℹ️ Learn more about this analysis"):
                            st.markdown("""
                                ### About Geospatial Analysis

                                This section uses **HeatMap visualization** to identify **spatial hotspots** of wildlife trafficking activity.

                                - HeatMap is based on geographic coordinates of cases.
                                - Radius controls the smoothing around each point (in pixels).
                                - You can filter by species and time to explore dynamics interactively.

                                ---
                                **Use Cases:**
                                - Detect trafficking corridors or hubs.
                                - Compare species-specific patterns geographically.
                                - Support decision-making for targeted enforcement or prevention.

                                **Note:** This map uses geographic coordinates (EPSG:4326) and is rendered using `folium`.
                            """)
                                
        else:
            st.warning("⚠️ Please select at least one species to explore the data.")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def place_logo_bottom_right(image_path, width=70, link_url="https://wcs.org/"):
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
            <a href="{link_url}" target="_blank">
                <img src="data:image/png;base64,{img_base64}" width="{width}"/>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Chamada da função para exibir a logo com link
place_logo_bottom_right("wcs.png")

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
    now = datetime.now(brt).strftime("%Y-%m-%d %H:%M:%S (BRT)")

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

        # Salvar figura se existir
        if 'fig' in locals() and hasattr(fig, "savefig"):
            trend_buf = BytesIO()
            fig.savefig(trend_buf, format="png", bbox_inches="tight")
            trend_buf.seek(0)
            trend_base64 = base64.b64encode(trend_buf.read()).decode("utf-8")
            html_sections.append(f'<img src="data:image/png;base64,{trend_base64}" width="700">')
        else:
            html_sections.append("<p><i>No trend figure available.</i></p>")

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

# --- Função com cache curto para carregar os dados de usuários ---
@st.cache_data(ttl=30)
def load_users_data():
    return pd.DataFrame(users_ws.get_all_records())

# --- Função para registrar sessões (apenas após login validado) ---
def log_session(username, email):
    try:
        session_ws = get_worksheet("Sessions")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_ws.append_row([username, email, timestamp])
    except Exception as e:
        st.warning(f"\u26a0\ufe0f Failed to log session: {e}")

# --- LOGIN ---
st.sidebar.markdown("---")

if "user" in st.session_state:
    st.sidebar.markdown(f"✅ **{st.session_state['user']}** is connected.")
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

else:
    st.sidebar.markdown("## 🔐 Login to Aurum")

    if "login_username" not in st.session_state:
        st.session_state["login_username"] = ""
    if "login_password" not in st.session_state:
        st.session_state["login_password"] = ""

    username = st.sidebar.text_input("Username", key="login_username")
    password = st.sidebar.text_input("Password", type="password", key="login_password")

    login_col, _ = st.sidebar.columns([1, 1])
    login_button = login_col.button("Login")

    def verify_password(password, hashed):
        return password == hashed

    if login_button and username and password:
        # ← AQUI: só carrega os dados após o clique no botão
        users_df = load_users_data()
        user_row = users_df[users_df["Username"] == username]

        if not user_row.empty and str(user_row.iloc[0]["Approved"]).strip().lower() == "true":
            hashed_pw = user_row.iloc[0]["Password"].strip()

            if verify_password(password, hashed_pw):
                # Armazena dados da sessão
                st.session_state["user"] = username
                st.session_state["user_email"] = user_row.iloc[0]["E-Mail"]
                st.session_state["is_admin"] = str(user_row.iloc[0]["Is_Admin"]).strip().lower() == "true"

                # Registra sessão apenas uma vez
                if "session_logged" not in st.session_state:
                    log_session(username, st.session_state["user_email"])
                    st.session_state["session_logged"] = True

                # Limpa campos de login
                st.session_state.pop("login_username", None)
                st.session_state.pop("login_password", None)

                st.rerun()
            else:
                st.error("Incorrect password.")
        else:
            st.error("User not approved or does not exist.")

# --- FORMULÁRIO DE ACESSO (REQUISIÇÃO) ---
# Inicializa estado
if "show_sidebar_request" not in st.session_state:
    st.session_state["show_sidebar_request"] = False

# Botão fixo na sidebar
request_toggle = st.sidebar.button("📩 Request Access")

# Alterna a visibilidade do formulário
if request_toggle:
    st.session_state["show_sidebar_request"] = not st.session_state["show_sidebar_request"]

# Exibe o formulário de request access se ativado
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
                timestamp = datetime.now(brt).strftime("%Y-%m-%d %H:%M:%S (BRT)")
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
                st.rerun()  # Atualiza visualmente após envio

# --- DADOS DE ENTRADA (substitua pelos reais) ---
request_df = pd.DataFrame(requests_ws.get_all_records())
users_df = pd.DataFrame(users_ws.get_all_records())

# --- Verifica se é admin ---
if st.session_state.get("is_admin"):

    # Exibe a tab de forma clicável
    selected_tab = tabs(
        options=["Admin Panel"],
        default_value="",
        key="admin_tab"
    )

    # Conteúdo só aparece após o clique
    if selected_tab == "Admin Panel":
        st.markdown("## 🛡️ Admin Panel - Approve Access Requests")

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
                        try:
                            user_row = request_df[request_df["Username"] == new_user]
                            if user_row.empty:
                                st.warning("User not found in access requests.")
                            else:
                                row_index = user_row.index[0]
                                is_admin_str = "TRUE" if is_admin else "FALSE"
                                email = user_row.iloc[0]["E-mail"].strip()

                                # Atualiza aba Access Requests
                                requests_ws.update_cell(row_index + 2, request_df.columns.get_loc("Approved") + 1, "TRUE")
                                requests_ws.update_cell(row_index + 2, request_df.columns.get_loc("Is_Admin") + 1, is_admin_str)

                                # Adiciona na aba Users se ainda não estiver
                                if new_user not in users_df["Username"].values:
                                    users_ws.append_row([
                                        new_user,
                                        new_password,
                                        email,
                                        is_admin_str,
                                        "TRUE"
                                    ])

                                st.success(f"✅ {new_user} has been approved and added to the system.")
                                st.info("🔐 The user is now authorized to log into Aurum.")
                        except Exception as e:
                            st.error(f"❌ Failed to approve user: {e}")

# --- FORMULÁRIO ---
def get_worksheet(sheet_name="Aurum_data"):
    gc = gspread.authorize(credentials)
    sh = gc.open_by_key("1HVYbot3Z9OBccBw7jKNw5acodwiQpfXgavDTIptSKic")
    return sh.worksheet(sheet_name)

@st.cache_data(ttl=60)
def load_all_records(sheet_name="Aurum_data"):
    worksheet = get_worksheet(sheet_name)
    return worksheet.get_all_records()
    
# --- Função para carregar dados de qualquer aba ---
def load_sheet_data(sheet_name, sheets):
    try:
        worksheet = sheets.worksheet(sheet_name)
        records = worksheet.get_all_records()
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"❌ Failed to load data from sheet '{sheet_name}': {e}")
        return pd.DataFrame()

# --- Função para submissão de alertas ---
def display_alert_submission_form(sheet_id):
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    credentials = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
    client = gspread.authorize(credentials)
    sheets = client.open_by_key(sheet_id)

    field_keys = {
        "title": "alert_title_input",
        "description": "alert_description_input",
        "category": "alert_category_select",
        "risk_level": "alert_risk_select",
        "species": "alert_species_input",
        "country": "alert_country_input",
        "source_link": "alert_source_input",
        "latitude": "alert_lat_input",
        "longitude": "alert_lon_input",
        "author_choice": "alert_author_choice"
    }

    categories = ["Species", "Country", "Marketplace", "Operation", "Policy", "Other"]
    risk_levels = ["Low", "Medium", "High"]

    for key in ["title", "description", "species", "country", "source_link", "latitude", "longitude"]:
        st.session_state.setdefault(field_keys[key], "")

    if st.session_state.get(field_keys["category"]) not in categories:
        st.session_state[field_keys["category"]] = categories[0]
    if st.session_state.get(field_keys["risk_level"]) not in risk_levels:
        st.session_state[field_keys["risk_level"]] = risk_levels[0]

    st.session_state.setdefault(field_keys["author_choice"], "Show my username")

    with st.form("alert_form"):
        title = st.text_input("Alert Title", key=field_keys["title"])
        description = st.text_area("Alert Description", key=field_keys["description"])
        category = st.selectbox("Category", categories, key=field_keys["category"])
        risk_level = st.selectbox("Risk Level", risk_levels, key=field_keys["risk_level"])
        species = st.text_input("Species involved (optional)", key=field_keys["species"])
        country = st.text_input("Country or Region (optional)", key=field_keys["country"])
        source_link = st.text_input("Source Link (optional)", key=field_keys["source_link"])
        latitude = st.text_input("Latitude (e.g., -9.5342)", key=field_keys["latitude"])
        longitude = st.text_input("Longitude (e.g., -43.3525)", key=field_keys["longitude"])

        st.caption(
            "Only alerts with valid coordinates (latitude and longitude) will appear on the map.\n"
            "By submitting, you agree to make this information publicly visible to all Aurum users."
        )
        
        author_choice = st.radio(
            "Choose how to display your name:",
            ["Show my username", "Submit anonymously"],
            key=field_keys["author_choice"]
        )

        created_by = st.session_state["user_email"]
        display_as = st.session_state["user"] if author_choice == "Show my username" else "Anonymous"

        submitted = st.form_submit_button("📤 Submit Alert")

    if submitted:
        if not title or not description:
            st.warning("Title and Description are required.")
        else:
            # Validação opcional de latitude/longitude
            lat_val = st.session_state[field_keys["latitude"]].strip()
            lon_val = st.session_state[field_keys["longitude"]].strip()
            try:
                lat = float(lat_val) if lat_val else ""
                lon = float(lon_val) if lon_val else ""
            except ValueError:
                st.warning("Latitude and Longitude must be numeric.")
                return

            alert_id = str(uuid4())
            created_at = datetime.now(brt).strftime("%Y-%m-%d %H:%M:%S (BRT)")
            public = True

            alert_row = [
                alert_id, created_at, created_by, display_as, title, description,
                category, species, country, risk_level, source_link,
                lat, lon,  # inseridos na posição correta
                str(public)
            ]

            try:
                worksheet = sheets.worksheet("Alerts")
                worksheet.append_row(alert_row, value_input_option="USER_ENTERED")
                st.success("✅ Alert submitted successfully!")
                st.balloons()

                for k in field_keys.values():
                    if k in st.session_state:
                        del st.session_state[k]                
                st.rerun()
                st.stop()

            except Exception as e:
                st.error(f"❌ Failed to submit alert: {e}")

def display_alert_update_timeline(sheet_id):
    scope = ["https://www.googleapis.com/auth/spreadsheets"]
    credentials = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
    client = gspread.authorize(credentials)
    sheets = client.open_by_key(sheet_id)

    try:
        df_alerts = pd.DataFrame(sheets.worksheet("Alerts").get_all_records())
        df_alerts.columns = [col.strip() for col in df_alerts.columns]

        df_updates = pd.DataFrame(sheets.worksheet("Alert Updates").get_all_records())
        if df_updates.empty:
            df_updates = pd.DataFrame(columns=["Alert ID", "Timestamp", "User", "Update Text"])
        else:
            df_updates.columns = [col.strip() for col in df_updates.columns]

        user = st.session_state.get("user")
        user_email = st.session_state.get("user_email")

        # Alertas criados pelo usuário (usando o e-mail como referência)
        created_alerts = df_alerts[df_alerts["Created By"] == user_email]

        # Alertas que o usuário atualizou (usando username nos updates)
        relevant_updates = df_updates[
            (df_updates["User"] == user) | (df_updates["User"] == "Anonymous")
        ]
        updated_alert_ids = relevant_updates["Alert ID"].unique()
        updated_alerts = df_alerts[df_alerts["Alert ID"].isin(updated_alert_ids)]

        # Junta alertas criados e alertas atualizados
        df_user_alerts = pd.concat([created_alerts, updated_alerts]).drop_duplicates(subset="Alert ID")

        if df_user_alerts.empty:
            st.info("You haven't submitted or updated any alerts yet.")
            return

        # 🔵 Fixar seleção do alerta no session_state
        if "selected_alert_id" not in st.session_state:
            st.session_state["selected_alert_id"] = None

        selected_title = st.selectbox(
            "Select an alert to update:",
            df_user_alerts["Title"].tolist()
        )

        selected_row = df_user_alerts[df_user_alerts["Title"] == selected_title].iloc[0]
        alert_id = selected_row["Alert ID"]

        # 🔵 Salva o alerta selecionado fixo
        st.session_state["selected_alert_id"] = alert_id

        timeline = df_updates[df_updates["Alert ID"] == alert_id].sort_values("Timestamp")

        if not timeline.empty:
            st.markdown("### Update Timeline")
            for _, row in timeline.iterrows():
                st.markdown(f"**{row['Timestamp']}** – *{row['User']}*: {row['Update Text']}")
        else:
            st.info("This alert has no updates yet.")

        update_author_choice = st.radio(
            "Choose how to display your name in this update:",
            ["Show my username", "Submit anonymously"],
            key="update_author_choice"
        )
        update_user = user if update_author_choice == "Show my username" else "Anonymous"

        update_text_key = f"update_text_{alert_id}"
        field_keys = {"update_text": update_text_key}

        with st.form(f"update_form_{alert_id}"):
            st.markdown("**Add a new update to this alert:**")
            new_update = st.text_area("Update Description", key=update_text_key)
            submitted = st.form_submit_button("➕ Add Update")

            if submitted:
                if not new_update.strip():
                    st.warning("Update description is required.")
                else:
                    timestamp = datetime.now(brt).strftime("%Y-%m-%d %H:%M:%S (BRT)")
                    update_row = [alert_id, timestamp, update_user, new_update.strip()]

                    try:
                        try:
                            update_ws = sheets.worksheet("Alert Updates")
                        except gspread.exceptions.WorksheetNotFound:
                            update_ws = sheets.add_worksheet(title="Alert Updates", rows="1000", cols="4")
                            update_ws.append_row(["Alert ID", "Timestamp", "User", "Update Text"])

                        update_ws.append_row(update_row)
                        st.success("✅ Update added to alert!")

                        # 🔥 Apaga o campo de texto depois de submeter
                        for k in field_keys.values():
                            if k in st.session_state:
                                del st.session_state[k]

                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Failed to add update: {e}")

    except Exception as e:
        st.error(f"❌ Could not load alerts or updates: {e}")

# --- Interface em colunas: Alertas (superior) e Casos (inferior) ---
if "user" in st.session_state:
    
    # --- MENU SUPERIOR COM TABS ---
    selected_tab = tabs(
        options=["Alerts Management", "Cases Management", "Data Requests", "Collaboration"],
        default_value="",
        key="main_tab"
    )

    # ----------------------------
    # 🔔 ALERTS MANAGEMENT
    # ----------------------------
    if selected_tab == "Alerts Management":
        st.markdown("### Alerts Management")
        col1, col2 = st.columns(2)

        with col1:
            with st.expander("**Submit New Alert**", expanded=False):
                display_alert_submission_form(SHEET_ID)

        with col2:
            with st.expander("**Update My Alerts**", expanded=False):
                display_alert_update_timeline(SHEET_ID)

    # ----------------------------
    # 📁 CASES MANAGEMENT
    # ----------------------------
    elif selected_tab == "Cases Management":
        st.markdown("### Case Management")
        col3, col4 = st.columns(2)

        # --- Submit new case ---
        with col3:
            with st.expander("**Submit New Case**", expanded=False):
                field_keys = {
                    "case_id": "case_id_input",
                    "n_seized": "n_seized_input",
                    "year": "year_input",
                    "country": "country_input",
                    "seizure_status": "seizure_status_input",
                    "transit": "transit_input",
                    "notes": "notes_input",
                    "latitude": "latitude_input",
                    "longitude": "longitude_input",
                    "kg": "kg_input",
                    "parts": "parts_input"
                }

                default_values = {
                    "case_id": "",
                    "n_seized": "",
                    "year": 2024,
                    "country": "",
                    "seizure_status": "",
                    "transit": "",
                    "notes": "",
                    "latitude": "",
                    "longitude": "",
                    "kg": "",
                    "parts": ""
                }

                for key, default in default_values.items():
                    st.session_state.setdefault(field_keys[key], default)

                with st.form("aurum_form"):
                    case_id = st.text_input("Case #", key=field_keys["case_id"])
                    seizure_country = st.text_input("Country of seizure or shipment")
                    n_seized = st.text_input("N seized specimens", key=field_keys["n_seized"])

                    year_value = st.session_state.get(field_keys["year"], 2024)
                    try:
                        year_value = int(year_value)
                    except:
                        year_value = 2024
                    year = st.number_input("Year", step=1, min_value=1900, max_value=2100, value=year_value, key=field_keys["year"])

                    country = st.text_input("Country of offenders", key=field_keys["country"])
                    seizure_status = st.text_input("Seizure status", key=field_keys["seizure_status"])
                    transit = st.text_input("Transit feature", key=field_keys["transit"])
                    notes = st.text_area("Additional notes", key=field_keys["notes"])
                    latitude = st.text_input("Latitude", key=field_keys["latitude"])
                    longitude = st.text_input("Longitude", key=field_keys["longitude"])

                    # Extração automática de kg e partes
                    import re
                    kg_matches = re.findall(r"(\d+(?:\.\d+)?)\s*kg", n_seized)
                    parts_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:part|parts|fangs|horns|claws|feathers|scales|shells)", n_seized, re.IGNORECASE)
                    total_kg = sum(float(k) for k in kg_matches)
                    total_parts = sum(float(p) for p in parts_matches)

                    kg = st.text_input("Estimated weight (kg)", value=str(total_kg) if total_kg else "", key=field_keys["kg"])
                    parts = st.text_input("Animal parts seized", value=str(total_parts) if total_parts else "", key=field_keys["parts"])

                    submitted = st.form_submit_button("Submit Case")

                if submitted:
                    new_row = [
                        datetime.now(brt).strftime("%Y-%m-%d %H:%M:%S (BRT)"),
                        case_id,
                        seizure_country,
                        n_seized,
                        year,
                        country,
                        seizure_status,
                        transit,
                        notes,
                        latitude,
                        longitude,
                        kg,
                        parts,
                        st.session_state["user"]
                    ]
                    worksheet = get_worksheet()
                    worksheet.append_row(new_row)
                    st.success("✅ Case submitted to Aurum successfully!")
                    for k in field_keys.values():
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()

        # --- Edit cases ---
        with col4:
            with st.expander("**Edit My Cases**", expanded=False):
                try:
                    records = load_all_records()
                    df_user = pd.DataFrame(records)
                    df_user = df_user[df_user["Author"] == st.session_state["user"]]

                    if df_user.empty:
                        st.info("You haven't submitted any cases yet.")
                    else:
                        selected_case = st.selectbox("Select a case to edit:", df_user["Case #"].unique())
                        if selected_case:
                            row_index = df_user[df_user["Case #"] == selected_case].index[0] + 2
                            current_row = df_user.loc[df_user["Case #"] == selected_case].iloc[0]

                            with st.form("edit_case_form"):
                                new_case_id = st.text_input("Case #", value=current_row["Case #"])
                                new_seizure_country = st.text_input("Country of seizure or shipment", value=current_row["Country of seizure or shipment"])
                                new_n_seized = st.text_input("N seized specimens", value=current_row["N seized specimens"])
                                new_year = st.number_input("Year", step=1, min_value=1900, max_value=2100, value=int(current_row["Year"]))
                                new_country = st.text_input("Country of offenders", value=current_row["Country of offenders"])
                                new_status = st.text_input("Seizure status", value=current_row["Seizure status"])
                                new_transit = st.text_input("Transit feature", value=current_row["Transit feature"])
                                new_notes = st.text_area("Additional notes", value=current_row["Notes"])
                                new_lat = st.text_input("Latitude", value=current_row.get("Latitude", ""))
                                new_lon = st.text_input("Longitude", value=current_row.get("Longitude", ""))

                                kg_matches_edit = re.findall(r"(\d+(?:\.\d+)?)\s*kg", new_n_seized)
                                parts_matches_edit = re.findall(r"(\d+(?:\.\d+)?)\s*(?:part|parts|fangs|horns|claws|feathers|scales|shells)", new_n_seized, re.IGNORECASE)
                                new_kg = st.text_input("Estimated weight (kg)", value=str(sum(float(k) for k in kg_matches_edit)) if kg_matches_edit else current_row.get("Estimated weight (kg)", ""))
                                new_parts = st.text_input("Animal parts seized", value=str(sum(float(p) for p in parts_matches_edit)) if parts_matches_edit else current_row.get("Animal parts seized", ""))

                                submitted_edit = st.form_submit_button("Save Changes")

                            if submitted_edit:
                                updated_row = [
                                    current_row["Timestamp"],
                                    new_case_id,
                                    new_seizure_country,
                                    new_n_seized,
                                    new_year,
                                    new_country,
                                    new_status,
                                    new_transit,
                                    new_notes,
                                    new_lat,
                                    new_lon,
                                    new_kg,
                                    new_parts,
                                    st.session_state["user"]
                                ]
                                worksheet = get_worksheet()
                                worksheet.update(f"A{row_index}:N{row_index}", [updated_row])
                                st.success("✅ Case updated successfully!")
                                for k in field_keys.values():
                                    if k in st.session_state:
                                        del st.session_state[k]
                                st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to load or update your cases: {e}")
                    
        # --- Upload múltiplo ---
        st.subheader("Upload Multiple Cases (Batch Mode)")
        uploaded_file_batch = st.file_uploader("Upload an Excel or CSV file with multiple cases", type=["xlsx", "csv"], key="uploaded_file_batch")

        if uploaded_file_batch is not None:
            st.info("📄 File uploaded. Click the button below to confirm batch submission.")
            submit_batch = st.button("📥 **Submit Batch Upload**")

            if submit_batch:
                try:
                    if uploaded_file_batch.name.endswith(".csv"):
                        batch_data = pd.read_csv(uploaded_file_batch)
                    else:
                        batch_data = pd.read_excel(uploaded_file_batch)

                    batch_data.columns = (
                        batch_data.columns
                        .str.normalize('NFKD')
                        .str.encode('ascii', errors='ignore')
                        .str.decode('utf-8')
                        .str.strip()
                        .str.lower()
                    )

                    required_cols_original = [
                        "Case #", "Country of seizure or shipment", "N seized specimens", "Year",
                        "Country of offenders", "Seizure status", "Transit feature", "Notes",
                        "Latitude", "Longitude", "Estimated weight (kg)", "Animal parts seized"
                    ]
                    required_cols_normalized = [col.lower() for col in required_cols_original]

                    missing_cols = [
                        orig for orig, norm in zip(required_cols_original, required_cols_normalized)
                        if norm not in batch_data.columns
                    ]

                    if missing_cols:
                        st.error("🚫 Upload blocked: the uploaded file has incorrect formatting.")
                        st.markdown(f"""**Missing columns**: {', '.join(missing_cols)}""")
                    else:
                        batch_data = batch_data.fillna("")
                        batch_data["Timestamp"] = datetime.now(brt).strftime("%Y-%m-%d %H:%M:%S (BRT)")
                        batch_data["Author"] = st.session_state["user"]
                        rename_map = dict(zip(required_cols_normalized, required_cols_original))
                        batch_data.rename(columns=rename_map, inplace=True)
                        ordered_cols = ["Timestamp"] + required_cols_original + ["Author"]
                        batch_data = batch_data[ordered_cols]
                        worksheet = get_worksheet()
                        worksheet.append_rows(batch_data.values.tolist(), value_input_option="USER_ENTERED")
                        st.success("✅ Batch upload completed successfully!")
                        del st.session_state["uploaded_file_batch"]
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error during upload: {e}")

        # --- Visualização dos próprios casos ---
        st.markdown("## My Cases")
        try:
            records = load_all_records()
            if not records:
                st.info("No data available at the moment.")
            else:
                data = pd.DataFrame(records)

                if not st.session_state.get("is_admin"):
                    data = data[data["Author"] == st.session_state["user"]]

                if "N seized specimens" in data.columns:
                    species_matches = data["N seized specimens"].str.extractall(r'\d+\s*([A-Z][a-z]+(?:_[a-z]+)+)')
                    species_list = sorted(species_matches[0].dropna().unique())
                    selected_species = st.multiselect("Filter by species:", species_list)
                    if selected_species:
                        data = data[data["N seized specimens"].str.contains("|".join(selected_species))]

                # Garante que as novas colunas existam (caso contrário, cria vazias para evitar erros)
                for col in ["Latitude", "Longitude", "Estimated weight (kg)", "Animal parts seized"]:
                    if col not in data.columns:
                        data[col] = ""

                st.dataframe(data)

        except Exception as e:
            st.error(f"❌ Failed to load data: {e}")

    # ----------------------------
    # 📊 DATA REQUESTS
    # ----------------------------
    elif selected_tab == "Data Requests":
        st.markdown("## Data Requests")
        st.markdown("Use this form to request access to datasets uploaded to Aurum.")

        species_key = "datareq_species"
        years_key = "datareq_years"
        country_key = "datareq_country"
        reason_key = "datareq_reason"

        if st.session_state.get("datareq_submitted_success"):
            st.success("✅ Your data request was submitted successfully.")
            del st.session_state["datareq_submitted_success"]

        if st.session_state.get("datareq_submitted_reset"):
            st.session_state[species_key] = ""
            st.session_state[years_key] = ""
            st.session_state[country_key] = ""
            st.session_state[reason_key] = ""
            del st.session_state["datareq_submitted_reset"]

        with st.form("data_request_form"):
            species = st.text_input("Species of interest (e.g., _Anodorhynchus leari_)", key=species_key)
            years = st.text_input("Year(s) of interest (e.g., 2022 or 2015–2020)", key=years_key)
            country = st.text_input("Country or region of interest (e.g., All or Brazil or South America)", key=country_key)
            reason = st.text_area("Justify your request:", key=reason_key)

            submitted = st.form_submit_button("Submit Data Request")

            if submitted:
                if not species or not years or not reason:
                    st.warning("Species, year(s), and justification are required.")
                else:
                    try:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        scope = ["https://www.googleapis.com/auth/spreadsheets"]
                        credentials = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
                        client = gspread.authorize(credentials)
                        sheet = client.open_by_key(SHEET_ID)

                        try:
                            req_ws = sheet.worksheet("Data Requests")
                        except gspread.exceptions.WorksheetNotFound:
                            req_ws = sheet.add_worksheet(title="Data Requests", rows="1000", cols="7")
                            req_ws.append_row(["Timestamp", "User", "Species", "Year(s)", "Country", "Reason", "Status"])

                        req_ws.append_row([
                            timestamp,
                            st.session_state["user_email"],
                            species,
                            years,
                            country,
                            reason,
                            "Pending"
                        ])

                        st.session_state["datareq_submitted_success"] = True
                        st.session_state["datareq_submitted_reset"] = True
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Failed to submit your request: {e}")

    # ----------------------------
    # 🤝 COLLABORATION
    # ----------------------------
    elif selected_tab == "Collaboration":
        st.markdown("## Collaboration Area")

        # --- Carrega dados de usuário da planilha
        scope = ["https://www.googleapis.com/auth/spreadsheets"]
        credentials = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
        client = gspread.authorize(credentials)
        sheet = client.open_by_key(SHEET_ID)
        users_ws = sheet.worksheet("Users")
        df_users = pd.DataFrame(users_ws.get_all_records())
        df_users.columns = [col.strip().title() for col in df_users.columns]

        # --- Leitura da aba 'Projects'
        try:
            projects_ws = sheet.worksheet("Projects")
            df_projects = pd.DataFrame(projects_ws.get_all_records())
            df_projects.columns = [col.strip().title() for col in df_projects.columns]
        except Exception as e:
            st.error(f"❌ Failed to load 'Projects' sheet: {e}")
            df_projects = pd.DataFrame()

        # --- Identifica o usuário logado
        email = st.session_state.get("user_email")
        user_row = df_users[df_users["E-Mail"] == email]

        if user_row.empty:
            st.error("❌ User not found.")
            st.stop()

        user_role = user_row["Role"].values[0].strip().lower()
        user_projects = user_row["Projects"].values[0].strip()
        user_projects_list = [p.strip() for p in user_projects.split(",")] if user_projects else []

        def is_admin(): return user_role == "admin"
        def is_lead(): return user_role == "lead"
        def is_member(): return user_role == "member"
        def is_standard_user(): return user_role == "user"
        def has_project_access(project): return "all" in user_projects_list or project in user_projects_list

        if not (is_admin() or is_lead() or is_member()):
            st.warning("🔐 You do not have access to the collaboration area.")
            st.stop()

        # --- Subtabs para navegação
        collab_tab = tabs(
            options=["Investigation Dashboard", "Create Investigation", "Update Investigations", "Manage Members"],
            default_value="Investigation Dashboard",
            key="collab_inner_tabs"
        )

        selected_project = None

        # --- DASHBOARD
        if collab_tab == "Investigation Dashboard":
            st.markdown("### Investigations You Collaborate On")

            # Garante que as colunas estão padronizadas
            df_projects.columns = [col.strip().title() for col in df_projects.columns]

            if "Project Id" not in df_projects.columns:
                st.error("⚠️ Column 'Project Id' not found in the Projects sheet. Please check your header formatting.")
                st.stop()

            user_projects = df_projects[df_projects["Collaborators"].str.contains(email, na=False)]

            if user_projects.empty:
                st.info("You are not listed as a collaborator on any investigations.")
            else:
                selected_investigation = st.selectbox("Select an investigation to explore:", user_projects["Project Id"].tolist())

                if selected_investigation not in user_projects["Project Id"].values:
                    st.warning("Selected investigation not found.")
                    st.stop()

                selected_data = user_projects[user_projects["Project Id"] == selected_investigation].iloc[0]

                # --- Leitura de atualizações do feed
                try:
                    updates_ws = sheet.worksheet("Project_Updates")
                    df_updates = pd.DataFrame(updates_ws.get_all_records())
                    df_updates.columns = [col.strip() for col in df_updates.columns]
                except Exception:
                    df_updates = pd.DataFrame()

                if "Project ID" in df_updates.columns:
                    filtered_updates = df_updates[df_updates["Project ID"] == selected_investigation]
                else:
                    filtered_updates = pd.DataFrame()

                if "Timestamp" in filtered_updates.columns:
                    sorted_updates = filtered_updates.sort_values("Timestamp", ascending=False)
                else:
                    sorted_updates = filtered_updates

                # --- Montagem da timeline em HTML seguro
                if sorted_updates.empty:
                    timeline_html = "<p>No updates have been submitted for this investigation yet.</p>"
                else:
                    timeline_items = [
                        f"<p><strong>{row.get('Date', 'Unknown Date')}</strong> — <em>{row.get('Type', 'Unspecified')}</em><br>"
                        f"👤 {row.get('Submitted By', 'Unknown')}<br>"
                        f"{row.get('Description', '')}</p><hr style='margin:8px 0;'>"
                        for _, row in sorted_updates.iterrows()
                    ]
                    timeline_html = "<div>" + "".join(timeline_items) + "</div>"

                # --- Exibição final
                with st.container():

                    # Monta timeline como HTML de linha única com links (se houver)
                    timeline_items = []
                    for _, row in sorted_updates.iterrows():
                        date = row.get("Date", "Unknown Date")
                        type_ = row.get("Type", "Unspecified")
                        author = row.get("Submitted By", "Unknown")
                        desc = row.get("Description", "")
                        links_raw = row.get("Links", "").strip()

                        # Processa os links (se houver)
                        links_html = ""
                        if links_raw:
                            links = [link.strip() for part in links_raw.splitlines() for link in part.split(",") if link.strip()]
                            links_html = " · " + " | ".join(
                                f"<a href='{link}' target='_blank'>🔗 Link</a>" for link in links
                            )

                        # Monta a linha
                        timeline_items.append(
                            f"<p><strong>{date}</strong> — <em>{type_}</em> · 👤 {author} · {desc}{links_html}</p>"
                        )

                    timeline_html = "<div>" + "".join(timeline_items) + "</div>" if timeline_items else "<p>No updates have been submitted for this investigation yet.</p>"

                    st.markdown(
                        f"""
                        <div style="border: 1px solid #cccccc; border-radius: 12px; padding: 16px; background-color: #f9f9f9;">
                            <h4>Summary for: <b>{selected_data.get('Project Name', 'Unknown')}</b></h4>
                            <ul style="list-style-type: none; padding-left: 0;">
                                <li><strong>Lead:</strong> {selected_data.get('Lead', 'N/A')}</li>
                                <li><strong>Cases Involved:</strong> {selected_data.get('Cases Involved', 'N/A')}</li>
                                <li><strong>Species:</strong> {selected_data.get('Target Species', 'N/A')}</li>
                                <li><strong>Countries:</strong> {selected_data.get('Countries Covered', 'N/A')}</li>
                                <li><strong>Status:</strong> {selected_data.get('Project Status', 'N/A')}</li>
                                <li><strong>Summary:</strong> {selected_data.get('Summary', 'No summary provided.')}</li>
                            </ul>
                            <h5 style="margin-top: 24px;">Investigation Timeline</h5>
                            {timeline_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)


        # --- CRIAÇÃO DE PROJETOS
        elif collab_tab == "Create Investigation" and (is_admin() or is_lead()):
            st.markdown("### Create New Investigation")

            field_keys = {
                "project": "create_project_code",
                "members": "create_project_emails"
            }

            with st.form("create_project_form"):
                new_project = st.text_input(
                    "Investigation code (no spaces, e.g., trafick_br)",
                    key=field_keys["project"]
                )
                new_members_raw = st.text_area(
                    "Add collaborators' emails (comma-separated)",
                    placeholder="email1@org.org, email2@org.org",
                    key=field_keys["members"]
                )

                st.markdown("#### Additional Investigation Metadata")
                name = st.text_input("Investigation name")
                species = st.text_input("Target species (comma-separated)")
                countries = st.text_input("Countries covered")
                cases = st.text_input("Cases involved (comma-separated Case #)")
                monitoring = st.selectbox("Monitoring type", ["Passive", "Active", "Mixed"])
                status = st.selectbox("Investigation status", ["Ongoing", "Finalized", "On Hold", "Cancelled"])
                summary = st.text_area("Investigation summary (brief description)")

                submit_new_project = st.form_submit_button("Create Investigation")

                if submit_new_project:
                    if not new_project.strip():
                        st.warning("Please provide an investigation code.")
                    else:
                        new_project = new_project.strip()
                        emails = [e.strip() for e in new_members_raw.split(",") if e.strip()]
                        updated = 0

                        for idx, row in df_users.iterrows():
                            user_email = row["E-Mail"].strip()
                            if user_email in emails:
                                current_projects = row["Projects"].strip()
                                project_list = [p.strip() for p in current_projects.split(",")] if current_projects else []
                                if new_project not in project_list:
                                    project_list.append(new_project)
                                    df_users.at[idx, "Projects"] = ", ".join(sorted(set(project_list)))
                                    updated += 1

                        new_project_entry = {
                            "Project ID": new_project,
                            "Project Name": name,
                            "Lead": email,
                            "Collaborators": ", ".join(emails),
                            "Creation Date": datetime.today().strftime("%Y-%m-%d"),
                            "Cases Involved": cases,
                            "Target Species": species,
                            "Countries Covered": countries,
                            "Monitoring Type": monitoring,
                            "Project Status": status,
                            "Summary": summary,
                            "Last Update": datetime.today().strftime("%Y-%m-%d"),
                            "Public": "FALSE"
                        }

                        try:
                            users_ws.update([df_users.columns.values.tolist()] + df_users.values.tolist())

                            current_projects_data = projects_ws.get_all_values()
                            if current_projects_data:
                                header = current_projects_data[0]
                                new_row = [new_project_entry.get(col, "") for col in header]
                                projects_ws.append_row(new_row)
                            else:
                                header = list(new_project_entry.keys())
                                new_row = list(new_project_entry.values())
                                projects_ws.update([header, new_row])

                            st.success(f"✅ Project '{new_project}' created and assigned to {updated} user(s).")
                            for k in field_keys.values():
                                if k in st.session_state:
                                    del st.session_state[k]
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to update sheet: {e}")

        # --- UPDATE INVESTIGATIONS e MANAGE MEMBERS: Selecionar projeto primeiro
        elif collab_tab in ["Update Investigations", "Manage Members"]:
            user_projects_list = []
            if not df_users.empty and "Projects" in df_users.columns:
                user_row = df_users[df_users["E-Mail"] == email]
                if not user_row.empty:
                    raw_projects = user_row.iloc[0]["Projects"]
                    user_projects_list = [p.strip() for p in raw_projects.split(",") if p.strip()]

            if user_projects_list:
                selected_project = st.selectbox("Select an investigation to manage:", user_projects_list)
                if not has_project_access(selected_project):
                    st.warning("You do not have access to this investigation.")
                    st.stop()
            else:
                st.warning("You do not have access to any investigation.")
                st.stop()

            # --- ABA: UPDATE INVESTIGATIONS
            if collab_tab == "Update Investigations":
                st.markdown(f"### Updates for Investigation: **{selected_project}**")

                try:
                    updates_ws = sheet.worksheet("Project_Updates")
                    df_updates = pd.DataFrame(updates_ws.get_all_records())
                except Exception:
                    updates_ws = sheet.add_worksheet(title="Project_Updates", rows=1000, cols=6)
                    updates_ws.update([["Project ID", "Date", "Submitted By", "Description", "Type", "Timestamp"]])
                    df_updates = pd.DataFrame()

                project_updates = df_updates[df_updates["Project ID"] == selected_project] if not df_updates.empty else pd.DataFrame()

                if not project_updates.empty:
                    st.markdown("#### Project Update Feed")
                    for _, row in project_updates.sort_values("Timestamp", ascending=False).iterrows():
                        st.markdown(
                            f"**{row['Date']}** — *{row['Type']}*  \\\\ 👤 {row['Submitted By']}  \\\\ {row['Description']}"
                        )
                else:
                    st.info("No updates have been submitted for this investigation yet.")

                # Reset form state if needed
                if "clear_update_fields" in st.session_state:
                    st.session_state.update_type_input = ""
                    st.session_state.update_desc_input = ""
                    st.session_state.update_links = ""
                    del st.session_state["clear_update_fields"]

                st.markdown("#### Submit a New Update")
                with st.form("submit_project_update"):
                    update_date = st.date_input("Date of event", value=datetime.today())
                    update_type = st.selectbox(
                        "Type of update",
                        ["", "Movement", "Suspicious Activity", "Legal Decision", "Logistic Operation", "Other"],
                        key="update_type_input"
                    )
                    update_desc = st.text_area("Description of update", key="update_desc_input")
                    update_links = st.text_area(
                        "Link(s) (optional, comma-separated or multiline)",
                        key="update_links"
                    )

                    submit_update = st.form_submit_button("Submit Update")

                    if submit_update:
                        new_entry = {
                            "Project ID": selected_project,
                            "Date": update_date.strftime("%Y-%m-%d"),
                            "Submitted By": email,
                            "Description": update_desc,
                            "Type": update_type,
                            "Links": update_links,
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }

                        try:
                            current_data = updates_ws.get_all_values()
                            if current_data:
                                header = current_data[0]
                                new_row = [new_entry.get(col, "") for col in header]
                                updates_ws.append_row(new_row)
                            else:
                                # Se a aba estiver vazia, cria com as novas colunas
                                header = list(new_entry.keys())
                                new_row = list(new_entry.values())
                                updates_ws.update([header, new_row])

                            st.success("Update submitted successfully.")
                            st.session_state["clear_update_fields"] = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to submit update: {e}")

            # --- ABA: MANAGE MEMBERS
            elif collab_tab == "Manage Members":
                st.markdown("### Add Members to This Investigation")
                with st.form("manage_project_members_add"):
                    new_emails = st.text_area(
                        f"Add user emails to '{selected_project}' (comma-separated)",
                        placeholder="email1@org.org, email2@org.org",
                        key="add_members_input"
                    )
                    submit_add = st.form_submit_button("Add Members")

                    if submit_add:
                        if not new_emails.strip():
                            st.warning("Please enter at least one email.")
                        else:
                            emails = [e.strip() for e in new_emails.split(",") if e.strip()]
                            added = 0
                            for idx, row in df_users.iterrows():
                                user_email = row["E-Mail"].strip()
                                if user_email in emails:
                                    current_projects = row["Projects"].strip()
                                    project_list = [p.strip() for p in current_projects.split(",")] if current_projects else []
                                    if selected_project not in project_list:
                                        project_list.append(selected_project)
                                        df_users.at[idx, "Projects"] = ", ".join(sorted(set(project_list)))
                                        added += 1
                            try:
                                users_ws.update([df_users.columns.values.tolist()] + df_users.values.tolist())
                                st.success(f"{added} user(s) added to '{selected_project}'.")
                                if "add_members_input" in st.session_state:
                                    del st.session_state["add_members_input"]
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to update sheet: {e}")

                st.markdown("### Remove Member from This Project")
                project_members = df_users[df_users["Projects"].str.contains(selected_project, case=False)]
                member_emails = sorted(set(project_members["E-Mail"].str.strip()))

                with st.form("remove_project_member_form"):
                    email_to_remove = st.selectbox("Select member to remove:", member_emails, key="remove_member_input")
                    submit_remove = st.form_submit_button("Remove Member")

                    if submit_remove:
                        for idx, row in df_users.iterrows():
                            if row["E-Mail"].strip() == email_to_remove:
                                projects = [p.strip() for p in row["Projects"].split(",") if p.strip()]
                                if selected_project in projects:
                                    projects.remove(selected_project)
                                    df_users.at[idx, "Projects"] = ", ".join(projects)
                        try:
                            users_ws.update([df_users.columns.values.tolist()] + df_users.values.tolist())
                            st.success(f"User '{email_to_remove}' removed from '{selected_project}'.")
                            if "remove_member_input" in st.session_state:
                                del st.session_state["remove_member_input"]
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to update sheet: {e}")

if uploaded_file is None and st.session_state.get("user"):
    try:
        records = load_all_records()
        df_dashboard = pd.DataFrame(records)

        if not df_dashboard.empty and "N seized specimens" in df_dashboard.columns:
            def expand_multi_species_rows(df):
                expanded_rows = []
                for _, row in df.iterrows():
                    text = str(row.get('N seized specimens', ''))
                    matched_species = set()

                    # 0. Trata casos do tipo "1100 kg (Espécie A + Espécie B)"
                    shared_matches = re.findall(
                        r'(\d+(?:\.\d+)?)\s*kg\s*\(([^)]+)\)', text
                    )
                    for qty_str, species_group in shared_matches:
                        qty = float(qty_str)
                        species_list = [s.strip() for s in re.split(r'\s*\+\s*', species_group)]
                        for species in species_list:
                            if re.match(r"(?i)^bush ?meat$", species):
                                species = "Bushmeat"
                            matched_species.add(species)
                            new_row = row.copy()
                            new_row["Species"] = species
                            new_row["N_seized"] = 0
                            new_row["Estimated weight (kg)"] = qty
                            new_row["Shared weight"] = True
                            new_row["Animal parts seized"] = 0
                            expanded_rows.append(new_row)

                    # 1. Extrai entradas com número + unidade + espécie
                    matches = re.findall(
                        r'(\d+(?:\.\d+)?)\s*(kg|parts?|fangs?|claws?|feathers?|scales?|shells?)?\s*([A-Z][a-z]+(?: [a-z]+)+|[Bb]ush ?[Mm]eat)',
                        text,
                        flags=re.IGNORECASE
                    )
                    for qty, unit, species in matches:
                        qty = float(qty)
                        species = species.strip()
                        if re.match(r"(?i)^bush ?meat$", species):
                            species = "Bushmeat"
                        if species in matched_species:
                            continue
                        matched_species.add(species)

                        new_row = row.copy()
                        new_row["Species"] = species
                        new_row["N_seized"] = 0
                        new_row["Estimated weight (kg)"] = 0
                        new_row["Shared weight"] = False
                        new_row["Animal parts seized"] = 0

                        unit = (unit or "").lower()
                        if unit == "kg":
                            new_row["Estimated weight (kg)"] = qty
                        elif unit in ["part", "parts", "fang", "fangs", "claw", "claws", "feather", "feathers", "scale", "scales", "shell", "shells"]:
                            new_row["Animal parts seized"] = qty
                        else:
                            new_row["N_seized"] = qty

                        expanded_rows.append(new_row)

                    # 2. Garante inclusão de espécies mencionadas mesmo sem número
                    all_species = re.findall(r'\b([A-Z][a-z]+ [a-z]+|[Bb]ush ?[Mm]eat)\b', text)
                    for species in set(all_species):
                        if re.match(r"(?i)^bush ?meat$", species):
                            species = "Bushmeat"
                        if species not in matched_species:
                            new_row = row.copy()
                            new_row["Species"] = species
                            new_row["N_seized"] = 0
                            new_row["Estimated weight (kg)"] = 0
                            new_row["Shared weight"] = False
                            new_row["Animal parts seized"] = 0
                            expanded_rows.append(new_row)

                df_exp = pd.DataFrame(expanded_rows)
                df_exp["Species_clean"] = df_exp["Species"].str.strip()

                def format_species_italics(name):
                    if name.lower().startswith("bushmeat"):
                        return name
                    if re.match(r"^[A-Z][a-z]+ [a-z]+$", name):
                        return f"_{name}_"
                    elif re.match(r"^[A-Z][a-z]+ sp\\.?$", name):
                        return f"_{name[:-1]}_ sp."
                    elif re.match(r"^[A-Z][a-z]+ spp\\.?$", name):
                        return f"_{name[:-1]}_ spp."
                    else:
                        return name

                df_exp["Species_display"] = df_exp["Species_clean"].apply(format_species_italics)
                return df_exp
                
            df_dashboard = expand_multi_species_rows(df_dashboard)
            df_dashboard = df_dashboard[df_dashboard["Species"].notna()]
            df_dashboard["N_seized"] = pd.to_numeric(df_dashboard["N_seized"], errors="coerce").fillna(0)
            df_dashboard["Estimated weight (kg)"] = pd.to_numeric(df_dashboard["Estimated weight (kg)"], errors="coerce").fillna(0)
            df_dashboard["Animal parts seized"] = pd.to_numeric(df_dashboard["Animal parts seized"], errors="coerce").fillna(0)

            dashboard_tab = tabs(
                options=["Summary Dashboard", "Distribution of Cases"],
                default_value="",
                key="dashboard_tabs"
            )

            available_species = sorted(df_dashboard["Species"].unique())

            if dashboard_tab == "Summary Dashboard":
                st.markdown("## Summary Dashboard")

                selected_species_dash = st.selectbox(
                    "Select a species to view:",
                    ["All species"] + available_species,
                    key="species_summary_dashboard"
                )

                if selected_species_dash == "All species":
                    total_species = df_dashboard[df_dashboard["Species"].str.lower() != "bushmeat"]["Species"].nunique()
                    total_cases_all = df_dashboard["Case #"].nunique()
                    total_individuals_all = int(df_dashboard["N_seized"].sum())
                    total_countries_all = df_dashboard["Country of seizure or shipment"].nunique() if "Country of seizure or shipment" in df_dashboard.columns else 0

                    # ✅ Usa colunas diretas da planilha Aurum_data
                    if "Shared weight" in df_dashboard.columns:
                        df_weights = df_dashboard.copy()
                        df_weights["Shared weight"] = df_weights["Shared weight"].fillna(False)

                        total_non_shared = pd.to_numeric(
                            df_weights[df_weights["Shared weight"] == False].get("Estimated weight (kg)", 0),
                            errors="coerce"
                        ).fillna(0).sum()

                        unique_shared_cases = df_weights[df_weights["Shared weight"] == True].drop_duplicates(subset=["Case #"])
                        shared_kg = pd.to_numeric(
                            unique_shared_cases.get("Estimated weight (kg)", 0),
                            errors="coerce"
                        ).fillna(0).sum()

                        total_kg = total_non_shared + shared_kg
                    else:
                        total_kg = pd.to_numeric(
                            df_dashboard.get("Estimated weight (kg)", 0),
                            errors="coerce"
                        ).fillna(0).sum()

                    total_parts = pd.to_numeric(df_dashboard.get("Animal parts seized", 0), errors="coerce").fillna(0).sum()

                    st.markdown("### Global Summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Species seized", total_species)
                    col2.metric("Total cases", total_cases_all)
                    col3.metric("Countries involved", total_countries_all)

                    col4, col5, col6 = st.columns(3)
                    col4.metric("Individuals seized", total_individuals_all)
                    col5.metric("Estimated weight (kg)", f"{total_kg:.1f}")
                    col6.metric("Animal parts seized", int(total_parts))

                else:
                    df_species = df_dashboard[df_dashboard["Species"] == selected_species_dash]
                    if "Year" in df_species.columns and not df_species.empty:
                        try:
                            df_species["Year"] = pd.to_numeric(df_species["Year"], errors="coerce")
                            n_cases = df_species["Case #"].nunique()
                            n_countries = df_species["Country of seizure or shipment"].nunique()
                            if df_species["N_seized"].max() > 0:
                                idx_max = df_species["N_seized"].idxmax()
                                max_row = df_species.loc[idx_max]
                                max_apreensao = f"{max_row['Country of seizure or shipment']} in {int(max_row['Year'])}"
                            else:
                                max_apreensao = "No data"

                            st.markdown("### Key Indicators for Selected Species")
                            col_a, col_b, col_c = st.columns(3)
                            col_a.metric("Cases recorded", n_cases)
                            col_b.metric("Countries with seizures", n_countries)
                            col_c.metric("Largest seizure", max_apreensao)

                            col1, col2 = st.columns(2)
                            with col1:
                                fig_scatter = px.scatter(
                                    df_species,
                                    x="Year",
                                    y="N_seized",
                                    title="Individuals Seized per Case",
                                    labels={"N_seized": "Individuals", "Year": "Year"}
                                )
                                st.plotly_chart(fig_scatter, use_container_width=True)

                            with col2:
                                df_bar = df_species.groupby("Year", as_index=False)["N_seized"].sum()
                                fig_bar = px.bar(
                                    df_bar,
                                    x="Year",
                                    y="N_seized",
                                    title="Total Individuals per Year",
                                    labels={"N_seized": "Total Individuals", "Year": "Year"}
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)

                        except Exception as e:
                            st.warning(f"Could not render plots: {e}")

                    st.markdown("### Species co-occurring in same cases")
                    cases_with_selected = df_dashboard[df_dashboard["Species"] == selected_species_dash]["Case #"].unique()
                    coocurrence_df = df_dashboard[df_dashboard["Case #"].isin(cases_with_selected)]
                    co_species = coocurrence_df[coocurrence_df["Species"] != selected_species_dash]["Species"].unique()

                    def format_species_italic(name):
                        if name.lower().startswith("bushmeat"):
                            return name
                        if re.match(r"^[A-Z][a-z]+ [a-z]+$", name):
                            return f"_{name}_"
                        elif re.match(r"^[A-Z][a-z]+ sp\\.?$", name):
                            return f"_{name[:-1]}_ sp."
                        elif re.match(r"^[A-Z][a-z]+ spp\\.?$", name):
                            return f"_{name[:-1]}_ spp."
                        else:
                            return name

                    if len(co_species) > 0:
                        formatted_species = [format_species_italic(s) for s in sorted(co_species)]
                        st.markdown(", ".join(formatted_species))
                    else:
                        st.info("No other species recorded with the selected species.")

            elif dashboard_tab == "Distribution of Cases":
                st.markdown("## Distribution of Recorded Cases")

                selected_species_dash = st.selectbox(
                    "Select a species to view:",
                    ["All species"] + available_species,
                    key="species_distribution_dashboard"
                )

                col1, col2 = st.columns([1, 1.4])

                with col1:
                    st.markdown("#### Species Information")

                    if selected_species_dash == "All species":
                        st.info("Select a species to view notes.")
                    else:
                        import urllib.parse
                        base_url = "https://raw.githubusercontent.com/acarvalho-wcs/aurum/main/.images/"
                        species_filename = urllib.parse.quote(selected_species_dash.replace(" ", "_") + ".jpg")
                        image_url = base_url + species_filename

                        st.image(image_url, use_container_width=True)
                        st.markdown(f"<div style='text-align: center'><em>{selected_species_dash}</em></div>", unsafe_allow_html=True)

                with col2:
                    st.markdown("#### Heatmap of Recorded Cases by Location")

                    if "Latitude" not in df_dashboard.columns or "Longitude" not in df_dashboard.columns:
                        st.warning("This analysis requires 'Latitude' and 'Longitude' columns in the dataset.")
                    else:
                        df_geo = df_dashboard.dropna(subset=["Latitude", "Longitude"]).copy()

                        df_geo["Latitude"] = (
                            df_geo["Latitude"]
                            .astype(str)
                            .str.replace(",", ".", regex=False)
                            .astype(float)
                            .round(3)
                        )
                        df_geo["Longitude"] = (
                            df_geo["Longitude"]
                            .astype(str)
                            .str.replace(",", ".", regex=False)
                            .astype(float)
                            .round(3)
                        )

                        if selected_species_dash != "All species":
                            df_geo = df_geo[df_geo["Species"] == selected_species_dash]

                        if df_geo.empty:
                            st.info("No valid geolocation data found for the selected species.")
                        else:
                            import folium
                            from folium.plugins import HeatMap
                            import geopandas as gpd
                            import re
                            from streamlit_shadcn_ui import tabs

                            def extract_specimens_only(cell):
                                if pd.isna(cell):
                                    return 0
                                clean = re.sub(r"\b\d+(\.\d+)?\s*kg\b", "", str(cell), flags=re.IGNORECASE)
                                numbers = re.findall(r"\b\d+\b", clean)
                                return sum(int(n) for n in numbers)

                            df_geo_unique = df_geo.drop_duplicates(subset=["Case #", "Latitude", "Longitude"])
                            gdf = gpd.GeoDataFrame(
                                df_geo_unique,
                                geometry=gpd.points_from_xy(df_geo_unique["Longitude"], df_geo_unique["Latitude"]),
                                crs="EPSG:4326"
                            )

                            bounds = gdf.total_bounds
                            center_lat = (bounds[1] + bounds[3]) / 2
                            center_lon = (bounds[0] + bounds[2]) / 2

                            st.markdown("**Select weighting method for heatmap:**")

                            METHOD_CASE = "Per case"
                            METHOD_SPECIMENS = "By number of specimens"
                            METHOD_WEIGHT = "By weight (kg)"
                            METHOD_PARTS = "By animal parts"

                            tab_labels = [
                                METHOD_CASE,
                                METHOD_SPECIMENS,
                                METHOD_WEIGHT,
                                METHOD_PARTS
                            ]

                            if "method_tab" not in st.session_state:
                                st.session_state["method_tab"] = METHOD_CASE

                            method = tabs(
                                options=tab_labels,
                                key="method_tab"
                            )

                            st.markdown(f"*Current method: **{method}***")

                            gdf["weight"] = 1

                            if method == METHOD_SPECIMENS and "N seized specimens" in gdf.columns:
                                gdf["weight"] = gdf["N seized specimens"].apply(extract_specimens_only)
                                gdf = gdf[gdf["weight"] > 0]

                            elif method == METHOD_WEIGHT and "Estimated weight (kg)" in gdf.columns:
                                gdf["weight"] = pd.to_numeric(gdf["Estimated weight (kg)"], errors="coerce")
                                gdf = gdf[gdf["weight"] > 0]

                            elif method == METHOD_PARTS and "Animal parts seized" in gdf.columns:
                                gdf["weight"] = pd.to_numeric(gdf["Animal parts seized"], errors="coerce").fillna(0)
                                gdf = gdf[gdf["weight"] > 0]

                            if gdf.empty:
                                st.info("No valid data found for this method.")
                            else:
                                max_weight = gdf["weight"].max()
                                min_weight = gdf["weight"].min()
                                st.caption(f"Max value for selected method: **{max_weight:.1f}** – Min: **{min_weight:.1f}**")

                                if max_weight == min_weight:
                                    normalized_weights = [1.0] * len(gdf)
                                else:
                                    normalized_weights = [
                                        0.1 + 0.9 * (w - min_weight) / (max_weight - min_weight)
                                        for w in gdf["weight"]
                                    ]

                                radius_val = st.slider("HeatMap radius (px)", 5, 50, 25, key="heatmap_radius")

                                m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
                                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

                                heat_data = [
                                    [row.Latitude, row.Longitude, norm]
                                    for row, norm in zip(gdf.itertuples(), normalized_weights)
                                ]
                                HeatMap(data=heat_data, radius=radius_val).add_to(m)

                                legend_html = '''
                                    <div style="
                                        position: fixed;
                                        bottom: 40px;
                                        right: 20px;
                                        z-index: 9999;
                                        background-color: white;
                                        padding: 10px;
                                        border:2px solid gray;
                                        border-radius:5px;
                                        font-size:14px;
                                        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
                                        <b>HeatMap Intensity</b>
                                        <div style="height: 10px; width: 120px;
                                            background: linear-gradient(to right, blue, cyan, lime, yellow, orange, red);
                                            margin: 5px 0;"></div>
                                        <div style="display: flex; justify-content: space-between;">
                                            <span>Low</span>
                                            <span>Medium</span>
                                            <span>High</span>
                                        </div>
                                        <div style="margin-top:6px; font-size:10px; color:gray;">Generated with Aurum</div>
                                    </div>
                                '''
                                m.get_root().html.add_child(folium.Element(legend_html))

                                html_str = m.get_root().render()
                                st.components.v1.html(html_str, height=400)

                                # Observação sobre pesos compartilhados entre espécies
                                if (
                                    selected_species_dash != "All species" and
                                    "N seized specimens" in df_geo.columns
                                ):
                                    grouped_df = df_geo[
                                        df_geo["N seized specimens"].astype(str).str.contains(r"\(.*\+.*\)")
                                    ]
                                    co_species = []
                                    for text in grouped_df["N seized specimens"]:
                                        match = re.search(r"\(([^)]+)\)", text)
                                        if match:
                                            species_group = [s.strip() for s in match.group(1).split("+")]
                                            if selected_species_dash in species_group:
                                                others = [s for s in species_group if s != selected_species_dash]
                                                co_species.extend(others)

                                    co_species = sorted(set(co_species))
                                    if co_species:
                                        formatted = " and ".join(f"`{s}`" for s in co_species)
                                        st.caption(
                                            f"**Some weights were reported as grouped values with {formatted} "
                                            f"(e.g., '1100 kg (Species A + Species B)').**"
                                        )

                                from io import BytesIO

                                map_html = m.get_root().render()
                                safe_species = selected_species_dash.replace(" ", "_").replace("/", "_")
                                filename = f"aurum_heatmap_{safe_species}.html"
                                map_bytes = BytesIO(map_html.encode("utf-8"))

                                col_btn1, col_btn2, col_btn3 = st.columns([5, 2, 2])
                                with col_btn3:
                                    st.download_button(
                                        label="Download HTML",
                                        data=map_bytes,
                                        file_name=filename,
                                        mime="text/html",
                                        use_container_width=True
                                    )

    except Exception as e:
        st.error(f"❌ Failed to load dashboard summary: {e}")
                
# --- SUGGESTIONS AND COMMENTS (SIDEBAR) ---
if "show_sidebar_feedback" not in st.session_state:
    st.session_state["show_sidebar_feedback"] = False

# --- BOTÃO FIXO NA SIDEBAR ---
feedback_toggle = st.sidebar.button("💬 Suggestions and Comments")

# Alterna visibilidade do formulário
if feedback_toggle:
    st.session_state["show_sidebar_feedback"] = not st.session_state["show_sidebar_feedback"]

# Exibe o formulário se o botão estiver ativado
if st.session_state["show_sidebar_feedback"]:
    with st.sidebar.form("suggestion_form"):
        st.markdown("### 💬 Feedback Form")
        name = st.text_input("Name", key="suggestion_name")
        email = st.text_input("E-mail", key="suggestion_email")
        institution = st.text_input("Institution", key="suggestion_institution")
        message = st.text_area("Suggestions or comments", key="suggestion_message")
        submitted = st.form_submit_button("Submit")

        if submitted:
            if not name or not email or not institution or not message:
                st.warning("All fields are required.")
            else:
                timestamp = datetime.now(brt).strftime("%Y-%m-%d %H:%M:%S (BRT)")

                try:
                    scope = ["https://www.googleapis.com/auth/spreadsheets"]
                    credentials = Credentials.from_service_account_info(
                        st.secrets["gcp_service_account"], scopes=scope)
                    client = gspread.authorize(credentials)
                    sheet = client.open_by_key(SHEET_ID)

                    try:
                        suggestion_ws = sheet.worksheet("Suggestions")
                    except gspread.exceptions.WorksheetNotFound:
                        suggestion_ws = sheet.add_worksheet(title="Suggestions", rows="1000", cols="5")
                        suggestion_ws.append_row(["Timestamp", "Name", "Email", "Institution", "Message"])

                    suggestion_ws.append_row([
                        timestamp,
                        name,
                        email,
                        institution,
                        message.strip()
                    ])

                    st.success("✅ Thank you for your feedback!")

                    st.session_state["show_sidebar_feedback"] = False
                    st.rerun()  # fecha o formulário
                except Exception as e:
                    st.error(f"❌ Failed to submit feedback: {e}")

def display_suggestions_section(SHEET_ID):
    """Exibe sugestões enviadas pelo formulário (apenas admins)."""
    if not st.session_state.get("is_admin"):
        return

    st.markdown("## 💬 User Suggestions and Comments")

    try:
        scope = ["https://www.googleapis.com/auth/spreadsheets"]
        credentials = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
        client = gspread.authorize(credentials)
        sheet = client.open_by_key(SHEET_ID)

        suggestions_ws = sheet.worksheet("Suggestions")
        df = pd.DataFrame(suggestions_ws.get_all_records())
        df.columns = [col.strip() for col in df.columns]

        if df.empty:
            st.info("No feedback has been submitted yet.")
        else:
            st.dataframe(df.sort_values("Timestamp", ascending=False))

    except gspread.exceptions.WorksheetNotFound:
        st.warning("📭 The 'Suggestions' sheet was not found.")
    except Exception as e:
        st.error(f"❌ Failed to load suggestions: {e}")

# CHAMADA DA VISUALIZAÇÃO (APENAS ADMIN)
display_suggestions_section(SHEET_ID)

st.sidebar.markdown("---")    
st.sidebar.markdown("**How to cite:** Carvalho, A. F. Aurum: A Platform for Criminal Intelligence in Wildlife Trafficking. Wildlife Conservation Society, 2025.")
