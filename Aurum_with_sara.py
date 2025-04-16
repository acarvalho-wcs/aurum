import streamlit as st
import pandas as pd
from datetime import datetime
import re
import unicodedata
import os
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
from google.oauth2.service_account import Credentials
import pygsheets

# Configuração da página
st.set_page_config(page_title="Aurum Dashboard", layout="wide")
st.title("Aurum - Wildlife Trafficking Analytics")
st.markdown("**Select an analysis from the sidebar to begin.**")

# Função para normalização
def normalize_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Função para expandir espécies
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

# --- AUTENTICAÇÃO ---
st.sidebar.markdown("## 🔐 Aurum Gateway")
user = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login = st.sidebar.button("Login")

if login:
    if user and password:
        st.session_state["user"] = user
        st.session_state["is_admin"] = user == "acarvalho" and password == "admin"
        st.success(f"Logged in as {user}")
    else:
        st.warning("Please provide both username and password.")

# Seleção de fonte de dados
use_gateway_data = False
if "user" in st.session_state:
    source_choice = st.sidebar.radio("Select data source:", ["Manual Upload (.xlsx)", "Aurum Gateway (Google Sheets)"])
    use_gateway_data = source_choice == "Aurum Gateway (Google Sheets)"

df = None
df_selected = None

# --- DADOS DO GATEWAY ---
if use_gateway_data:
    try:
        SHEET_ID = "1HVYbot3Z9OBccBw7jKNw5acodwiQpfXgavDTIptSKic"
        credentials = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=["https://www.googleapis.com/auth/spreadsheets"])
        gc = pygsheets.authorize(custom_credentials=credentials)
        sheet = gc.open_by_key(SHEET_ID)
        worksheet = sheet.worksheet_by_title("Sheet1")
        df = worksheet.get_as_df()
        st.success("✅ Using Aurum Gateway data.")
    except Exception as e:
        st.error("Failed to load data from Google Sheets.")
        st.exception(e)
else:
    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded.")

# --- PRÉ-PROCESSAMENTO ---
if df is not None:
    df.columns = df.columns.str.strip().str.replace(' ', '', regex=True)
    if 'Year' in df.columns:
        df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})').astype(float)
    df = expand_multi_species_rows(df).reset_index(drop=True)

    # Enriquecimento com valores por país
    if os.path.exists("country_offenders_values.csv"):
        country_score = pd.read_csv("country_offenders_values.csv", encoding="ISO-8859-1")
        country_map = dict(zip(country_score["Country"].str.strip(), country_score["Value"]))
        def score_countries(val):
            if not isinstance(val, str): return 0
            countries = [c.strip() for c in val.split("+")]
            return sum(country_map.get(c, 0) for c in countries)
        df["Offender_value"] = df["Country of offenders"].apply(lambda x: score_countries(x))

    st.sidebar.markdown("## 🐾 Select Species")
    species_options = sorted(df['Species'].dropna().unique())
    selected_species = st.sidebar.multiselect("Choose one or more species:", species_options)

    if selected_species:
        df_selected = df[df["Species"].isin(selected_species)]

        # Opções de análise
        st.sidebar.markdown("## 🔍 Choose Analyses")
        analyze_viz = st.sidebar.checkbox("Data Visualization")
        analyze_trend = st.sidebar.checkbox("Trend Analysis")
        analyze_cooc = st.sidebar.checkbox("Species Co-occurrence")
        analyze_anomaly = st.sidebar.checkbox("Anomaly Detection")
        analyze_network = st.sidebar.checkbox("Network Analysis")

        st.markdown("## Filtered Data Preview")
        st.dataframe(df_selected.head())

# Show logo
place_logo_bottom_right("wcs.jpg")

# About section
st.sidebar.markdown("---")
show_about = st.sidebar.button("**About Aurum**")
if show_about:
    st.markdown("## About Aurum")
    st.markdown("""
    **Aurum** is a modular and interactive toolkit designed to support the detection and analysis of **wildlife trafficking** and organized environmental crime. Developed by the Wildlife Conservation Society (WCS) – Brazil, it empowers analysts, researchers, and enforcement professionals with data-driven insights through a user-friendly interface.

    The platform enables the upload and processing of case-level data and offers a suite of analytical tools, including:

    - **Trend Analysis**: Explore temporal patterns using segmented regression (TCS), expanding mean, and CUSUM to detect shifts in trafficking intensity over time.
    - **Species Co-occurrence**: Identify statistically significant co-trafficking relationships between species using chi-square tests and network-based representations.
    - **Anomaly Detection**: Apply multiple methods (Isolation Forest, LOF, DBSCAN, Mahalanobis distance, Z-Score) to identify outlier cases based on numerical features.
    - **Criminal Network Analysis**: Visualize co-occurrence networks to reveal potential connections and logistical consolidation among species and locations.
    - **Interactive Visualization**: Generate customized plots and dashboards based on uploaded data and selected variables.
    """)

# Export options
st.sidebar.markdown("---")
st.sidebar.markdown("## Export Options")
export_xlsx = st.sidebar.button("Export Cleaned data.xlsx")
export_html = st.sidebar.button("Export Analysis Report (.html)")

if export_xlsx and df_selected is not None:
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
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_sections = [
        f"<h1>Aurum Wildlife Trafficking Report</h1>",
        f"<p><strong>Generated:</strong> {now}</p>",
        f"<p><strong>Selected Species:</strong> {', '.join(selected_species)}</p>",
        "<h2>Data Sample</h2>",
        df_selected.head(10).to_html(index=False)
    ]

    if show_trend:
        html_sections.extend([
            "<h2>Trend Analysis</h2>",
            f"<p><strong>TCS:</strong> {tcs:.2f}</p>"
        ])
        trend_buf = BytesIO()
        fig.savefig(trend_buf, format="png", bbox_inches="tight")
        trend_buf.seek(0)
        trend_base64 = base64.b64encode(trend_buf.read()).decode("utf-8")
        html_sections.append(f'<img src="data:image/png;base64,{trend_base64}" width="700">')

    if show_cooc and co_results:
        html_sections.append("<h2>Species Co-occurrence</h2>")
        for sp_a, sp_b, chi2, p, table in co_results:
            html_sections.append(f"<h4>{sp_a} × {sp_b}</h4>")
            html_sections.append(table.to_html())
            html_sections.append(f"<p>Chi² = {chi2:.2f} | p = {p:.4f}</p>")

    if show_anomaly and 'vote_df' in locals():
        html_sections.extend([
            "<h2>Anomaly Detection</h2>",
            f"<p><strong>Consensus Outlier Ratio:</strong> {consensus_ratio:.2%}</p>",
            "<h4>Top Anomalies</h4>",
            top_outliers.to_html(index=False)
        ])

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

# Citation
st.sidebar.markdown("How to cite: Carvalho, A. F. Detecting Organized Wildlife Crime with *Aurum*: A Toolkit for Wildlife Trafficking Analysis. Wildlife Conservation Society, 2025.")
