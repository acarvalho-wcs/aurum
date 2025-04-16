import streamlit as st
import pandas as pd
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
import pygsheets
from google.oauth2.service_account import Credentials

# Page setup
st.set_page_config(page_title="Aurum Dashboard", layout="wide")
st.title("Aurum - Wildlife Trafficking Analytics")
st.markdown("**Select an analysis from the sidebar to begin.**")

# Google Sheets setup
sheet_id = "1HVYbot3Z9OBccBw7jKNw5acodwiQpfXgavDTIptSKic"
try:
    credentials = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    gc = pygsheets.authorize(custom_credentials=credentials)
    spreadsheet = gc.open_by_key(sheet_id)
    worksheet = spreadsheet.worksheet_by_title("Sheet1")
    gateway_df = worksheet.get_as_df()
except Exception as e:
    st.error("Failed to connect to Google Sheets.")
    st.stop()

# User authentication
st.sidebar.markdown("## 🔐 Aurum Gateway")
user = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login = st.sidebar.button("Login")

if login:
    if user and password:
        st.session_state["user"] = user
        st.session_state["is_admin"] = (user == "acarvalho" and password == "admin")
        st.success(f"Logged in as {user}")
    else:
        st.warning("Please provide both username and password.")

# --- Data source selection (manual upload vs Google Sheets) ---
source_option = None
if "user" in st.session_state:
    st.sidebar.markdown("## 📊 Data Source")
    source_option = st.sidebar.radio("Select data source:", ["Manual Upload (.xlsx)", "Aurum Gateway (Google Sheets)"])

    if source_option == "Aurum Gateway (Google Sheets)":
        st.sidebar.success("Using Aurum Gateway data.")
        df = pd.DataFrame(worksheet.get_all_records())

        if not df.empty:
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

            # Country scoring (optional)
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
                    st.sidebar.markdown("✅ Offender_value column added.")

            # Logistic Convergence and Stage
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

            # --- Species selection and analysis options ---
            st.sidebar.markdown("---")
            st.sidebar.markdown("## 🐾 Select Species")
            species_options = sorted(df['Species'].dropna().unique())
            selected_species = st.sidebar.multiselect("Choose one or more species:", species_options)

            df_selected = df[df['Species'].isin(selected_species)] if selected_species else None

            if df_selected is not None and not df_selected.empty:
                st.sidebar.markdown("## 🔍 Choose Analyses")
                do_viz = st.sidebar.checkbox("Data Visualization")
                do_trend = st.sidebar.checkbox("Trend Analysis")
                do_cooc = st.sidebar.checkbox("Species Co-occurrence")
                do_anomaly = st.sidebar.checkbox("Anomaly Detection")
                do_network = st.sidebar.checkbox("Network Analysis")

                st.markdown("## Filtered Data Preview")
                st.dataframe(df_selected.head())

                # You can now reuse the same analysis functions from the manual upload section here
                # by wrapping them inside `if do_viz:`, `if do_trend:`, etc.

            else:
                st.warning("Please select one or more species to proceed.")

# File upload
uploaded_file = None
if not use_gateway_data:
    st.sidebar.markdown("## 📂 Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload your Excel file (.xlsx)", type=["xlsx"])
    st.sidebar.markdown("**Download Template**")
    with open("Aurum_template.xlsx", "rb") as f:
        st.sidebar.download_button(
            label="Download Template",
            data=f,
            file_name="aurum_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Load data
if use_gateway_data:
    df = gateway_df.copy()
else:
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            st.stop()
    else:
        st.stop()

# Preprocess
try:
    df.columns = df.columns.str.strip().str.replace('\xa0', '', regex=True)
    if 'Year' in df.columns:
        df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})').astype(float)

    def expand_multi_species_rows(df):
        expanded = []
        for _, row in df.iterrows():
            matches = re.findall(r'(\d+)\s*([A-Z]{2,}\d{0,3})', str(row.get('N seized specimens', '')))
            if matches:
                for qty, species in matches:
                    new_row = row.copy()
                    new_row['N_seized'] = float(qty)
                    new_row['Species'] = species
                    expanded.append(new_row)
            else:
                expanded.append(row)
        return pd.DataFrame(expanded)

    df = expand_multi_species_rows(df).reset_index(drop=True)

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

except Exception as e:
    st.error(f"❌ Error preprocessing data: {e}")
    st.stop()

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
