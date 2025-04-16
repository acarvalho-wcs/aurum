
import streamlit as st
import pandas as pd
import pygsheets
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

# Configuração da página
st.set_page_config(page_title="Aurum Dashboard", layout="wide")

st.title("Aurum - Wildlife Trafficking Analytics")
st.markdown("**Select an analysis from the sidebar to begin.**")

# --- GOOGLE SHEETS ---
scope = ["https://www.googleapis.com/auth/spreadsheets"]
credentials = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
gc = pygsheets.authorize(custom_credentials=credentials)

# Abre a planilha e a aba correta
sheet_id = "1HVYbot3Z9OBccBw7jKNw5acodwiQpfXgavDTIptSKic"
spreadsheet = gc.open_by_key(sheet_id)
worksheet = spreadsheet.worksheet_by_title("Sheet1")

# --- AUTENTICAÇÃO ---
st.sidebar.markdown("## 🔐 Aurum Gateway")
user = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login = st.sidebar.button("Login")

if login:
    if user and password:
        if user == "acarvalho" and password == "admin":
            st.session_state["user"] = user
            st.session_state["is_admin"] = True
            st.success("Logged in as admin.")
        else:
            st.session_state["user"] = user
            st.session_state["is_admin"] = False
            st.success(f"Logged in as {user}")
    else:
        st.warning("Please provide both username and password.")

# --- SELEÇÃO DA FONTE DE DADOS ---
df = None
uploaded_file = None

if "user" in st.session_state:
    st.sidebar.markdown("## 📊 Data Source")
    data_source = st.sidebar.radio("Select data source:", ["Manual Upload (.xlsx)", "Aurum Gateway (Google Sheets)"], index=0)

    if data_source == "Aurum Gateway (Google Sheets)":
        df = worksheet.get_as_df()
        st.sidebar.success("Using Aurum Gateway data.")
    else:
        uploaded_file = st.sidebar.file_uploader("**Upload your Excel file (.xlsx).**", type=["xlsx"])
else:
    uploaded_file = st.sidebar.file_uploader("**Upload your Excel file (.xlsx).**", type=["xlsx"])

# Salva o template
st.sidebar.markdown("**Download Template**")
with open("Aurum_template.xlsx", "rb") as f:
    st.sidebar.download_button(
        label="Download a data template for wildlife trafficking analysis in Aurum",
        data=f,
        file_name="aurum_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# PROCESSAMENTO DO ARQUIVO
if uploaded_file is not None and (not "data_source" in locals() or data_source == "Manual Upload (.xlsx)"):
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded successfully.")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

# Mostra preview se login e dados carregados
if df is not None and "user" in st.session_state:
    st.markdown("### Data Preview")
    st.dataframe(df.head())
