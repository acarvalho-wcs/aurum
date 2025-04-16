import streamlit as st
import pandas as pd

# Configuração da página
st.set_page_config(page_title="Aurum Dashboard", layout="wide")

# Título e logotipo
st.title("🐾 Aurum - Wildlife Crime Analysis Dashboard")
st.markdown("Select an analysis from the sidebar to begin.")

# Upload do arquivo
st.sidebar.markdown("## 📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file (.xlsx):", type=["xlsx"])

df = None
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.write("### Preview of uploaded data:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

# Painel lateral para seleção de análise
st.sidebar.markdown("---")
st.sidebar.title("🔍 Analysis Menu")
selected_analysis = st.sidebar.radio(
    "Choose an analysis to explore:",
    [
        "Trend Analysis",
        "Species Co-occurrence",
        "Anomaly Detection",
        "Network Analysis",
        "Organized Crime Score (OCS)",
        "SARA Model"
    ]
)

# Espaço reservado para renderizar a análise selecionada
if df is not None:
    st.markdown(f"### You selected: `{selected_analysis}`")
    st.info("The analysis module will appear here when implemented.")
else:
    st.warning("⚠️ Please upload a valid dataset to continue.")
