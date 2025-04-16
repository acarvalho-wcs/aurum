import streamlit as st

# Configuração da página
st.set_page_config(page_title="Aurum Dashboard", layout="wide")

# Título e logotipo
st.title("Aurum - Wildlife Crime Analysis Dashboard")
st.markdown("Select an analysis from the sidebar to begin.")

# Painel lateral para seleção de análise
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
st.markdown(f"### You selected: `{selected_analysis}`")
st.info("The analysis module will appear here when implemented.")
