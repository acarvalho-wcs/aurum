import streamlit as st
import pandas as pd
import re
import unicodedata

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

        # Limpeza de colunas
        df.columns = df.columns.str.strip().str.replace('\xa0', '', regex=True)

        # Conversão de colunas e extração do ano
        if 'Year' in df.columns:
            df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})').astype(float)

        # --- Expansão de múltiplas espécies em uma única linha ---
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

        df = expand_multi_species_rows(df)
        df = df.reset_index(drop=True)

        # --- Feature: Logistic Convergence ---
        if 'Case #' in df.columns and 'Species' in df.columns:
            species_per_case = df.groupby('Case #')['Species'].nunique()
            df['Logistic Convergence'] = df['Case #'].map(
                lambda x: "Yes" if species_per_case.get(x, 0) > 1 else "No"
            )

        # --- Função para normalizar texto ---
        def normalize_text(text):
            if not isinstance(text, str):
                text = str(text)
            text = text.strip().lower()
            text = unicodedata.normalize("NFKD", text)
            text = re.sub(r'\s+', ' ', text)
            return text

        # --- Inferred Stage ---
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

        # --- Seletor de espécie ---
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 🧬 Select Species")
        species_options = sorted(df['Species'].dropna().unique())
        selected_species = st.sidebar.multiselect("Select one or more species:", species_options)

        if selected_species:
            df_selected = df[df['Species'].isin(selected_species)]

            # Opção para visualização de gráficos
            show_viz = st.sidebar.checkbox("📊 Show Data Visualization", value=False)
            if show_viz:
                st.sidebar.markdown("---")
                st.sidebar.markdown("## 📊 Data Visualization")

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
        else:
            st.warning("⚠️ Please select at least one species to explore the data.")

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
if df_selected is not None:
    st.markdown(f"### You selected: `{selected_analysis}`")
    st.info("The analysis module will appear here when implemented.")
else:
    st.warning("⚠️ Please upload a valid dataset and select species to continue.")
