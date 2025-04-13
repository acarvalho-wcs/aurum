co_score = 0.0  # Prevent NameError on co_score usage
import pandas as pd
import numpy as np
import re
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2_contingency
import networkx as nx
import statsmodels.api as sm

# --- NEW COMPONENT: Trend Component using TCS ---
def trend_component(df, year_col='Year', count_col='N_seized', breakpoint=2015, min_obs=5):
    df_pre = df[df[year_col] <= breakpoint]
    df_post = df[df[year_col] > breakpoint]

    if len(df_pre) < min_obs or len(df_post) < min_obs:
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

    if tcs > 1:
        log = f"TCS = {tcs:.2f} → Strong recent upward trend (potential coordination)"
    elif tcs < 0.5:
        log = f"TCS = {tcs:.2f} → No relevant recent trend"
    else:
        log = f"TCS = {tcs:.2f} → Moderate recent trend"

    return tcs, log

# Load and clean data (handles multiple species)
def load_and_clean_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.replace('\xa0', '', regex=True)
    df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})').astype(float)

    def expand_multi_species_rows(df):
        expanded_rows = []
        for _, row in df.iterrows():
            matches = re.findall(r'(\d+)\s*([A-Z]{2,})', str(row['N seized specimens']))
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
    return df

# Compute crime score

def org_crime_score(df, binary_features, species_col='Species', year_col='Year',
                    count_col='N_seized', country_col='Country of offenders',
                    location_col='Location of seizure or shipment', weights=None):

    default_weights = {'trend': 0.25, 'chi2': 0.15, 'anomaly': 0.20, 'network': 0.30}
    if weights:
        default_weights.update({k: weights.get(k, v) for k, v in default_weights.items()})

    score = 0
    log = {}

    # TREND COMPONENT (replaces old R² logic)
    selected_species = []
    breakpoint = st.session_state.get('trend_breakpoint', 2015)
    tcs, tcs_log = trend_component(df, year_col=year_col, count_col=count_col, breakpoint=breakpoint)
    if tcs > 1:
        score += default_weights['trend']
        log['trend'] = f"+{default_weights['trend']:.2f} ({tcs_log})"
    elif tcs < 0.5:
        score -= default_weights['trend']
        log['trend'] = f"-{default_weights['trend']:.2f} ({tcs_log})"
    else:
        log['trend'] = f"0 ({tcs_log})"

    
    # Chi-squared
    if 'Species' in df.columns:
        species_list = df['Species'].dropna().unique().tolist()
        co_results = general_species_cooccurrence(df, species_list)
        co_score, co_log = compute_cooccurrence_score(co_results, weight=default_weights['chi2'])
        score += co_score
        log['chi2'] = f"{co_score:+.2f} ({co_log})"

    
# Anomaly detection (interactive configuration)
st.markdown("## 🚨 Anomaly Detection")

# Step 1: Feature selection
available_features = df.select_dtypes(include=[np.number]).columns.tolist()
binary_features = st.multiselect("🔧 Select features to include in anomaly detection:", available_features)

# Step 2: Model selection
st.markdown("### 🧪 Choose which models to apply:")
use_iforest = st.checkbox("Isolation Forest", value=True)
use_lof = st.checkbox("Local Outlier Factor", value=True)
use_dbscan = st.checkbox("DBSCAN", value=True)
use_zscore = st.checkbox("Z-score", value=True)
use_mahalanobis = st.checkbox("Mahalanobis Distance", value=True)

# Step 3: Run button
if st.button("▶️ Run Anomaly Detection"):
    if all(f in df.columns for f in binary_features):
        X = StandardScaler().fit_transform(df[binary_features])

        votes = []
        methods_used = []

        if use_iforest:
            iforest = IsolationForest(random_state=42).fit_predict(X)
            votes.append(iforest)
            methods_used.append("Isolation Forest")

        if use_lof:
            lof = LocalOutlierFactor().fit_predict(X)
            votes.append(lof)
            methods_used.append("Local Outlier Factor")

        if use_dbscan:
            dbscan = DBSCAN(eps=1.2, min_samples=2).fit_predict(X)
            votes.append(dbscan)
            methods_used.append("DBSCAN")

        if use_zscore:
            z_scores = np.abs(X)
            z_outliers = np.any(z_scores > 3, axis=1).astype(int)
            z_outliers = np.where(z_outliers == 1, -1, 1)
            votes.append(z_outliers)
            methods_used.append("Z-score")

        if use_mahalanobis:
            try:
                cov = np.cov(X, rowvar=False)
                inv_cov = np.linalg.inv(cov)
                mean = np.mean(X, axis=0)
                diff = X - mean
                md = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
                threshold_md = np.percentile(md, 97.5)
                mahalanobis = np.where(md > threshold_md, -1, 1)
            except np.linalg.LinAlgError:
                mahalanobis = np.ones(len(X))
            votes.append(mahalanobis)
            methods_used.append("Mahalanobis")

        if votes:
            outlier_votes = sum(pd.Series(votes).apply(lambda x: (np.array(x) == -1).sum()))
            ratio = outlier_votes / (len(df) * len(votes))

            st.markdown("### 🔍 Detected Anomalous Cases")
            anomaly_votes = pd.DataFrame({"Case #": df["Case #"]})

            for name, vote in zip(methods_used, votes):
                anomaly_votes[name] = (np.array(vote) == -1).astype(int)

            anomaly_votes["Total Votes"] = anomaly_votes.drop(columns=["Case #"]).sum(axis=1)
            flagged_cases = anomaly_votes[anomaly_votes["Total Votes"] >= 2]

            if not flagged_cases.empty:
                st.dataframe(flagged_cases.sort_values(by="Total Votes", ascending=False))
            else:
                st.info("No significant anomalies detected across selected methods.")
        else:
            st.warning("Please select at least one anomaly detection method.")
    else:
        st.error("Selected features are not available in the dataset.")

# Network structure
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row['Case #'])

    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i >= j:
                continue
            shared = sum([
                row1[year_col] == row2[year_col],
                row1[species_col] == row2[species_col],
                row1[country_col] == row2[country_col]
            ])
            if shared >= 2:
                G.add_edge(row1['Case #'], row2['Case #'])

    density = nx.density(G)
    components = nx.number_connected_components(G)
    if density > 0.2 and components < len(df) / 3:
        score += default_weights['network']
        log['network'] = f'+{default_weights["network"]:.2f} (density = {density:.2f}, {components} comps)'
    else:
        log['network'] = f'0 (density = {density:.2f}, {components} comps)'

    
    # Co-occurrence emergence flag (e.g., sp_a + sp_b since 2023)
    log['cooccurrence_flag'] = detect_emerging_cooccurrence(df, 'sp_a', 'sp_b', year_threshold=2023)

    return max(-1.0, min(1.0, score)), log



import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from itertools import combinations

def general_species_cooccurrence(df, species_list, case_col='Case #', year_col='Year',
                                 start_year=None, end_year=None):
    if start_year:
        df = df[df[year_col] >= start_year]
    if end_year:
        df = df[df[year_col] <= end_year]

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
        else:
            chi2, p = None, None
        results.append((sp_a, sp_b, chi2, p, table))
    return results

def compute_cooccurrence_score(results, weight=0.15):
    """
    Calculates the Co-occurrence Score using the formula:

        Cooccurrence Score = w_chi2 * (|S| / |P|)

    Where:
    - |S| is the number of species pairs with significant co-occurrence (p < 0.05)
    - |P| is the total number of tested species pairs
    - w_chi2 is the predefined weight assigned to this component
    """
    total_pairs = len(results)
    significant_pairs = sum(1 for _, _, _, p, _ in results if p is not None and p < 0.05)

    if total_pairs == 0:
        score = 0.0
        desc = "No species pairs to evaluate"
    else:
        proportion = significant_pairs / total_pairs
        score = weight * proportion
        desc = f"{significant_pairs}/{total_pairs} significant pairs → Score = {score:.2f}"

    return score, desc

def format_cooccurrence_table(table, sp_a, sp_b):
    """
    Formata a tabela 2x2 com rótulos de presença/ausência com base nas duas espécies analisadas.
    Exemplo:
        - Linhas: "sp_a = 0", "sp_a = 1"
        - Colunas: "sp_b = 0", "sp_b = 1"
    """
    formatted = table.copy()
    formatted.index = [f"{sp_a} = {i}" for i in formatted.index]
    formatted.columns = [f"{sp_b} = {i}" for i in formatted.columns]
    formatted.index.name = f"{sp_a} / {sp_b}"
    return formatted

def detect_emerging_cooccurrence(df, species_list, year_threshold=2023):
    """
    Identifica combinações de espécies que passaram a ocorrer juntas pela primeira vez após um determinado ano.
    Retorna uma lista de mensagens interpretativas.
    """
    df = df.dropna(subset=['Year'])
    df = df[df['Year'] >= year_threshold]

    case_species = df.groupby('Case #')['Species'].unique()
    seen_pairs = set()
    emergent_pairs = set()

    for species_list_case in case_species:
        for sp_a, sp_b in combinations(sorted(species_list_case), 2):
            pair = tuple(sorted([sp_a, sp_b]))
            if pair not in seen_pairs:
                emergent_pairs.add(pair)
            seen_pairs.add(pair)

    messages = []
    for sp_a, sp_b in sorted(emergent_pairs):
        co_cases = df[(df['Species'] == sp_a) | (df['Species'] == sp_b)]
        first_year = int(co_cases['Year'].min())
        messages.append(f"⚠️ {sp_a} + {sp_b} co-occurrence detected since {first_year}")

    if not messages:
        return "No emerging species co-occurrence detected since {year_threshold}."
    return messages


import matplotlib.pyplot as plt

def plot_expanding_mean(df, species_col='Species', year_col='Year', count_col='N_seized'):
    df_expanding = df[[year_col, species_col, count_col]].copy()
    df_expanding = df_expanding.sort_values(by=[species_col, year_col])
    df_expanding['Expanding Mean'] = (
        df_expanding.groupby(species_col)[count_col]
        .expanding().mean().shift(1).reset_index(level=0, drop=True)
    )

    species_list = df_expanding[species_col].unique()
    color_palette = plt.get_cmap('tab10')

    plt.figure(figsize=(10, 6))

    for i, species in enumerate(species_list):
        subset = df_expanding[df_expanding[species_col] == species]
        plt.plot(subset[year_col], subset[count_col], linestyle='None', marker='o',
                 label=f"{species} - N Seized", color=color_palette(i), alpha=0.6)
        plt.plot(subset[year_col], subset['Expanding Mean'], linestyle='--',
                 label=f"{species} - Previous Mean", color=color_palette(i))

    plt.title("Individual Seizure Size vs Expanding Mean by Species")
    plt.xlabel("Year")
    plt.ylabel("Number of Individuals per Case")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt.gcf())

import streamlit as st

# Supported languages
languages = {
    "English": "en",
    "Português": "pt",
    "Español": "es"
}

# Initial Interface (always in English)
col1, col2, col3 = st.columns([1, 2, 3])
with col1:
    st.image("wcs.jpg", width=200)
with col2:
    st.image("logo.jpg", width=200)
with col3:
    st.title("**Welcome to Aurum!**")
st.write("Please, select your language:")

# Language selection
selected_language = st.selectbox(
    label="Language / Idioma / Idioma",
    options=list(languages.keys())
)

# Store selected language in session_state
st.session_state['language'] = languages[selected_language]

# After language selection, show content (currently English only)
if selected_language == "English":
    st.header("🔎 What is Aurum?")
    st.markdown("""
    **Aurum** is an analytical tool designed to detect **organized crime patterns** in wildlife trafficking data.  
    It applies advanced statistical and machine learning methods to identify unusual patterns, emerging trends, 
    and potential organized crime involvement in wildlife seizures.
    
    Aurum evaluates seizure data through multiple analytical lenses, providing robust insights to support wildlife 
    law enforcement efforts and conservation policies.
    """)

    st.header("📊 Available Analyses")
    with st.expander("🔍 Click here to see detailed explanation of each analysis"):
        st.markdown("""
        Aurum offers two main pathways for analysis:

        - **Random Exploratory Analysis:** Enables quick and flexible exploration of the dataset, with tools including:

            - **Trend Analysis:** Uses segmented linear regression and the Temporal Contrast Score (TCS) to detect emerging patterns over time. 
              TCS compares historical and recent seizure trends, helping uncover sudden increases in activity that may indicate coordination or market shifts. 
              Visual tools include regression plots, expanding means by species, and optional CUSUM charts for visualizing inflection points.

            - **Species Co-occurrence:** Analyzes whether species appear together in seizures more often than expected by chance.
              Uses chi-squared tests on contingency tables to detect statistically significant co-occurrence patterns.
              Also includes a qualitative flag for newly emerging combinations of species—valuable for early warning of coordinated logistics.

            - **Anomaly Detection:** Combines five unsupervised learning techniques—Isolation Forest, Local Outlier Factor, DBSCAN, Z-score, and Mahalanobis Distance—
              to detect seizures that diverge from normal behavior (e.g., extremely large quantities, unusual species mixes, or rare countries).
              Votes from all methods are aggregated, and a consensus above a threshold contributes to the Organized Crime Score.

            - **Network Analysis:** Builds a graph where cases are nodes and connections form when seizures share key attributes (same year, country, or species).
              Dense, connected components suggest structured trafficking routes or central planning, which increases the OCS.

        - **Organized Crime Score (OCS):**  
          A composite score aggregating the four components above (Trend, Co-occurrence, Anomaly, and Network). 
          Each contributes a weighted portion to the final score, which ranges from -1.0 (no organization) to +1.0 (strong signals of coordination).
        """)

    st.write("---")
    st.write("Continue by selecting your desired analysis from the options that will be provided next.")
    

elif selected_language == "Português":
    st.header("🔎 O que é o Aurum?")
    st.markdown("""
    **Aurum** é uma ferramenta analítica projetada para detectar **padrões de crime organizado** em dados de tráfico de fauna.  
    Ele utiliza métodos estatísticos avançados e aprendizado de máquina para identificar padrões incomuns, tendências emergentes 
    e possíveis envolvimentos do crime organizado em apreensões de animais silvestres.

    O Aurum analisa os dados das apreensões por meio de múltiplas abordagens analíticas, fornecendo informações robustas para apoiar 
    as ações de fiscalização e políticas de conservação da vida selvagem.
    """)

    st.header("📊 Análises Disponíveis")
    with st.expander("🔍 Clique aqui para ver detalhes sobre cada análise"):
        st.markdown("""
        O Aurum oferece dois caminhos principais para análise:

        - **Análise exploratória aleatória:** Permite explorar os dados rapidamente e de forma flexível, com as seguintes ferramentas:

            - **Análise de Tendência:** Usa regressão linear segmentada e o Índice de Contraste Temporal (TCS) para detectar mudanças no padrão ao longo do tempo.
              O TCS compara tendências históricas e recentes de apreensões, ajudando a identificar aumentos súbitos de atividade que podem indicar coordenação ou mudanças logísticas. 
              Ferramentas visuais incluem gráficos de regressão, médias acumuladas por espécie e gráficos CUSUM opcionais para visualizar pontos de inflexão.

            - **Coocorrência de Espécies:** Avalia se espécies aparecem juntas nas apreensões mais do que seria esperado ao acaso.
              Utiliza testes qui-quadrado para detectar padrões estatisticamente significativos e inclui um alerta interpretativo para combinações emergentes de espécies — útil como sinal precoce de coordenação.

            - **Detecção de Anomalias:** Combina cinco métodos não supervisionados — Isolation Forest, Local Outlier Factor, DBSCAN, Z-score e Distância de Mahalanobis —
              para identificar apreensões que se desviam do padrão normal (ex: quantidades muito altas, misturas incomuns de espécies, países atípicos).
              Os votos dos modelos são agregados, e se o consenso ultrapassar um limiar, isso contribui positivamente para o Índice de Crime Organizado.

            - **Análise de Rede:** Constrói um gráfico onde os nós são apreensões e as conexões se formam quando casos compartilham atributos como ano, país ou espécie.
              Componentes densos e conectados sugerem rotas logísticas estruturadas ou planejamento centralizado — indícios fortes de crime organizado.

        - **Índice de Crime Organizado (OCS):**  
          Uma pontuação composta que agrega os quatro componentes anteriores (Tendência, Coocorrência, Anomalias e Rede). 
          Cada um contribui com um peso para o índice final, que varia de -1.0 (nenhum indício) até +1.0 (fortes sinais de coordenação).
        """)

    st.write("---")
    st.write("Continue selecionando a análise desejada a partir das opções que serão fornecidas em seguida.")

elif selected_language == "Español":
    st.header("🔎 ¿Qué es Aurum?")
    st.markdown("""
    **Aurum** es una herramienta analítica diseñada para detectar **patrones de crimen organizado** en datos de tráfico de fauna silvestre.  
    Aplica métodos estadísticos avanzados y aprendizaje automático para identificar patrones inusuales, tendencias emergentes 
    y posibles involucramientos del crimen organizado en incautaciones de animales silvestres.

    Aurum evalúa los datos de incautaciones mediante múltiples enfoques analíticos, brindando información sólida para apoyar 
    esfuerzos de vigilancia y políticas de conservación de la vida silvestre.
    """)

    st.header("📊 Análisis Disponibles")
    with st.expander("🔍 Haga clic aquí para ver detalles sobre cada análisis"):
        st.markdown("""
        Aurum ofrece dos formas principales de análisis:

        - **Análisis exploratorio aleatorio:** Permite explorar rápidamente los datos de incautaciones con las siguientes herramientas:

            - **Análisis de tendencias:** Utiliza regresión lineal segmentada y el Índice de Contraste Temporal (TCS) para detectar cambios recientes en el comportamiento del tráfico.
              El TCS compara tendencias históricas con tendencias recientes para identificar incrementos súbitos que pueden indicar coordinación u operaciones logísticas nuevas. 
              Se incluyen gráficos de regresión, medias acumulativas por especie y gráficos CUSUM opcionales para detectar puntos de inflexión.

            - **Co-ocurrencia de especies:** Evalúa si ciertas especies tienden a aparecer juntas con más frecuencia de la esperada por azar.
              Se aplican pruebas chi-cuadrado para detectar asociaciones estadísticamente significativas, junto con una alerta interpretativa cuando surgen combinaciones nuevas de especies en años recientes.

            - **Detección de anomalías:** Combina cinco métodos de aprendizaje no supervisado — Isolation Forest, Local Outlier Factor, DBSCAN, Z-score y Distancia de Mahalanobis —
              para identificar incautaciones atípicas (como grandes volúmenes, combinaciones inusuales de especies o países inesperados).
              Las decisiones de todos los modelos se agregan, y si hay suficiente consenso, el caso contribuye al Índice de Crimen Organizado.

            - **Análisis de redes:** Crea un grafo donde los nodos son incautaciones y las conexiones se basan en coincidencias de atributos clave (año, país, especie).
              La existencia de componentes densos y conectados puede reflejar rutas organizadas de tráfico o planificación centralizada.

        - **Índice de Crimen Organizado (OCS):**  
          Una puntuación compuesta que combina los cuatro componentes anteriores (Tendencia, Co-ocurrencia, Anomalías y Red). 
          Cada uno aporta un peso al índice final, el cual va de -1.0 (sin señales de organización) a +1.0 (fuertes indicios de coordinación).
        """)

    st.write("---")
    st.write("Continúe seleccionando el análisis deseado entre las opciones que se proporcionarán a continuación.")

import pandas as pd
import streamlit as st
import re

# --- Inicializa variável de idioma com fallback ---
lang = st.session_state.get('language', 'en')

# --- Mensagens multilíngues ---
file_upload_labels = {
    'en': "📂 Please upload your Excel file (.xlsx) containing wildlife seizure data:",
    'pt': "📂 Por favor, faça upload do arquivo Excel (.xlsx) contendo os dados de apreensão de fauna:",
    'es': "📂 Por favor, cargue el archivo Excel (.xlsx) que contiene los datos de incautaciones de fauna:"
}

upload_success_msg = {
    'en': "✅ File uploaded and cleaned successfully!",
    'pt': "✅ Arquivo carregado e limpo com sucesso!",
    'es': "✅ ¡Archivo cargado y limpiado con éxito!"
}

columns_msg = {
    'en': "📋 The uploaded file contains these columns:",
    'pt': "📋 O arquivo carregado contém as seguintes colunas:",
    'es': "📋 El archivo cargado contiene estas columnas:"
}

# --- Upload do arquivo ---
uploaded_file = st.file_uploader(file_upload_labels[lang], type=["xlsx"])
run_trend = run_cooccurrence = run_anomaly = run_network = run_ocs = False
co_score = 0.0
selected_species = []
df_selected = pd.DataFrame()

# Se arquivo for carregado, salva na sessão
if uploaded_file is not None:
    st.session_state['uploaded_file'] = uploaded_file

# Se há um arquivo salvo na sessão, processa
if st.session_state.get('uploaded_file') is not None:
    df_raw = pd.read_excel(st.session_state['uploaded_file'])

    # --- Função atualizada para expandir múltiplas espécies ---
    def expand_multi_species_rows(df, column='N seized specimens'):
        """
        Corrige casos onde há múltiplas espécies no formato:
        '12 FS03 + 5 FS04' → separa corretamente em linhas distintas com quantidade e nome.
        """
        expanded_rows = []

        for _, row in df.iterrows():
            cell = str(row.get(column, "")).strip()

            # Divide por + (com ou sem espaços)
            parts = re.split(r'\s*\+\s*', cell)

            for part in parts:
                # Tenta capturar quantidade + nome da espécie
                match = re.match(r'^(\d+)\s+(.+)$', part.strip())
                if match:
                    qty, species = match.groups()
                    new_row = row.copy()
                    new_row['N_seized'] = float(qty)
                    new_row['Species'] = species.strip()
                    expanded_rows.append(new_row)
                else:
                    # Se não encontrar número, ignora ou define N_seized como 1
                    continue

        return pd.DataFrame(expanded_rows).reset_index(drop=True)


    # --- Aplica a limpeza ---
    df_clean = expand_multi_species_rows(df_raw)

    # ===============================
    # 🔄 INFERÊNCIA AUTOMÁTICA DE COLUNAS PARA O MODELO SARA
    # ===============================
    
    # --- 1. LOGISTIC CONVERGENCE: marca "Yes" se um caso tiver mais de uma espécie ---
    species_per_case = df_clean.groupby("Case #")["Species"].nunique()
    df_clean["Logistic Convergence"] = df_clean["Case #"].map(
        lambda x: "Yes" if species_per_case.get(x, 0) > 1 else "No"
    )

    # --- 2. INFERRED STAGE: infere o estágio com base no status, local e convergência ---
    import re
    import unicodedata

    # 🔧 Função auxiliar para normalizar texto
    def normalize_text(text):
        if not isinstance(text, str):
            text = str(text)
        text = text.strip().lower()
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r'\s+', ' ', text)  # remove múltiplos espaços
        return text

    # --- 2. INFERRED STAGE: infere o estágio com base no status, local e convergência ---
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

    df_clean["Inferred Stage"] = df_clean.apply(infer_stage, axis=1)

    

    # --- Mostra mensagens e dados processados ---
    st.success(upload_success_msg[lang])
    st.subheader(columns_msg[lang])
    st.write(list(df_clean.columns))

    st.subheader("🔍 Preview of cleaned data:")
    st.dataframe(df_clean.head())

    # --- Seleção de espécies para análise ---
    st.subheader("🧬 Choose species for analysis:")

    species_options = sorted(df_clean['Species'].dropna().unique())
    selected_species = st.multiselect("Select one or more species:", species_options)

    # Mensagem adaptada por idioma
    species_warning = {
        'en': "⚠️ Please select at least one species to proceed.",
        'pt': "⚠️ Por favor, selecione ao menos uma espécie para continuar.",
        'es': "⚠️ Por favor, seleccione al menos una especie para continuar."
    }

    if not selected_species:
        st.warning(species_warning[lang])

if selected_species:
    df_selected = df_clean[df_clean['Species'].isin(selected_species)]

    st.markdown("## 📊 Visualização Personalizada")
    st.markdown("Use esse painel para explorar visualmente seus dados antes das análises avançadas.")

    chart_type = st.selectbox("📍 Escolha o tipo de gráfico:", ["Barras", "Pizza", "Linha", "Dispersão"])
    x_axis = st.selectbox("🧭 Eixo X:", df_selected.columns)
    y_axis = st.selectbox("📐 Eixo Y:", df_selected.columns)

    import plotly.express as px
    if chart_type == "Barras":
        fig = px.bar(df_selected, x=x_axis, y=y_axis, color='Species', title="📊 Gráfico de Barras")
        st.plotly_chart(fig)

    elif chart_type == "Pizza":
        fig = px.pie(df_selected, names=x_axis, values=y_axis, title="🥧 Gráfico de Pizza")
        st.plotly_chart(fig)

    elif chart_type == "Linha":
        fig = px.line(df_selected, x=x_axis, y=y_axis, color='Species', title="📈 Gráfico de Linhas", markers=True)
        st.plotly_chart(fig)

    elif chart_type == "Dispersão":
        fig = px.scatter(df_selected, x=x_axis, y=y_axis, color='Species', title="🔘 Gráfico de Dispersão")
        st.plotly_chart(fig)

    st.success(f"✅ {len(df_selected)} registros selecionados para análise.")

    st.markdown("### 📊 Choose the analyses you want to perform:")

    run_trend = st.checkbox("📈 Trend Analysis", value=True)
    run_cooccurrence = st.checkbox("🧬 Species Co-occurrence", value=True)
    run_anomaly = st.checkbox("🚨 Anomaly Detection", value=True)
    run_network = st.checkbox("🕸️ Network Analysis", value=True)
    run_ocs = st.checkbox("🧮 Calculate Organized Crime Score (OCS)", value=True)

    if run_trend:
            st.subheader("📈 Trend Analysis")

            # Permitir escolher breakpoint
            default_breakpoint = st.session_state.get("trend_breakpoint", 2015)
            breakpoint_year = st.number_input(
                "Choose a breakpoint year (used to split time series into two phases):",
                min_value=1990, max_value=2030, value=int(default_breakpoint), step=1
            )
            st.session_state["trend_breakpoint"] = breakpoint_year

            # Executa a análise com o breakpoint atual
            tcs, trend_log = trend_component(
                df_selected, year_col="Year", count_col="N_seized", breakpoint=breakpoint_year
            )

            st.markdown(f"**Trend Coordination Score (TCS):** `{tcs:.2f}`")
            st.info(trend_log)

            # --- Gerar gráfico de tendência ---
            def plot_trend_split(df, breakpoint, species_list):
                fig, ax = plt.subplots(figsize=(8, 5))

                for species in species_list:
                    subset = df[df["Species"] == species]
                    ax.scatter(subset["Year"], subset["N_seized"], label=f"{species}", alpha=0.6)

                    df_pre = subset[subset["Year"] <= breakpoint]
                    df_post = subset[subset["Year"] > breakpoint]

                    if len(df_pre) > 1:
                        model_pre = sm.OLS(df_pre["N_seized"], sm.add_constant(df_pre["Year"])).fit()
                        ax.plot(df_pre["Year"], model_pre.predict(sm.add_constant(df_pre["Year"])), linestyle="--")

                    if len(df_post) > 1:
                        model_post = sm.OLS(df_post["N_seized"], sm.add_constant(df_post["Year"])).fit()
                        ax.plot(df_post["Year"], model_post.predict(sm.add_constant(df_post["Year"])), linestyle="-.")

                ax.axvline(breakpoint, color='red', linestyle=':', label=f"Breakpoint = {breakpoint}")
                ax.set_title("📈 Trend split by breakpoint year")
                ax.set_xlabel("Year")
                ax.set_ylabel("Number of Individuals Seized")
                ax.grid(True)
                ax.legend()
                return fig

            fig = plot_trend_split(df_selected, breakpoint_year, selected_species)
            st.pyplot(fig)

            # Opções visuais adicionais
            st.markdown("🛠️ Optional visual tools:")
            show_cusum = st.checkbox("📉 Show CUSUM plot", value=False)
            show_expanding = st.checkbox("📈 Show Expanding Mean plot", value=False)

            if show_cusum:
                st.subheader("📉 CUSUM Plot")

                def plot_cusum(df, species, year_col="Year", count_col="N_seized"):
                    subset = df[df["Species"] == species]
                    if subset.empty:
                        return

                    # Agrupar por ano e somar
                    yearly = subset.groupby(year_col)[count_col].sum().sort_index()
                    years = yearly.index
                    values = yearly.values

                    mean = values.mean()
                    cusum = (values - mean).cumsum()

                    cusum_pos = [v if v > 0 else np.nan for v in cusum]
                    cusum_neg = [v if v < 0 else np.nan for v in cusum]

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(years, values, color='black', marker='o', label='Seizure Total (Yearly)', linewidth=1.5)
                    ax.plot(years, cusum_pos, color='blue', label='CUSUM +', linewidth=2)
                    ax.plot(years, cusum_neg, color='red', label='CUSUM -', linewidth=2)
                    ax.axhline(0, color='gray', linestyle='--')
                    ax.set_title(f"CUSUM and Total – {species}")
                    ax.set_xlabel("Year")
                    ax.set_ylabel("Individuals / CUSUM")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)

                for species in selected_species:
                    plot_cusum(df_selected, species)

            if show_expanding:
                st.subheader("📈 Expanding Mean Plot")
                plot_expanding_mean(df_selected, species_col="Species", year_col="Year", count_col="N_seized")

            # Refinar análise com novo breakpoint
            st.markdown("---")
            st.markdown("🔁 Do you want to test with a different breakpoint?")
            if st.checkbox("Refine trend analysis with a new breakpoint"):
               new_breakpoint = st.slider("Select a new breakpoint year:", 1990, 2030, value=breakpoint_year)
               st.session_state["trend_breakpoint"] = new_breakpoint  # atualiza o estado global

               tcs, trend_log = trend_component(df_selected, year_col="Year", count_col="N_seized", breakpoint=new_breakpoint)
               st.markdown(f"**New TCS:** `{tcs:.2f}`")
               st.info(trend_log)

               fig = plot_trend_split(df_selected, new_breakpoint, selected_species)
               st.pyplot(fig)


            # Exibir detalhes da regressão
            st.markdown("🔎 Do you want more detailed results?")
            if st.checkbox("Show slope, R² and p-values for each period"):
                detailed_results = []

                for species in selected_species:
                    row = {"Species": species}
                    subset = df_selected[df_selected["Species"] == species]

                    sub_pre = subset[subset["Year"] <= breakpoint_year]
                    if len(sub_pre) > 1:
                        X_pre = sm.add_constant(sub_pre["Year"])
                        y_pre = sub_pre["N_seized"]
                        model_pre = sm.OLS(y_pre, X_pre).fit()
                        row["Slope (pre)"] = model_pre.params["Year"]
                        row["R² (pre)"] = model_pre.rsquared
                        row["p (pre)"] = model_pre.pvalues["Year"]
                    else:
                        row["Slope (pre)"] = row["R² (pre)"] = row["p (pre)"] = None

                    sub_post = subset[subset["Year"] > breakpoint_year]
                    if len(sub_post) > 1:
                        X_post = sm.add_constant(sub_post["Year"])
                        y_post = sub_post["N_seized"]
                        model_post = sm.OLS(y_post, X_post).fit()
                        row["Slope (post)"] = model_post.params["Year"]
                        row["R² (post)"] = model_post.rsquared
                        row["p (post)"] = model_post.pvalues["Year"]
                    else:
                        row["Slope (post)"] = row["R² (post)"] = row["p (post)"] = None

                    detailed_results.append(row)

                df_details = pd.DataFrame(detailed_results)
                st.markdown("📋 **Detailed Trend Regression Results**")
                st.dataframe(df_details.style.format({
                    "Slope (pre)": "{:.2f}", "R² (pre)": "{:.2f}", "p (pre)": "{:.4f}",
                    "Slope (post)": "{:.2f}", "R² (post)": "{:.2f}", "p (post)": "{:.4f}"
                }))

if run_cooccurrence:
    st.subheader("🧬 Species Co-occurrence Analysis")

    if st.button("🔄 Run / Refresh Co-occurrence Analysis"):
        species_list = df_selected["Species"].dropna().unique().tolist()
        co_results = general_species_cooccurrence(df_selected, species_list)
        st.session_state["co_results"] = co_results
        st.session_state["species_list"] = species_list
    else:
        co_results = st.session_state.get("co_results", [])
        species_list = st.session_state.get("species_list", [])

    if co_results:
        co_score, co_log = compute_cooccurrence_score(co_results, weight=0.15)
        st.markdown(f"**Species Co-occurrence Score:** `{co_score:.2f}`")
        st.info(co_log)

        st.markdown("### 🚩 Emerging Co-occurrence Alerts")
        flags = detect_emerging_cooccurrence(df_selected, species_list, year_threshold=2023)
        if isinstance(flags, list):
            for msg in flags:
                st.warning(msg)
        else:
            st.info(flags)

        # Mostrar detalhes dos pares
        if st.checkbox("📋 Show full 2×2 tables and test details"):
            for sp_a, sp_b, chi2, p, table in co_results:
                st.markdown(f"**{sp_a} × {sp_b}**")

                formatted = format_cooccurrence_table(table, sp_a, sp_b)
                st.dataframe(formatted)

                if chi2 is not None:
                    st.markdown(f"Chi² = `{chi2:.2f}` | p = `{p:.4f}`")
                else:
                    st.markdown("Not enough data to compute chi².")
                st.markdown("---")

# Explicação automática detalhada para o operador

# ======================
# 🔎 SARA PANEL
# ======================
if 'Inferred Stage' in df_selected.columns:
    st.markdown("## 🧠 SARA Model – Wildlife Crime Staging")

    selected_sp = st.selectbox("Select a species for SARA overview:", df_selected['Species'].unique())

    df_sara = df_selected[df_selected['Species'] == selected_sp].copy()

    # ✅ Normaliza rótulos longos para interpretar corretamente
    df_sara['Stage Category'] = df_sara['Inferred Stage'].astype(str).str.extract(r'^([^\(:]+)').iloc[:, 0].str.strip()

    st.markdown("### 📋 Inferred Stages (Summary Table)")
    available_cols = [col for col in ['Case #', 'Year', 'Seizure Status', 'Transit Feature', 'Inferred Stage'] if col in df_sara.columns]
    st.dataframe(df_sara[available_cols])

    st.markdown("### 📊 Distribution of Stages")
    stage_counts = df_sara['Stage Category'].value_counts()
    st.bar_chart(stage_counts)

    st.markdown("### 🧠 Interpretation")
    if "Preparation" in stage_counts.index:
        st.info("Some cases were classified as **Preparation**, indicating early stages of wildlife crime such as planning, installation of traps, or organization of collection efforts. These incidents often precede the actual capture and reflect the intentionality of illicit activity.")
    if "Captivity" in stage_counts.index:
        st.info("There are records involving **Captivity**, referring to animals held in illegal enclosures, breeding facilities, or private collections. These cases may represent either the origin (source) or a temporary holding point in the trafficking route.")
    if "Logistic Consolidation" in stage_counts.index:
        st.success("There are cases involving **Logistic Consolidation**, where multiple species were found together in the same seizure. This may indicate centralized handling, organized trafficking operations, or shared supply chains among actors.")
    if "Post-capture Displacement" in stage_counts.index:
        st.warning("Some seizures were recorded far from the native range of the species involved. This suggests **Post-capture Displacement**, where wildlife has been moved away from its original habitat, increasing the likelihood of trafficking and ecological impact.")
    if "Transport" in stage_counts.index:
        st.warning("Some seizures occurred during **Transport**, such as at airports, borders, highways, or ports. This stage suggests that the wildlife had already been captured and was in transit — often across jurisdictions — indicating operational mobility of traffickers.")
    if "Unclassified" in stage_counts.index:
        st.info("A number of records could not be automatically classified into a specific stage. These may lack sufficient information or fall outside the defined typology. Manual review is recommended.")

    st.markdown("---")


# Explicação automática multilíngue
if run_cooccurrence and co_results:
    with st.expander("🧠 Interpretation / Interpretação / Interpretación"):
        if lang == "pt":
            if co_score > 0.1:
                st.success("O escore de coocorrência é alto, indicando que as espécies tendem a aparecer juntas nas apreensões com maior frequência do que o esperado ao acaso. Isso pode refletir uma logística estruturada ou fontes de origem compartilhadas.")
            elif co_score > 0.0:
                st.warning("Alguns pares de espécies apresentam sinais iniciais de coocorrência, embora o padrão ainda seja fraco. Pode ser um indício precoce de coordenação.")
            else:
                st.info("Não há evidências estatísticas de coocorrência entre as espécies analisadas. Os eventos de tráfico parecem ocorrer de forma independente.")

            st.markdown("""
            ### 🔬 Como esse escore é calculado?
            O **Cooccurrence Score** reflete a proporção de pares de espécies com associação significativa (p < 0.05), ajustado pelo peso do componente:

            Coocorrência = (pares significativos ÷ total de pares) × peso do componente
            
            Onde:
            - \( |S| \) = número de pares com p < 0.05  
            - \( |P| \) = total de pares testados  
            - \( w_{\chi^2} \) = peso desse componente no índice OCS
            """)

        elif lang == "es":
            if co_score > 0.1:
                st.success("El puntaje de coocurrencia es alto, lo que indica que las especies tienden a aparecer juntas en incautaciones más a menudo de lo esperado por azar. Esto puede reflejar logísticas compartidas o rutas comunes.")
            elif co_score > 0.0:
                st.warning("Algunos pares de especies muestran signos iniciales de coocurrencia, aunque el patrón sigue siendo débil. Puede ser una señal temprana de coordinación.")
            else:
                st.info("No hay evidencia estadística de que las especies aparezcan juntas más de lo esperado por azar. Las incautaciones parecen independientes.")

            st.markdown("""
            ### 🔬 ¿Cómo se calcula este puntaje?
            El **Cooccurrence Score** refleja la proporción de pares de especies significativamente asociados (p < 0.05), ajustado por su peso:

            Coocurrencia = (pares significativos ÷ total de pares) × peso del componente
            
            Donde:
            - \( |S| \) = número de pares con p < 0.05  
            - \( |P| \) = total de pares evaluados  
            - \( w_{\chi^2} \) = peso del componente en el OCS
            """)

        else:
            if co_score > 0.1:
                st.success("The co-occurrence score is high, suggesting that the species tend to be trafficked together more often than by chance. This may indicate coordinated logistics or shared sourcing.")
            elif co_score > 0.0:
                st.warning("Some species pairs are showing weak but emerging co-occurrence patterns. These may reflect early signs of coordination.")
            else:
                st.info("No statistical evidence of non-random co-occurrence. Most trafficking appears to occur independently between species.")

            st.markdown("""
            ### 🔬 How is this score calculated?
            The **Co-occurrence Score** reflects the proportion of significantly associated species pairs (p < 0.05), weighted for the OCS:

            Co-occurrence = (significant pairs ÷ total pairs) × component weight
            
            Where:
            - \( |S| \): number of significant pairs  
            - \( |P| \): total pairs tested  
            - \( w_{\chi^2} \): component weight in the final score
            """)
