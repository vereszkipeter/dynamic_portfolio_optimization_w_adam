import streamlit as st
import pandas as pd
import numpy as np
import h5py
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- Oldal Konfiguráció ---
st.set_page_config(
    page_title="Portfólió Optimalizáció Elemző",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Kétnyelvű Tartalom ---
TRANSLATIONS = {
    "hu": {
        "page_title": "Dinamikus portfólió optimalizáció elemző",
        "intro_text": """
        Ez az interaktív alkalmazás a BVAR-SV + Entrópia Pooling + Többperiódusos Optimalizáció munkafolyamat eredményeit mutatja be.
        A lenti ábrák és táblázatok segítségével felfedezheti a hatékonysági frontot, és részletesen megvizsgálhatja az egyes optimális portfóliók időbeli allokációs pályáját és konvergenciáját.
        """,
        "sidebar_header": "Vezérlőpult",
        "language_select": "Nyelv",
        "portfolio_select": "Válasszon egy portfóliót a részletes elemzéshez:",
        "summary_header": "📊 Eredmények áttekintése",
        "diag_table_title": "Diagnosztikai táblázat",
        "frontier_header": "🗺️ A hatékonysági front",
        "frontier_xaxis": "Évesített kockázat (cCVaR)",
        "frontier_yaxis": "Évesített hozam",
        "frontier_points": "Portfólió pontok",
        "frontier_line": "Hatékony front",
        "detail_header": "🔬 Részletes elemzés: Portfólió",
        "allocation_title": "Dinamikus allokációs pálya",
        "allocation_xaxis": "Hónap",
        "allocation_yaxis": "Súly",
        "allocation_legend": "Eszköz",
        "convergence_title": "Optimalizáció konvergenciája",
        "convergence_xaxis": "Epoch",
        "convergence_yaxis": "Veszteség értéke",
        "convergence_legend": "Veszteség típusa",
        "data_error_title": "Hiba az adatfájlok betöltésekor!",
        "data_error_body": "Nem található a `../data/optimization_results.h5` vagy a `../data/financial_factors_meta.csv.gz` fájl. Kérjük, ellenőrizze az elérési utakat.",
        "what_is_this_title": "Mi ez az alkalmazás?",
    },
    "en": {
        "page_title": "Dynamic Portfolio Optimization Analyzer",
        "intro_text": """
        This interactive application visualizes the results of the BVAR-SV + Entropy Pooling + Multi-Period Optimization workflow.
        Using the charts and tables below, you can explore the efficient frontier and examine the dynamic allocation path and convergence properties of each optimal portfolio in detail.
        """,
        "sidebar_header": "Controls",
        "language_select": "Language",
        "portfolio_select": "Select a portfolio for detailed analysis:",
        "summary_header": "📊 Results Overview",
        "diag_table_title": "Diagnostics Summary",
        "frontier_header": "🗺️ The Efficient Frontier",
        "frontier_xaxis": "Annualized Risk (cCVaR)",
        "frontier_yaxis": "Annualized Return",
        "frontier_points": "Portfolio Points",
        "frontier_line": "Efficient Frontier",
        "detail_header": "🔬 Detailed Analysis: Portfolio",
        "allocation_title": "Dynamic Allocation Path",
        "allocation_xaxis": "Month",
        "allocation_yaxis": "Weight",
        "allocation_legend": "Asset",
        "convergence_title": "Optimization Convergence",
        "convergence_xaxis": "Epoch",
        "convergence_yaxis": "Loss Value",
        "convergence_legend": "Loss Type",
        "data_error_title": "Error loading data files!",
        "data_error_body": "Could not find `../data/optimization_results.h5` or `../data/financial_factors_meta.csv.gz`. Please check the file paths.",
        "what_is_this_title": "What is this app?",
    }
}

# --- Adatbetöltés Gyorsítótárazással ---
@st.cache_data
def load_data(results_path: Path, meta_path: Path):
    if not results_path.exists() or not meta_path.exists():
        return None

    meta_df = pd.read_csv(meta_path)
    investable_meta = meta_df[meta_df['type'] == 'asset_return'].copy()
    asset_names = investable_meta['name'].tolist()

    all_portfolios = []
    with h5py.File(results_path, 'r') as f:
        num_points = len([key for key in f.keys() if key.startswith('point_')])
        
        for i in range(1, num_points + 1):
            group_name = f'point_{i:02d}'
            grp = f[group_name]
            
            loss_values = grp['loss_history_values'][:]
            loss_cols = grp['loss_history_values'].attrs['columns']
            loss_history_df = pd.DataFrame(loss_values, columns=loss_cols)

            weights_data = grp['weights'][:]
            if weights_data.shape[1] != len(asset_names):
                st.error(f"Inconsistency in Portfolio P{i}: {weights_data.shape[1]} weights found, but {len(asset_names)} asset names expected.")
                columns = [f"Asset_{j+1}" for j in range(weights_data.shape[1])]
            else:
                columns = asset_names

            point_data = {
                "id": i,
                "weights": pd.DataFrame(weights_data, columns=columns),
                "achieved_wealth": grp['achieved_wealth'][()],
                "terminal_cCVaR": grp['terminal_cCVaR'][()],
                "loss_history": loss_history_df
            }
            all_portfolios.append(point_data)

    diagnostics_list = []
    for p_data in all_portfolios:
        metrics = {
            "Portfolio_ID": p_data['id'],
            "Annualized_Return": p_data['achieved_wealth']**(12 / 60) - 1,
            "Annualized_Risk_cCVaR": p_data['terminal_cCVaR'] / np.sqrt(60 / 12)
        }
        diagnostics_list.append(metrics)
    
    diagnostics_df = pd.DataFrame(diagnostics_list)
    return all_portfolios, diagnostics_df

# --- Vizuális Függvények ---
def plot_efficient_frontier(df, lang):
    t = TRANSLATIONS[lang]
    df_sorted = df.sort_values("Annualized_Risk_cCVaR").copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sorted['Annualized_Risk_cCVaR'], y=df_sorted['Annualized_Return'], mode='lines', line=dict(color='crimson', width=3), name=t['frontier_line']))
    fig.add_trace(go.Scatter(x=df['Annualized_Risk_cCVaR'], y=df['Annualized_Return'], mode='markers+text', text=df['Portfolio_ID'], textposition="top center", marker=dict(color='crimson', size=12, opacity=0.8), name=t['frontier_points']))
    fig.update_layout(title=t['frontier_header'], xaxis_title=t['frontier_xaxis'], yaxis_title=t['frontier_yaxis'], xaxis_tickformat='.0%', yaxis_tickformat='.0%', showlegend=False, template='plotly_white', height=500)
    return fig

def plot_allocation(weights_df, lang):
    t = TRANSLATIONS[lang]
    df_plot = weights_df.copy()
    df_plot[t['allocation_xaxis']] = range(1, len(df_plot) + 1)
    df_melted = df_plot.melt(id_vars=t['allocation_xaxis'], var_name=t['allocation_legend'], value_name=t['allocation_yaxis'])
    fig = px.area(df_melted, x=t['allocation_xaxis'], y=t['allocation_yaxis'], color=t['allocation_legend'], title=t['allocation_title'], color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_layout(yaxis_tickformat='.0%', legend_title_text=t['allocation_legend'], template='plotly_white', legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
    return fig

def plot_convergence(loss_df, lang):
    t = TRANSLATIONS[lang]
    df_melted = loss_df.melt(id_vars='Epoch', var_name=t['convergence_legend'], value_name=t['convergence_yaxis'])
    fig = px.line(df_melted, x='Epoch', y=t['convergence_yaxis'], color=t['convergence_legend'], title=t['convergence_title'], facet_col=t['convergence_legend'], facet_col_wrap=5)
    fig.update_yaxes(matches=None, title_text="")
    fig.update_xaxes(title_text=t['convergence_xaxis'])
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    fig.update_layout(showlegend=False, template='plotly_white')
    return fig

# --- App Törzse ---

# Adatfájlok útvonalai
RESULTS_FILE = Path(__file__).parent / ".." / "data" / "optimization_results.h5"
META_FILE = Path(__file__).parent / ".." / "data" / "financial_factors_meta.csv.gz"

# Adatbetöltés
loaded_data = load_data(RESULTS_FILE, META_FILE)

# Sidebar
with st.sidebar:
    st.image("app/optimization.png", width=100)
    
    # Nyelvválasztó a sidebar-ban
    lang_code = st.radio(
        label="Language / Nyelv",
        options=["hu", "en"],
        format_func=lambda x: "Magyar" if x == "hu" else "English",
        index=0, # Alapértelmezett a magyar
        horizontal=True,
    )
    t = TRANSLATIONS[lang_code]

    st.header(t['sidebar_header'])

    if loaded_data:
        all_portfolios, diagnostics_df = loaded_data
        portfolio_ids = [p['id'] for p in all_portfolios]
        selected_portfolio_id = st.select_slider(
            t['portfolio_select'],
            options=portfolio_ids,
            value=portfolio_ids[len(portfolio_ids) // 2]
        )
    else:
        selected_portfolio_id = 1
    
    with st.expander(t['what_is_this_title'], expanded=True):
        st.markdown(t['intro_text'])

# Fő oldal címe
st.title(t['page_title'])

# Fő tartalmi rész
if not loaded_data:
    st.error(t['data_error_title'], icon="🚨")
    st.warning(t['data_error_body'])
else:
    all_portfolios, diagnostics_df = loaded_data
    
    # 1. Szekció: Áttekintés
    st.header(t['summary_header'])
    
    col1_main, col2_main = st.columns([0.8, 1.2])
    with col1_main:
        st.subheader(t['diag_table_title'])
        st.dataframe(
            diagnostics_df.style.format({
                "Annualized_Return": "{:.2%}",
                "Annualized_Risk_cCVaR": "{:.2%}"
            }),
            use_container_width=True,
            hide_index=True
        )

    with col2_main:
        st.plotly_chart(plot_efficient_frontier(diagnostics_df, lang_code), use_container_width=True)

    st.divider()

    # 2. Szekció: Részletes Elemzés
    st.header(f"{t['detail_header']} {selected_portfolio_id}")
    
    selected_portfolio_data = next((p for p in all_portfolios if p['id'] == selected_portfolio_id), None)
    
    if selected_portfolio_data:
        with st.container(border=True):
             st.plotly_chart(plot_allocation(selected_portfolio_data['weights'], lang_code), use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.plotly_chart(plot_convergence(selected_portfolio_data['loss_history'], lang_code), use_container_width=True)