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
    layout="wide"
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
        "data_error_title": "Hiba az adatfájl betöltésekor!",
        "data_error_body": "Nem található a `../data/optimization_results.h5` fájl. Kérjük, ellenőrizze a fájl elérési útját és a mappaszerkezetet.",
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
        "data_error_title": "Error loading data file!",
        "data_error_body": "The file `../data/optimization_results.h5` was not found. Please check the file path and folder structure.",
        "what_is_this_title": "What is this app?",
    }
}


# --- Adatbetöltés Gyorsítótárazással ---
@st.cache_data
def load_data(file_path: Path):
    """
    Betölti és feldolgozza az adatokat a HDF5 fájlból.
    A Streamlit @st.cache_data dekorátora biztosítja, hogy a 38 perces futás
    eredményeit csak egyszer kelljen beolvasni a memóriába.
    """
    if not file_path.exists():
        return None

    all_portfolios = []
    with h5py.File(file_path, 'r') as f:
        # Kikeressük, hány portfólió van a fájlban
        num_points = len([key for key in f.keys() if key.startswith('point_')])
        
        for i in range(1, num_points + 1):
            group_name = f'point_{i:02d}'
            grp = f[group_name]
            
            # Loss history DataFrame rekonstruálása
            loss_values = grp['loss_history_values'][:]
            loss_cols = grp['loss_history_values'].attrs['columns']
            loss_history_df = pd.DataFrame(loss_values, columns=loss_cols)

            point_data = {
                "id": i,
                "weights": pd.DataFrame(grp['weights'][:]), # Oszlopneveket később adjuk hozzá
                "achieved_wealth": grp['achieved_wealth'][()],
                "terminal_cCVaR": grp['terminal_cCVaR'][()],
                "loss_history": loss_history_df
            }
            all_portfolios.append(point_data)

    # Diagnosztikai táblázat elkészítése
    diagnostics_list = []
    for p_data in all_portfolios:
        metrics = {
            "Portfolio_ID": p_data['id'],
            "Annualized_Return": p_data['achieved_wealth']**(12 / 60) - 1, # 60 hónapra
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
    fig.add_trace(go.Scatter(
        x=df_sorted['Annualized_Risk_cCVaR'], y=df_sorted['Annualized_Return'],
        mode='lines', line=dict(color='crimson', width=3), name=t['frontier_line']
    ))
    fig.add_trace(go.Scatter(
        x=df['Annualized_Risk_cCVaR'], y=df['Annualized_Return'],
        mode='markers+text', text=df['Portfolio_ID'], textposition="top center",
        marker=dict(color='crimson', size=12, opacity=0.8), name=t['frontier_points']
    ))
    fig.update_layout(
        title=t['frontier_header'],
        xaxis_title=t['frontier_xaxis'],
        yaxis_title=t['frontier_yaxis'],
        xaxis_tickformat='.0%',
        yaxis_tickformat='.0%',
        showlegend=False,
        template='plotly_white',
        height=500
    )
    return fig

def plot_allocation(weights_df, lang):
    t = TRANSLATIONS[lang]
    # A notebook-ból származó oszlopnevek. Ezt dinamikusan is lehetne tölteni.
    asset_names = ['USD Cash', 'HY Corp', 'US Large Cap', 'US Small Cap', 'US REIT', 'Gold', 
                   'Agriculture', 'Short Gov', 'Mid Gov', 'Long Gov', 'IG Corp']
    weights_df.columns = asset_names
    
    df_plot = weights_df.copy()
    df_plot[t['allocation_xaxis']] = range(1, len(df_plot) + 1)
    df_melted = df_plot.melt(id_vars=t['allocation_xaxis'], var_name=t['allocation_legend'], value_name=t['allocation_yaxis'])

    # Színpaletta, ami esztétikus
    color_map = px.colors.qualitative.Plotly
    
    fig = px.area(df_melted, x=t['allocation_xaxis'], y=t['allocation_yaxis'], color=t['allocation_legend'],
                  title=t['allocation_title'], labels={'value': t['allocation_yaxis']},
                  color_discrete_sequence=color_map)
    fig.update_layout(
        yaxis_tickformat='.0%',
        legend_title_text=t['allocation_legend'],
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    return fig

def plot_convergence(loss_df, lang):
    t = TRANSLATIONS[lang]
    df_melted = loss_df.melt(id_vars='Epoch', var_name=t['convergence_legend'], value_name=t['convergence_yaxis'])
    
    fig = px.line(df_melted, x='Epoch', y=t['convergence_yaxis'], color=t['convergence_legend'],
                  title=t['convergence_title'], facet_col=t['convergence_legend'], facet_col_wrap=5)
    fig.update_yaxes(matches=None, title_text="") # Eltávolítjuk a tengelycímkéket a tisztább képért
    fig.update_xaxes(title_text=t['convergence_xaxis'])
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1])) # Facet címkék tisztítása
    fig.update_layout(
        showlegend=False,
        template='plotly_white'
    )
    return fig

# --- App Törzse ---

# Adatok betöltése
HDF5_FILE = Path(__file__).parent / ".." / "data" / "optimization_results.h5"
loaded_data = load_data(HDF5_FILE)

# Sidebar felépítése
with st.sidebar:
    st.image("https://i.imgur.com/r3z1Q2L.png", width=150) # Egy kis branding :)
    lang_code = st.radio(
        label="Language / Nyelv",
        options=["en", "hu"],
        format_func=lambda x: "English" if x == "en" else "Magyar",
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
            value=portfolio_ids[len(portfolio_ids) // 2] # Kezdés középről
        )
    else:
        selected_portfolio_id = 1 # Placeholder, ha nincs adat
    
    with st.expander(t['what_is_this_title']):
        st.markdown(t['intro_text'])


# Fő oldal címe
st.title(t['page_title'])

# Hibakezelés, ha a fájl nem található
if not loaded_data:
    st.error(t['data_error_title'], icon="🚨")
    st.warning(t['data_error_body'])
else:
    # 1. Szekció: Áttekintés
    st.header(t['summary_header'])
    
    col1, col2 = st.columns([2, 3]) # Oszlopok a táblázatnak és a frontnak
    
    with col1:
        st.subheader(t['diag_table_title'])
        st.dataframe(
            diagnostics_df.style.format({
                "Annualized_Return": "{:.2%}",
                "Annualized_Risk_cCVaR": "{:.2%}"
            }),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(plot_efficient_frontier(diagnostics_df, lang_code), use_container_width=True)

    st.divider()

    # 2. Szekció: Részletes Elemzés
    st.header(f"{t['detail_header']} {selected_portfolio_id}")
    
    selected_portfolio_data = next((p for p in all_portfolios if p['id'] == selected_portfolio_id), None)
    
    if selected_portfolio_data:
        
        # Allokáció és Konvergencia egymás alatt
        
        with st.container(border=True):
             st.plotly_chart(plot_allocation(selected_portfolio_data['weights'], lang_code), use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True) # Kis térköz
        
        with st.container(border=True):
            st.plotly_chart(plot_convergence(selected_portfolio_data['loss_history'], lang_code), use_container_width=True)