import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="The keyword arguments have been deprecated.*")

# --- Oldal Konfiguráció ---
st.set_page_config(
    page_title="Dinamikus Portfólió Analizátor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Kétnyelvű Tartalom (Teljes, Bővített) ---
TRANSLATIONS = {
    "hu": {
        "page_title": "Dinamikus portfólió optimalizáció elemző",
        "sidebar_header": "Vezérlőpult",
        "portfolio_select": "Válasszon egy portfóliót:",
        "tab_intro": "Bevezető",
        "tab_historical": "Historikus adatok",
        "tab_frontier": "Hatékony front",
        "tab_allocation": "Allokáció és kényszerek",
        "tab_distribution": "Jövőbeli eloszlások",
        "tab_diagnostics": "Diagnosztika",
        "intro_header": "A kvantitatív munkafolyamat áttekintése",
        "step1_header": "1. Lépés: Piaci modell (BVAR-MSH)",
        "step1_text": """
        A piaci dinamikák előrejelzéséhez egy Bayes-i Vektor Autoregressziós modellt használunk Markov-rezsimváltós heteroszkedaszticitással (BVAR-MSH).
        - **Miért BVAR?** A Bayes-i keretrendszer lehetővé teszi a prior hiedelmek beépítését, ami stabilizálja a becslést és figyelembe veszi a paraméterbizonytalanságot.
        - **Miért MSH?** A pénzügyi piacok volatilitása nem állandó. Az MSH modell képes megkülönböztetni a 'nyugodt' és 'pánik' piaci rezsimeket, így realisztikusabb kockázati előrejelzéseket ad.
        - **Eredmény:** Egy nagyszámú, 60 hónapos, 14 változós szcenáriókészlet, amely a jövőbeli piaci pályák lehetséges alakulását írja le.
        """,
        "step2_header": "2. Lépés: Nézetek beépítése (Entrópia Pooling)",
        "step2_text": """
        A BVAR-MSH modell 'objektív' szcenárióit finomhangoljuk szubjektív piaci várakozásainkkal (nézetekkel). Az Entrópia Pooling a szcenáriók valószínűségeit módosítja, hogy azok megfeleljenek a kényszereinknek, miközben a lehető legközelebb marad az eredeti eloszláshoz.
        
        **Konkrét nézeteink:**
        - **Arany (GLD):** Évesített átlaghozama `[2.9%, 3.3%]` közé essen.
        - **Rövid Állampapír (BIL):** A negatív havi hozam valószínűsége legfeljebb `5%` legyen.
        - **Államkötvények (Treasury):** Az összes államkötvény (BIL, SHY, IEF, TLT) évesített hozama `[3.8%, 4.1%]` közé essen (a zéró körüli term prémium nézetünket tükrözve).
        - **Relatív Teljesítmény:**
            - A részvények (SPY, IWM) átlaghozama legyen magasabb, mint a kötvényeké.
            - A részvények átlagos volatilitása is legyen magasabb.
            - A részvények átlaghozamának felső korlátja `8%`.
        - **Volatilitási Rangsor:** A kötvények volatilitása a lejáratukkal növekedjen (`σ(SHY) < σ(BIL) < σ(IEF) < σ(TLT)`).
        """,
        "step3_header": "3. Lépés: Dinamikus optimalizáció",
        "step3_text": """
        A nézetekkel súlyozott szcenáriókon egy többperiódusos portfólió-optimalizációt hajtunk végre.
        - **Célfüggvény:** Komplex, hibrid célfüggvényt használunk, ami a terminális vagyon maximalizálása és farokkockázatának (cCVaR) minimalizálása mellett bünteti a pálya menti túlzott havi kockázatot és a magas forgási sebességet.
        - **Kényszerek:**
            - **Hard (időben változó):** Portfólió-összetételi korlátok (pl. `min_treasury`), amelyek az idő előrehaladtával lazulnak. A **bázisportfóliók** konvex kombinációjának optimalizálása garantálja ezek betartását.
            - **Soft (nemlineáris):** A célfüggvény büntetőtagjai által kezelt elvárások (pl. havi kockázati sáv).
        - **Technológia:** A komplexitás miatt modern, gradiens-alapú optimalizálót (PyTorch/ADAM) használunk, GPU-gyorsítással.
        """,
        "historical_header": "Historikus teljesítmény (100-ról induló index)",
        "asset_select": "Válasszon eszközöket:",
        "frontier_xaxis": "Évesített kockázat (cCVaR)",
        "frontier_yaxis": "Évesített hozam",
        "allocation_title": "Dinamikus súlypálya",
        "constraints_title": "Aktuális kényszerek (hónap: {month})",
        "wealth_dist_title": "Terminális vagyon eloszlása",
        "wealth_dist_xaxis": "Terminális vagyon (1$ befektetésből)",
        "fanchart_title": "Portfólió értékének előrejelzése (fanchart)",
        "date_axis": "Dátum",
        "weight_axis": "Súly",
        "value_axis": "Érték",
        "indexed_value_axis": "Index (100-ról indul)",
        "diagnostics_summary_title": "Portfóliók diagnosztikai összefoglalója",
        "convergence_title": "Optimalizáció konvergenciája",
        "loss_type_select": "Válasszon veszteség-komponenst:",
        "data_error_title": "Hiba az adatfájlok betöltésekor!",
        "data_error_body": "A `streamlit_data` mappa vagy annak tartalma nem található. Kérjük, ellenőrizze a telepítést.",
    },
    "en": {
        "page_title": "Dynamic Portfolio Optimization Analyzer",
        "sidebar_header": "Controls",
        "portfolio_select": "Select a Portfolio:",
        "tab_intro": "Introduction",
        "tab_historical": "Historical Data",
        "tab_frontier": "Efficient Frontier",
        "tab_allocation": "Allocation & Constraints",
        "tab_distribution": "Future Distributions",
        "tab_diagnostics": "Diagnostics",
        "intro_header": "The Quantitative Workflow Overview",
        "step1_header": "Step 1: Market Model (BVAR-MSH)",
        "step1_text": """
        To forecast market dynamics, we use a Bayesian Vector Autoregressive model with Markov-Switching Heteroscedasticity (BVAR-MSH).
        - **Why BVAR?** The Bayesian framework allows incorporating prior beliefs, stabilizing estimation, and accounting for parameter uncertainty.
        - **Why MSH?** Financial market volatility is not constant. The MSH model can distinguish between 'calm' and 'panic' regimes, leading to more realistic risk forecasts.
        - **Result:** A large set of 60-month, 14-variable scenarios describing the potential evolution of future market paths.
        """,
        "step2_header": "Step 2: Incorporating Views (Entropy Pooling)",
        "step2_text": """
        We refine the 'objective' scenarios from the BVAR-MSH model with our subjective market expectations (views). Entropy Pooling adjusts scenario probabilities to satisfy our constraints while minimally deviating from the original distribution.
        
        **Our Specific Views:**
        - **Gold (GLD):** Annualized average return to fall between `[2.9%, 3.3%]`.
        - **Short-Term T-Bill (BIL):** Probability of a negative monthly return to be at most `5%`.
        - **Treasuries:** All treasury bonds (BIL, SHY, IEF, TLT) to have an annualized return between `[3.8%, 4.1%]` (reflecting our near-zero term premium view).
        - **Relative Performance:**
            - The average return of equities (SPY, IWM) should be higher than bonds.
            - The average volatility of equities should also be higher.
            - The upper cap for the average equity return is `8%`.
        - **Volatility Ranking:** Bond volatility should increase with duration (`σ(SHY) < σ(BIL) < σ(IEF) < σ(TLT)`).
        """,
        "step3_header": "Step 3: Dynamic Optimization",
        "step3_text": """
        We perform a multi-period portfolio optimization on the view-weighted scenarios.
        - **Objective Function:** A complex, hybrid objective that maximizes terminal wealth and minimizes its tail risk (cCVaR), while also penalizing excessive path-dependent monthly risk and high turnover.
        - **Constraints:**
            - **Hard (Time-Varying):** Portfolio composition limits (e.g., `min_treasury`) that relax over time, guaranteed by optimizing a convex combination of **basis portfolios**.
            - **Soft (Non-Linear):** Expectations managed by penalty terms in the objective function (e.g., monthly risk bands).
        - **Technology:** We use a modern, gradient-based optimizer (PyTorch/ADAM) with GPU acceleration.
        """,
        "historical_header": "Historical Performance (Indexed to 100)",
        "asset_select": "Select Assets:",
        "frontier_xaxis": "Annualized Risk (cCVaR)",
        "frontier_yaxis": "Annualized Return",
        "allocation_title": "Dynamic Allocation Path",
        "constraints_title": "Active Constraints (Month: {month})",
        "wealth_dist_title": "Terminal Wealth Distribution",
        "wealth_dist_xaxis": "Terminal Wealth (from 1$ investment)",
        "fanchart_title": "Portfolio Value Projection (Fanchart)",
        "date_axis": "Date",
        "weight_axis": "Weight",
        "value_axis": "Value",
        "indexed_value_axis": "Index (starts at 100)",
        "diagnostics_summary_title": "Portfolio Diagnostics Summary",
        "convergence_title": "Optimization Convergence",
        "loss_type_select": "Select Loss Component:",
        "data_error_title": "Error Loading Data Files!",
        "data_error_body": "The `streamlit_data` directory or its contents were not found. Please check your installation.",
    }
}

# --- Minimál CSS ---
st.markdown("""
<style>
    h1 { text-align: center; font-weight: bold; }
    .stTabs [data-baseweb="tab"] { font-size: 1.05rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- Adatbázis Kapcsolat ---
@st.cache_resource
def get_db_connection():
    data_dir = Path(__file__).parent / "streamlit_data"
    db_path = str(data_dir / "database.duckdb")
    if not data_dir.exists() or not Path(db_path).exists():
        st.error(TRANSLATIONS['en']['data_error_body'], icon="🚨")
        return None
    
    con = duckdb.connect(database=db_path, read_only=True)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW fact_bvar_scenarios AS
        SELECT * FROM read_parquet('{data_dir}/fact_bvar_scenarios/**/*.parquet', hive_partitioning=1);
    """)
    return con

con = get_db_connection()

@st.cache_data
def run_query(query, **params):
    if not con: return pd.DataFrame()
    return con.execute(query, list(params.values())).fetchdf()

@st.cache_data
def get_color_map():
    """Konzisztens színleképezés az összes eszközhöz."""
    all_assets_df = run_query("SELECT name FROM read_parquet('streamlit_data/dim_assets.parquet') WHERE type = 'asset_return'")
    asset_names = sorted(all_assets_df['name'].unique())
    palette = px.colors.qualitative.Plotly
    color_map = {asset: palette[i % len(palette)] for i, asset in enumerate(asset_names)}
    return color_map

def apply_fillcolor_for_area(fig, color_map):
    """Area trace-eknél egységesíti a fillcolor-t a megadott színleképezéshez."""
    for tr in fig.data:
        if getattr(tr, "name", None) in color_map:
            tr.update(line=dict(color=color_map[tr.name], width=0),
                      fillcolor=color_map[tr.name],
                      opacity=1.0)

# --- App Törzse ---
if not con:
    st.stop()

# --- Nyelvválasztó és Logó a Sidebar-ban ---
logo_path = Path(__file__).parent / ".streamlit" / "optimization.png"
if logo_path.exists():
    st.sidebar.image(str(logo_path))

if 'lang' not in st.session_state:
    st.session_state.lang = "hu"
lang_options = {"Magyar": "hu", "English": "en"}
selected_lang_str = st.sidebar.radio("Nyelv / Language", options=list(lang_options.keys()), horizontal=True)
st.session_state.lang = lang_options[selected_lang_str]
t = TRANSLATIONS[st.session_state.lang]

# --- Oldal Címe és Sidebar Vezérlők ---
st.title(t['page_title'])
st.sidebar.header(t['sidebar_header'])

frontier_df = run_query("SELECT * FROM read_parquet('streamlit_data/fact_efficient_frontier.parquet')")

# --- Portfólió kiválasztás (stabil állapot) ---
if 'portfolio_id' not in st.session_state:
    st.session_state.portfolio_id = int(frontier_df['portfolio_id'].median())

def update_portfolio_id():
    st.session_state.portfolio_id = st.session_state.portfolio_slider_widget

selected_portfolio_id = st.sidebar.select_slider(
    t['portfolio_select'],
    options=sorted(frontier_df['portfolio_id'].unique()),
    value=st.session_state.portfolio_id,
    key='portfolio_slider_widget',
    on_change=update_portfolio_id
)

# --- Kontrolált "Tabs" (radio, vízszintesen), stabil fókusz ---
tab_keys = ["intro", "historical", "frontier", "allocation", "distribution", "diagnostics"]
tab_titles = [f"📑 {t[f'tab_{k}']}" for k in tab_keys]

if "active_tab_idx" not in st.session_state:
    st.session_state.active_tab_idx = 0

active_tab_title = st.radio(
    label="", options=tab_titles,
    index=st.session_state.active_tab_idx,
    horizontal=True
)
st.session_state.active_tab_idx = tab_titles.index(active_tab_title)
active_key = tab_keys[st.session_state.active_tab_idx]

# --- Bevezető ---
if active_key == "intro":
    st.header(t['intro_header'])
    st.markdown("---")
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.subheader(t['step1_header'], divider='blue')
        st.markdown(t['step1_text'])
    with col2:
        st.subheader(t['step2_header'], divider='green')
        st.markdown(t['step2_text'])
    with col3:
        st.subheader(t['step3_header'], divider='orange')
        st.markdown(t['step3_text'])

# --- Historikus adatok ---
if active_key == "historical":
    st.subheader(t['historical_header'])
    asset_types_df = run_query("SELECT symbol, name, type FROM read_parquet('streamlit_data/dim_assets.parquet')")
    symbol_to_name = dict(zip(asset_types_df['symbol'], asset_types_df['name']))
    name_to_symbol = {v: k for k, v in symbol_to_name.items()}
    symbol_to_type = dict(zip(asset_types_df['symbol'], asset_types_df['type']))

    default_assets = ['SPY', 'TLT', 'GLD', 'T10Y2Y']
    selected_asset_names = st.multiselect(
        t['asset_select'],
        options=sorted(asset_types_df['name'].unique()),
        default=[symbol_to_name.get(s, s) for s in default_assets]
    )
    if selected_asset_names:
        selected_symbols = [name_to_symbol[n] for n in selected_asset_names]
        performance_symbols = [s for s in selected_symbols if symbol_to_type.get(s) == 'asset_return']
        macro_symbols = [s for s in selected_symbols if symbol_to_type.get(s) != 'asset_return']

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if performance_symbols:
            perf_symbols_str = str(performance_symbols)[1:-1]
            hist_perf_df = run_query(f"""
                SELECT d.date, a.name, p.indexed_value
                FROM read_parquet('streamlit_data/fact_historical_performance.parquet') p
                JOIN read_parquet('streamlit_data/dim_assets.parquet') a ON p.symbol = a.symbol
                JOIN read_parquet('streamlit_data/dim_dates.parquet') d ON p.date = d.date
                WHERE p.symbol IN ({perf_symbols_str})
            """)
            for name in hist_perf_df['name'].unique():
                df_subset = hist_perf_df[hist_perf_df['name'] == name]
                fig.add_trace(go.Scatter(x=df_subset['date'], y=df_subset['indexed_value'], name=name), secondary_y=False)

        if macro_symbols:
            macro_symbols_str = str(macro_symbols)[1:-1]
            hist_macro_df = run_query(f"""
                SELECT d.date, a.name, m.value
                FROM read_parquet('streamlit_data/fact_historical_macro.parquet') m
                JOIN read_parquet('streamlit_data/dim_assets.parquet') a ON m.symbol = a.symbol
                JOIN read_parquet('streamlit_data/dim_dates.parquet') d ON m.date = d.date
                WHERE m.symbol IN ({macro_symbols_str})
            """)
            for name in hist_macro_df['name'].unique():
                df_subset = hist_macro_df[hist_macro_df['name'] == name]
                fig.add_trace(go.Scatter(x=df_subset['date'], y=df_subset['value'], name=name, line=dict(dash='dot')), secondary_y=True)

        fig.update_layout(legend_title_text="", xaxis_title=t['date_axis'])
        fig.update_yaxes(title_text=t['indexed_value_axis'], secondary_y=False)
        fig.update_yaxes(title_text="Érték / Value" if macro_symbols else "", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, width='stretch')

# --- Hatékony front ---
if active_key == "frontier":
    fig_frontier = px.line(
        frontier_df.sort_values("annualized_risk_ccvar"),
        x="annualized_risk_ccvar", y="annualized_return", markers=True,
        custom_data=['portfolio_id']
    ).add_scatter(
        x=frontier_df['annualized_risk_ccvar'], y=frontier_df['annualized_return'],
        mode='text', text=frontier_df['portfolio_id'], textposition='top center', showlegend=False
    )
    fig_frontier.update_layout(
        title="<b>" + t['tab_frontier'] + "</b>",
        xaxis_title=t['frontier_xaxis'], yaxis_title=t['frontier_yaxis'],
        xaxis_tickformat='.1%', yaxis_tickformat='.1%'
    )
    fig_frontier.update_traces(
        hovertemplate="<b>Portfolio %{customdata[0]}</b><br>Return: %{y:.2%}<br>Risk (cCVaR): %{x:.2%}<extra></extra>"
    )
    st.plotly_chart(fig_frontier, width='stretch')

# --- Allokáció és Kényszerek ---
if active_key == "allocation":
    st.subheader(f"{t['allocation_title']} P{selected_portfolio_id}")
    asset_color_map = get_color_map()
    col1, col2 = st.columns([3, 1])

    with col1:
        weights_df = run_query(
            "SELECT dd.date, da.name, pw.weight FROM read_parquet('streamlit_data/fact_portfolio_weights.parquet') AS pw "
            "JOIN read_parquet('streamlit_data/dim_assets.parquet') AS da ON pw.symbol = da.symbol "
            "JOIN read_parquet('streamlit_data/dim_dates.parquet') AS dd ON pw.timestep_index = dd.timestep_index "
            "WHERE pw.portfolio_id = ? AND dd.timestep_type = 'future'",
            portfolio_id=int(selected_portfolio_id)
        )
        assets_in_portfolio = weights_df[weights_df['weight'] > 0.001]['name'].unique()

        if not weights_df.empty:
            fig_weights = px.area(
                weights_df[weights_df['name'].isin(assets_in_portfolio)],
                x='date', y='weight', color='name',
                category_orders={"name": sorted(assets_in_portfolio)},
                color_discrete_map=asset_color_map
            )
            apply_fillcolor_for_area(fig_weights, asset_color_map)
            fig_weights.update_layout(
                yaxis_tickformat=".0%", legend_title_text="",
                xaxis_title=t['date_axis'], yaxis_title=t['weight_axis']
            )
            st.plotly_chart(fig_weights, width='stretch')
        else:
            st.warning("Nincs allokációs adat a kiválasztott portfólióhoz." if st.session_state.lang == 'hu'
                       else "No allocation data for the selected portfolio.")

    with col2:
        month = st.slider("Hónap / Month", 1, 60, 1)
        st.subheader(t['constraints_title'].format(month=month))
        if month <= 24:
            cs = {'min_treasury': 0.50, 'max_gld': 0.25, 'min_gld': 0.10, 'min_bil_shy': 0.10}
            st.markdown("##### Kényszer: `Szigorú` / `Strict`")
        elif month <= 48:
            cs = {'min_treasury': 0.45, 'max_gld': 0.30, 'min_gld': 0.08, 'min_bil_shy': 0.08}
            st.markdown("##### Kényszer: `Laza` / `Relaxed`")
        else:
            cs = {'min_treasury': 0.40, 'max_gld': 0.35, 'min_gld': 0.05, 'min_bil_shy': 0.05}
            st.markdown("##### Kényszer: `Nagyon laza` / `Very Relaxed`")
        st.markdown(f"- **Min. Treasury:** `{cs['min_treasury']:.0%}`")
        st.markdown(f"- **Min. Gold:** `{cs['min_gld']:.0%}`")
        st.markdown(f"- **Max. Gold:** `{cs['max_gld']:.0%}`")
        st.markdown(f"- **Min. Cash-like (BIL+SHY):** `{cs['min_bil_shy']:.0%}`")
        st.markdown(f"- **Max. egyedi eszköz súly:** `40%`")

    st.markdown("---")
    if 'weights_df' in locals() and not weights_df.empty:
        weights_for_month = weights_df[weights_df['date'] == weights_df['date'].unique()[month - 1]]
        weights_for_pie = weights_for_month[weights_for_month['weight'] > 0.001]
        if not weights_for_pie.empty:
            pie_title = (f"Allokáció a kiválasztott hónapban ({month})"
                         if st.session_state.lang == 'hu'
                         else f"Allocation in Selected Month ({month})")
            fig_pie = px.pie(
                weights_for_pie,
                names='name',
                values='weight',
                title=f"<b>{pie_title}</b>",
                color='name',
                category_orders={"name": sorted(assets_in_portfolio)},
                color_discrete_map=asset_color_map
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', sort=False)
            fig_pie.update_layout(showlegend=False, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, width='stretch')

# --- Jövőbeli eloszlások ---
if active_key == "distribution":
    st.subheader(f"{t['wealth_dist_title']} - P{selected_portfolio_id}")
    terminal_wealth_query = """
        WITH PortfolioReturns AS (
            SELECT s.scenario_id, SUM(s.return_value * w.weight) as portfolio_return
            FROM fact_bvar_scenarios AS s
            JOIN read_parquet('streamlit_data/fact_portfolio_weights.parquet') AS w
              ON s.timestep_index = w.timestep_index AND s.symbol = w.symbol
            WHERE w.portfolio_id = ?
            GROUP BY s.scenario_id, s.timestep_index
        ),
        CumulativeWealth AS (
            SELECT scenario_id, PRODUCT(1 + portfolio_return) as terminal_wealth
            FROM PortfolioReturns GROUP BY scenario_id
        )
        SELECT terminal_wealth FROM CumulativeWealth
    """
    terminal_wealth_df = run_query(terminal_wealth_query, portfolio_id=int(selected_portfolio_id))
    if not terminal_wealth_df.empty:
        #q99 = terminal_wealth_df['terminal_wealth'].quantile(0.99)
        plot_df = terminal_wealth_df#[terminal_wealth_df['terminal_wealth'] <= q99]
        fig_dist = px.histogram(plot_df, x="terminal_wealth", nbins=100, histnorm='probability density')
        fig_dist.update_layout(
            xaxis_title=t['wealth_dist_xaxis'],
            yaxis_title="Sűrűség" if st.session_state.lang == 'hu' else 'Density'
        )
        st.plotly_chart(fig_dist, width='stretch')
        #st.info(
        #    "Megjegyzés: Az ábra a jobb olvashatóság érdekében a terminális vagyonok felső 1%-át nem mutatja."
        #    if st.session_state.lang == 'hu'
        #    else "Note: For better readability, the top 1% of terminal wealth outcomes are not shown on this chart.",
        #    icon="ℹ️"
        #)

    st.subheader(f"{t['fanchart_title']} - P{selected_portfolio_id}")
    fanchart_query = """
        WITH PortfolioReturns AS (
            SELECT s.scenario_id, s.timestep_index, SUM(s.return_value * w.weight) as portfolio_return
            FROM fact_bvar_scenarios AS s
            JOIN read_parquet('streamlit_data/fact_portfolio_weights.parquet') AS w
              ON s.timestep_index = w.timestep_index AND s.symbol = w.symbol
            WHERE w.portfolio_id = ?
            GROUP BY s.scenario_id, s.timestep_index
        ),
        PathWealth AS (
             SELECT
                scenario_id, timestep_index,
                EXP(SUM(LN(GREATEST(1 + portfolio_return, 1e-9))) OVER (PARTITION BY scenario_id ORDER BY timestep_index)) as path_wealth
            FROM PortfolioReturns
        ),
        Quantiles AS (
            SELECT
                timestep_index,
                quantile_cont(path_wealth, 0.05) AS p05, quantile_cont(path_wealth, 0.25) AS p25,
                quantile_cont(path_wealth, 0.50) AS p50, quantile_cont(path_wealth, 0.75) AS p75,
                quantile_cont(path_wealth, 0.95) AS p95
            FROM PathWealth GROUP BY timestep_index
        )
        SELECT d.date, q.* FROM Quantiles q
        JOIN read_parquet('streamlit_data/dim_dates.parquet') d ON q.timestep_index = d.timestep_index
        WHERE d.timestep_type = 'future' ORDER BY d.date
    """
    fanchart_df = run_query(fanchart_query, portfolio_id=int(selected_portfolio_id))
    if not fanchart_df.empty:
        p05, p25, p50, p75, p95 = fanchart_df['p05'], fanchart_df['p25'], fanchart_df['p50'], fanchart_df['p75'], fanchart_df['p95']
        dates = fanchart_df['date']
        fig_fan = go.Figure([
            go.Scatter(x=dates, y=p95, mode='lines', line=dict(width=0), showlegend=False),
            go.Scatter(x=dates, y=p05, mode='lines', line=dict(width=0), fillcolor='rgba(31, 119, 180, 0.2)', fill='tonexty', name='90% Conf. Interval', hoverinfo='none'),
            go.Scatter(x=dates, y=p75, mode='lines', line=dict(width=0), showlegend=False),
            go.Scatter(x=dates, y=p25, mode='lines', line=dict(width=0), fillcolor='rgba(31, 119, 180, 0.4)', fill='tonexty', name='50% Conf. Interval', hoverinfo='none'),
            go.Scatter(x=dates, y=p50, mode='lines', line=dict(color='#1f77b4', width=3), name='Median (P50)')
        ])
        fig_fan.update_layout(
            yaxis_title=t['value_axis'], xaxis_title=t['date_axis'],
            legend_title_text="Confidence Interval", hovermode="x unified"
        )
        st.plotly_chart(fig_fan, width='stretch')

# --- Diagnosztika ---
if active_key == "diagnostics":
    st.subheader(t['diagnostics_summary_title'])
    diagnostics_df = run_query("SELECT * FROM read_parquet('streamlit_data/fact_diagnostics_summary.parquet')")
    if not diagnostics_df.empty:
        format_mapping = {
            "Evesitett_Hozam": "{:.2%}", "annualized_return": "{:.2%}",
            "Evesitett_Kockazat_cCVaR": "{:.2%}", "annualized_risk_ccvar": "{:.2%}",
            "Maximalis_Kenyszer_Sertes": "{:.2e}",
            "Atlagos_Havi_Forgas": "{:.2%}",
            "Atlagos_Havi_cCVaR": "{:.2%}",
            "Maximalis_Havi_cCVaR": "{:.2%}",
            "Atlagos_Koncentracio_HHI": "{:.3f}"
        }
        valid_formatters = {k: v for k, v in format_mapping.items() if k in diagnostics_df.columns}
        st.dataframe(diagnostics_df.style.format(valid_formatters), width='stretch')
    
    st.subheader(f"{t['convergence_title']} - P{selected_portfolio_id}")
    convergence_df = run_query(
        "SELECT * FROM read_parquet('streamlit_data/fact_convergence.parquet') WHERE portfolio_id = ?",
        portfolio_id=int(selected_portfolio_id)
    )
    if not convergence_df.empty:
        loss_types = sorted(convergence_df['loss_type'].unique())
        default_loss = 'total_loss' if 'total_loss' in loss_types else loss_types[0]
        selected_loss = st.selectbox(t['loss_type_select'], loss_types, index=loss_types.index(default_loss))
        fig_conv = px.line(
            convergence_df[convergence_df['loss_type'] == selected_loss],
            x="Epoch", y="value", title=f"{selected_loss}"
        )
        st.plotly_chart(fig_conv, width='stretch')
