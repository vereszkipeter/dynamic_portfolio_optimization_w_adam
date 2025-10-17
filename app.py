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

# --- Kétnyelvű Tartalom (BŐVÍTVE) ---
TRANSLATIONS = {
    "hu": {
        "page_title": "Dinamikus portfólió optimalizáció elemző",
        "sidebar_header": "Vezérlőpult",
        "portfolio_select": "Válassz egy \"hatékony\" portfóliót:",
        "tab_intro": "Bevezető",
        "tab_historical": "Historikus/projektált faktorok",
        "tab_frontier": "Hatékony front",
        "tab_allocation": "Allokáció és kényszerek",
        "tab_distribution": "Vagyoneloszlások",
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
        - **Volatilitási Rangsor:** A kötvények volatilitása a H/2-ben minimális (`σ(SHY) < σ(BIL) < σ(IEF) < σ(TLT)`).
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
        "show_pi": "Predikciós intervallumok mutatása",
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
        "stats_header": "Portfólió statisztikák",
        "monthly_stats": "Kiválasztott hónap",
        "total_path": "Teljes 60 hónapos pálya",
        "monthly_er": "Havi E[R]",
        "annualized_er": "Évesített E[R]",
        "monthly_ccvar": "Havi cCVaR",
        "annualized_ccvar": "Évesített cCVaR",
        "monthly_cvar": "Havi CVaR",
        "annualized_cvar": "Évesített CVaR",
        "value_col": "Érték",
    },
    "en": {
        "page_title": "Dynamic Portfolio Optimization Analyzer",
        "sidebar_header": "Controls",
        "portfolio_select": "Select an \"Efficient\" Portfolio:",
        "tab_intro": "Introduction",
        "tab_historical": "Historical and Projected Factors",
        "tab_frontier": "Efficient Frontier",
        "tab_allocation": "Allocation & Constraints",
        "tab_distribution": "Wealth Distribution",
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
        - **Volatility Ranking:** Bond volatility minima at H/2 (`σ(SHY) < σ(BIL) < σ(IEF) < σ(TLT)`).
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
        "show_pi": "Show Prediction Intervals",
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
        "stats_header": "Portfolio Statistics",
        "monthly_stats": "Selected Month",
        "total_path": "Full 60-Month Path",
        "monthly_er": "Monthly E[R]",
        "annualized_er": "Annualized E[R]",
        "monthly_ccvar": "Monthly cCVaR",
        "annualized_ccvar": "Annualized cCVaR",
        "monthly_cvar": "Monthly CVaR",
        "annualized_cvar": "Annualized CVaR",
        "value_col": "Value",
    }
}

# --- Minimál CSS ---
st.markdown("""
<style>
    h1 { text-align: center; font-weight: bold; }
    /* A label elrejtése a radio gomboknál */
    div[role="radiogroup"] > label {
        display: true;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_color_map():
    """Konzisztens színleképezés az összes eszközhöz."""
    # A _con-t a session_state-ből kell elérni a cache-elt függvényen belül is
    all_assets_df = run_query("SELECT name FROM dim_assets WHERE type = 'asset_return'")
    if all_assets_df.empty:
        return {}
    asset_names = sorted(all_assets_df['name'].unique())
    palette = px.colors.qualitative.Plotly
    return {asset: palette[i % len(palette)] for i, asset in enumerate(asset_names)}

def apply_fillcolor_for_area(fig, color_map):
    """
    Felülbírálja az area chart trace-ek színét és áttetszőségét a megadott
    színleképezés alapján a jobb olvashatóság érdekében.
    """
    for trace in fig.data:
        if trace.name in color_map:
            # A fillcolor-t a vonal színéhez igazítjuk és az opacity-t 1-re állítjuk
            trace.update(fillcolor=color_map[trace.name], opacity=1)


# --- Adatbázis Kapcsolat ---
@st.cache_resource
def get_db_connection():
    # A nyelvi fordítást itt még nem használhatjuk, ezért angolul írjuk ki a hibát
    data_dir = Path(__file__).parent / "streamlit_data"
    db_path = str(data_dir / "database.duckdb")
    if not data_dir.exists() or not Path(db_path).exists():
        st.error(TRANSLATIONS['en']['data_error_body'], icon="🚨")
        return None
    
    con = duckdb.connect(database=db_path, read_only=True)
    return con

# --- Adatlekérdező Függvény ---
@st.cache_data
def run_query(query, **params):
    # A kapcsolatot a session state-ből vesszük, hogy ne kelljen globális változót használni
    if '_con' not in st.session_state or st.session_state._con is None: return pd.DataFrame()
    return st.session_state._con.execute(query, list(params.values())).fetchdf()

# --- App Törzse ---
st.session_state._con = get_db_connection()
if not st.session_state._con:
    st.stop()

# --- Nyelvválasztó és Logó a Sidebar-ban ---
logo_path = Path(__file__).parent / ".streamlit" / "optimization.png"
if logo_path.exists(): st.sidebar.image(str(logo_path))

if 'lang' not in st.session_state: st.session_state.lang = "hu"
lang_options = {"Magyar": "hu", "English": "en"}
selected_lang_str = st.sidebar.radio("Nyelv / Language", options=list(lang_options.keys()), horizontal=True)
st.session_state.lang = lang_options[selected_lang_str]
t = TRANSLATIONS[st.session_state.lang]

# --- Oldal Címe és Sidebar Vezérlők ---
st.title(t['page_title'])
st.sidebar.header(t['sidebar_header'])
frontier_df = run_query("SELECT * FROM fact_efficient_frontier")

# --- Portfólió kiválasztás (stabil állapot) ---
st.session_state.portfolio_id = st.sidebar.select_slider(
    t['portfolio_select'],
    options=sorted(frontier_df['portfolio_id'].unique()),
    value=st.session_state.get('portfolio_id', int(frontier_df['portfolio_id'].median()))
)
selected_portfolio_id = st.session_state.portfolio_id

# --- Kontrolált "Tabs" (radio, vízszintesen), stabil fókusz ---
tab_keys = ["intro", "historical", "frontier", "allocation", "distribution", "diagnostics"]
tab_titles = [f"📑 {t[f'tab_{k}']}" for k in tab_keys]
active_tab_title = st.radio(
    label="Navigation", label_visibility="collapsed",
    options=tab_titles, horizontal=True, key='active_tab'
)
active_key = tab_keys[tab_titles.index(active_tab_title)]

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

# --- Historikus adatok (BŐVÍTVE) ---
if active_key == "historical":
    st.subheader(t['historical_header'])
    asset_types_df = run_query("SELECT symbol, name, type FROM dim_assets")
    symbol_to_name = dict(zip(asset_types_df['symbol'], asset_types_df['name']))
    name_to_symbol = {v: k for k, v in symbol_to_name.items()}
    symbol_to_type = dict(zip(asset_types_df['symbol'], asset_types_df['type']))

    c1, c2 = st.columns([3, 1])
    with c1:
        default_assets = ['SPY', 'TLT', 'GLD', 'T10Y2Y']
        selected_asset_names = st.multiselect(
            t['asset_select'],
            options=sorted(asset_types_df['name'].unique()),
            default=[symbol_to_name.get(s, s) for s in default_assets]
        )
    with c2:
        show_pi = st.checkbox(t['show_pi'], value=False)
    
    if selected_asset_names:
        selected_symbols = [name_to_symbol[n] for n in selected_asset_names]
        performance_symbols = [s for s in selected_symbols if symbol_to_type.get(s) == 'asset_return']
        macro_symbols = [s for s in selected_symbols if symbol_to_type.get(s) != 'asset_return']
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        if performance_symbols:
            hist_perf_df = run_query("SELECT d.date, a.name, p.indexed_value FROM fact_historical_performance p JOIN dim_assets a ON p.symbol = a.symbol JOIN dim_dates d ON p.date = d.date WHERE p.symbol IN (SELECT * FROM UNNEST(?))", symbols=performance_symbols)
            for name in hist_perf_df['name'].unique():
                df_subset = hist_perf_df[hist_perf_df['name'] == name]
                fig.add_trace(go.Scatter(x=df_subset['date'], y=df_subset['indexed_value'], name=name, mode='lines'), secondary_y=False)
            
            if show_pi:
                quantiles_df = run_query("SELECT d.date, a.name, q.* FROM fact_asset_forecast_quantiles q JOIN dim_assets a ON q.symbol = a.symbol JOIN dim_dates d ON q.timestep_index = d.timestep_index WHERE d.timestep_type = 'future' AND q.symbol IN (SELECT * FROM UNNEST(?))", symbols=performance_symbols)
                for name in quantiles_df['name'].unique():
                    df_subset = quantiles_df[quantiles_df['name'] == name]
                    fig.add_trace(go.Scatter(x=df_subset['date'], y=df_subset['p50'], name=f"{name} Median", line=dict(dash='dot')), secondary_y=False)
                    fig.add_trace(go.Scatter(x=df_subset['date'], y=df_subset['p95'], line=dict(width=0), showlegend=False), secondary_y=False)
                    fig.add_trace(go.Scatter(x=df_subset['date'], y=df_subset['p05'], fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', line=dict(width=0), showlegend=False, name=f"{name}_90pi"), secondary_y=False)

        if macro_symbols:
            hist_macro_df = run_query("SELECT d.date, a.name, m.value FROM fact_historical_macro m JOIN dim_assets a ON m.symbol = a.symbol JOIN dim_dates d ON m.date = d.date WHERE m.symbol IN (SELECT * FROM UNNEST(?))", symbols=macro_symbols)
            for name in hist_macro_df['name'].unique():
                df_subset = hist_macro_df[hist_macro_df['name'] == name]
                fig.add_trace(go.Scatter(x=df_subset['date'], y=df_subset['value'], name=name, line=dict(dash='dot')), secondary_y=True)

        fig.update_layout(legend_title_text="", xaxis_title=t['date_axis'])
        fig.update_yaxes(title_text=t['indexed_value_axis'], secondary_y=False)
        fig.update_yaxes(title_text=t.get('value_axis', 'Value') if macro_symbols else "", secondary_y=True, showgrid=False)
        st.plotly_chart(fig)#, width='stretch')

# --- Hatékony front ---
if active_key == "frontier":
    fig_frontier = px.line(frontier_df.sort_values("annualized_risk_ccvar"), x="annualized_risk_ccvar", y="annualized_return", markers=True, custom_data=['portfolio_id']).add_scatter(x=frontier_df['annualized_risk_ccvar'], y=frontier_df['annualized_return'], mode='text', text=frontier_df['portfolio_id'], textposition='top center', showlegend=False)
    fig_frontier.update_layout(title="<b>" + t['tab_frontier'] + "</b>", xaxis_title=t['frontier_xaxis'], yaxis_title=t['frontier_yaxis'], xaxis_tickformat='.1%', yaxis_tickformat='.1%')
    fig_frontier.update_traces(hovertemplate="<b>Portfolio %{customdata[0]}</b><br>Return: %{y:.2%}<br>Risk (cCVaR): %{x:.2%}<extra></extra>")
    st.plotly_chart(fig_frontier)#, width='stretch')

# --- Allokáció és Kényszerek (VÉGLEGES) ---
if active_key == "allocation":
    st.subheader(f"{t['allocation_title']} P{selected_portfolio_id}")
    asset_color_map = get_color_map()
    
    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        weights_df = run_query("SELECT dd.date, da.name, pw.weight FROM fact_portfolio_weights AS pw JOIN dim_assets AS da ON pw.symbol = da.symbol JOIN dim_dates AS dd ON pw.timestep_index = dd.timestep_index WHERE pw.portfolio_id = ? AND dd.timestep_type = 'future'", portfolio_id=int(selected_portfolio_id))
        assets_in_portfolio = weights_df[weights_df['weight'] > 0.001]['name'].unique()
        if not weights_df.empty:
            fig_weights = px.area(weights_df[weights_df['name'].isin(assets_in_portfolio)], x='date', y='weight', color='name', category_orders={"name": sorted(assets_in_portfolio)}, color_discrete_map=asset_color_map)
            apply_fillcolor_for_area(fig_weights, asset_color_map)
            fig_weights.update_layout(yaxis_tickformat=".0%", legend_title_text="", xaxis_title=t['date_axis'], yaxis_title=t['weight_axis'], legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="right", x=1))
            st.plotly_chart(fig_weights)#, width='stretch')
    
    with col2:
        st.subheader(t['stats_header'])
        month = st.slider(f"{t.get('monthly_stats', 'Month')}", 1, 60, 1, key="month_slider")

        monthly_stats_df = run_query("SELECT * FROM fact_monthly_portfolio_stats WHERE portfolio_id = ? AND timestep_index = ?", portfolio_id=int(selected_portfolio_id), timestep_index=int(month))
        total_stats_df = run_query("SELECT * FROM fact_diagnostics_summary WHERE portfolio_id = ?", portfolio_id=int(selected_portfolio_id))
        
        if not monthly_stats_df.empty and not total_stats_df.empty:
            stats = {
                (t['monthly_stats'], t['monthly_er']): f"{monthly_stats_df['monthly_er'].iloc[0]:.2%}",
                (t['monthly_stats'], t['annualized_er']): f"{(1 + monthly_stats_df['monthly_er'].iloc[0])**12 - 1:.2%}",
                (t['monthly_stats'], t['monthly_ccvar']): f"{monthly_stats_df['monthly_ccvar'].iloc[0]:.2%}",
                (t['monthly_stats'], t['annualized_ccvar']): f"{monthly_stats_df['monthly_ccvar'].iloc[0] * np.sqrt(12):.2%}",
                (t['monthly_stats'], t['monthly_cvar']): f"{monthly_stats_df['monthly_std_cvar'].iloc[0]:.2%}",
                (t['monthly_stats'], t['annualized_cvar']): f"{-(((1 + monthly_stats_df['monthly_er'].iloc[0])**12 - 1) - (monthly_stats_df['monthly_ccvar'].iloc[0] * np.sqrt(12))):.2%}",
                (t['total_path'], t['annualized_er']): f"{total_stats_df['Evesitett_Hozam'].iloc[0]:.2%}",
                (t['total_path'], t['annualized_ccvar']): f"{total_stats_df['Evesitett_Kockazat_cCVaR'].iloc[0]:.2%}",
                (t['total_path'], t['annualized_cvar']): f"{total_stats_df.get('Evesitett_Std_CVaR', [np.nan]).iloc[0]:.2%}",
            }
            stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=[t['value_col']])
            stats_df.index = pd.MultiIndex.from_tuples(stats_df.index)
            st.dataframe(stats_df)#, width='stretch')

    col3, col4 = st.columns([2, 1], gap="large")
    with col3:
        if 'weights_df' in locals() and not weights_df.empty:
            weights_for_month = weights_df[weights_df['date'] == weights_df['date'].unique()[month - 1]]
            weights_for_pie = weights_for_month[weights_for_month['weight'] > 0.001]
            if not weights_for_pie.empty:
                pie_title = f"{t['tab_allocation']} ({t.get('monthly_stats', 'Month')} {month})"
                fig_pie = px.pie(weights_for_pie, names='name', values='weight', title=f"<b>{pie_title}</b>", color='name', category_orders={"name": sorted(assets_in_portfolio)}, color_discrete_map=asset_color_map)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label', sort=False)
                fig_pie.update_layout(showlegend=False, margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig_pie)#, width='stretch')
    with col4:
        st.subheader(t['constraints_title'].format(month=month))
        cs_level = ("Szigorú" if st.session_state.lang == 'hu' else "Strict") if month <= 24 else (("Laza" if st.session_state.lang == 'hu' else "Relaxed") if month <= 48 else ("Nagyon laza" if st.session_state.lang == 'hu' else "Very Relaxed"))
        cs_values = {'min_treasury': 0.50, 'max_gld': 0.25, 'min_gld': 0.10, 'min_bil_shy': 0.10} if month <= 24 else ({'min_treasury': 0.45, 'max_gld': 0.30, 'min_gld': 0.08, 'min_bil_shy': 0.08} if month <= 48 else {'min_treasury': 0.40, 'max_gld': 0.35, 'min_gld': 0.05, 'min_bil_shy': 0.05})
        st.markdown(f"##### Kényszer: `{cs_level}`")
        st.markdown(f"- **Min. Treasury:** `{cs_values['min_treasury']:.0%}`")
        st.markdown(f"- **Min. Gold:** `{cs_values['min_gld']:.0%}`")
        st.markdown(f"- **Max. Gold:** `{cs_values['max_gld']:.0%}`")
        st.markdown(f"- **Min. Cash-like (BIL+SHY):** `{cs_values['min_bil_shy']:.0%}`")
        st.markdown(f"- **Max. egyedi eszköz súly:** `40%`")

# --- Jövőbeli eloszlások ---
if active_key == "distribution":
    st.subheader(f"{t['wealth_dist_title']} - P{selected_portfolio_id}")
    # A hosszú query-t használjuk a biztonság kedvéért
    terminal_wealth_df = run_query("""
        WITH PortfolioReturns AS (
            SELECT s.scenario_id, SUM(s.return_value * w.weight) as portfolio_return
            FROM fact_bvar_scenarios AS s
            JOIN fact_portfolio_weights AS w
              ON s.timestep_index = w.timestep_index AND s.symbol = w.symbol
            WHERE w.portfolio_id = ?
            GROUP BY s.scenario_id, s.timestep_index
        ),
        CumulativeWealth AS (
            SELECT scenario_id, PRODUCT(1 + portfolio_return) as terminal_wealth
            FROM PortfolioReturns GROUP BY scenario_id
        )
        SELECT terminal_wealth FROM CumulativeWealth
    """, portfolio_id=int(selected_portfolio_id))
    if not terminal_wealth_df.empty:
        fig_dist = px.histogram(terminal_wealth_df, x="terminal_wealth", nbins=60, histnorm='probability density', opacity=0.4)
        fig_dist.update_traces(marker_line_color='#1f77b4', marker_line_width=0.75)
        fig_dist.update_layout(xaxis_title=t['wealth_dist_xaxis'], yaxis_title=t.get('density', 'Density'))
        st.plotly_chart(fig_dist)#, width='stretch')

    st.subheader(f"{t['fanchart_title']} - P{selected_portfolio_id}")
    fanchart_df = run_query("""
        WITH PortfolioReturns AS (
            SELECT s.scenario_id, s.timestep_index, SUM(s.return_value * w.weight) as portfolio_return
            FROM fact_bvar_scenarios AS s
            JOIN fact_portfolio_weights AS w
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
        JOIN dim_dates d ON q.timestep_index = d.timestep_index
        WHERE d.timestep_type = 'future' ORDER BY d.date
    """, portfolio_id=int(selected_portfolio_id))
    if not fanchart_df.empty:
        dates = pd.to_datetime(fanchart_df['date'])
        fig_fan = go.Figure([
            go.Scatter(x=dates, y=fanchart_df['p95'], mode='lines', line=dict(width=0), showlegend=False),
            go.Scatter(x=dates, y=fanchart_df['p05'], mode='lines', line=dict(width=0), fillcolor='rgba(31, 119, 180, 0.2)', fill='tonexty', name='90% Conf. Interval', hoverinfo='none'),
            go.Scatter(x=dates, y=fanchart_df['p75'], mode='lines', line=dict(width=0), showlegend=False),
            go.Scatter(x=dates, y=fanchart_df['p25'], mode='lines', line=dict(width=0), fillcolor='rgba(31, 119, 180, 0.4)', fill='tonexty', name='50% Conf. Interval', hoverinfo='none'),
            go.Scatter(x=dates, y=fanchart_df['p50'], mode='lines', line=dict(color='#1f77b4', width=3), name='Median (P50)')
        ])
        fig_fan.update_layout(yaxis_title=t['value_axis'], xaxis_title=t['date_axis'], legend_title_text="Confidence Interval", hovermode="x unified")
        st.plotly_chart(fig_fan)#, width='stretch')

# --- Diagnosztika ---
if active_key == "diagnostics":
    st.subheader(t['diagnostics_summary_title'])
    diagnostics_df = run_query("SELECT * FROM fact_diagnostics_summary")
    if not diagnostics_df.empty:
        format_mapping = { "Evesitett_Hozam": "{:.2%}", "Evesitett_Kockazat_cCVaR": "{:.2%}", "Evesitett_Std_CVaR": "{:.2%}", "Maximalis_Kenyszer_Sertes": "{:.2e}", "Atlagos_Havi_Forgas": "{:.2%}", "Atlagos_Havi_cCVaR": "{:.2%}", "Maximalis_Havi_cCVaR": "{:.2%}", "Atlagos_Koncentracio_HHI": "{:.3f}" }
        st.dataframe(diagnostics_df.style.format(format_mapping))#, width='stretch')
    
    st.subheader(f"{t['convergence_title']} - P{selected_portfolio_id}")
    convergence_df = run_query("SELECT * FROM fact_convergence WHERE portfolio_id = ?", portfolio_id=int(selected_portfolio_id))
    if not convergence_df.empty:
        loss_types = sorted(convergence_df['loss_type'].unique())
        default_loss = 'total_loss' if 'total_loss' in loss_types else loss_types[0]
        selected_loss = st.selectbox(t['loss_type_select'], loss_types, index=loss_types.index(default_loss))
        fig_conv = px.line(convergence_df[convergence_df['loss_type'] == selected_loss], x="Epoch", y="value", title=f"{selected_loss}")
        st.plotly_chart(fig_conv)#, width='stretch')