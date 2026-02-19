"""
app.py — SPX Volatility Surface Changes Dashboard
Streamlit app replicating professional vol surface analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pytz

from fetch_data import (
    fetch_full_snapshot, save_snapshot, load_snapshot,
    list_snapshots, get_prior_snapshot, generate_synthetic_prior,
    build_vol_surface, build_term_structure, build_skew,
)

# ─────────────────────────────────────
# Page Config
# ─────────────────────────────────────
st.set_page_config(
    page_title="SPX Vol Surface",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────
# Theme / CSS
# ─────────────────────────────────────
DARK_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Outfit:wght@300;400;600;700&display=swap');

    .stApp { background-color: #0a0e27; }
    header[data-testid="stHeader"] { background-color: #0a0e27; }
    [data-testid="stToolbar"] { display: none; }
    .block-container { padding-top: 1rem; }

    h1, h2, h3, h4 { color: #e0e6ff; font-family: 'Outfit', sans-serif; }
    p, span, div, label { color: #b0b8d4; }

    /* Title bar */
    .title-bar {
        background: linear-gradient(135deg, #0d1340 0%, #141a4a 100%);
        border: 1px solid #1e2a6e;
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .title-main {
        font-family: 'Outfit', sans-serif;
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 1px;
    }
    .title-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #6b75a8;
    }
    .spot-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
        color: #00e5ff;
    }
    .spot-change-up { color: #00e676; font-weight: 600; }
    .spot-change-down { color: #ff5252; font-weight: 600; }

    /* Cards */
    .chart-card {
        background: linear-gradient(180deg, #0d1340 0%, #0a0e27 100%);
        border: 1px solid #1e2a6e;
        border-radius: 8px;
        padding: 8px;
        margin-bottom: 12px;
    }
    .chart-title {
        font-family: 'Outfit', sans-serif;
        font-size: 14px;
        font-weight: 600;
        color: #8892c4;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 4px;
        padding-left: 4px;
    }

    /* Date badges */
    .date-badge {
        display: inline-block;
        background: rgba(30, 42, 110, 0.6);
        border: 1px solid #2a3a8e;
        border-radius: 4px;
        padding: 2px 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        margin-right: 6px;
    }
    .date-current { color: #00e5ff; border-color: #00e5ff40; }
    .date-prior { color: #ff9800; border-color: #ff980040; }

    /* Plotly chart backgrounds */
    .js-plotly-plot .plotly .main-svg { border-radius: 6px; }

    /* Streamlit overrides */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #00e5ff;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Outfit', sans-serif;
        color: #6b75a8;
    }
    .stSelectbox label, .stSlider label {
        font-family: 'Outfit', sans-serif;
        color: #8892c4;
    }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────
# Color Scheme
# ─────────────────────────────────────
CS = {
    'bg': '#0a0e27',
    'plot_bg': '#0d1340',
    'grid': '#1a2050',
    'text': '#b0b8d4',
    'cyan': '#00e5ff',
    'gold': '#ffd740',
    'green': '#00e676',
    'red': '#ff5252',
    'orange': '#ff9800',
    'purple': '#bb86fc',
    'magenta': '#e040fb',
    'blue': '#448aff',
    'border': '#1e2a6e',
}


# ─────────────────────────────────────
# Data Loading (cached)
# ─────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_live_snapshot():
    """Fetch live data from Barchart. Cached 5 min."""
    return fetch_full_snapshot(num_expiries=8)


def ensure_data():
    """
    Load current + prior snapshots.
    - Current: live fetch (cached) → also save to disk
    - Prior: most recent saved snapshot before today, or synthetic
    """
    today_str = datetime.now().strftime('%Y-%m-%d')

    # Try loading today's saved snapshot first
    current = load_snapshot(today_str)
    if current is None:
        with st.spinner("Fetching live data from Barchart..."):
            current = load_live_snapshot()
        if current is not None:
            save_snapshot(current)

    if current is None:
        return None, None

    # Prior: find most recent snapshot before today
    prior = get_prior_snapshot(today_str)

    # If no prior exists, generate synthetic
    if prior is None:
        prior = generate_synthetic_prior(current)
        if prior is not None:
            # Save synthetic so it persists
            prior['date'] = (datetime.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            save_snapshot(prior)

    return current, prior


# ─────────────────────────────────────
# Chart Builders
# ─────────────────────────────────────

def create_vol_surface_table(current, prior, strike_step=50):
    """
    Build the main vol surface table with changes.
    Returns: surface_df, changes_df, spot
    """
    surf_now, spot = build_vol_surface(current, strike_step=strike_step)
    surf_prior, _ = build_vol_surface(prior, strike_step=strike_step)

    if surf_now.empty:
        return pd.DataFrame(), pd.DataFrame(), 0

    # Align indices
    common_strikes = surf_now.index.intersection(surf_prior.index) if not surf_prior.empty else surf_now.index
    common_expiries = [c for c in surf_now.columns if c in surf_prior.columns] if not surf_prior.empty else []

    changes = pd.DataFrame(index=common_strikes)
    for exp in common_expiries:
        changes[exp] = surf_now.loc[common_strikes, exp] - surf_prior.loc[common_strikes, exp]

    return surf_now, changes, spot


def create_surface_heatmap(surface_df, changes_df, spot):
    """Create the combined vol surface + changes heatmap chart."""
    if surface_df.empty:
        return go.Figure()

    # Build display table matching screenshot layout
    expiries = list(surface_df.columns)

    strikes = surface_df.index.values
    pct_spot = (strikes / spot * 100).round(2)

    # Vol surface values (as percentages for display)
    vol_text = []
    vol_z = []
    for strike in strikes:
        row_text = []
        row_z = []
        for exp in expiries:
            iv = surface_df.loc[strike, exp] if strike in surface_df.index and exp in surface_df.columns else np.nan
            if pd.notna(iv):
                row_text.append(f"{iv*100:.2f}")
                row_z.append(iv * 100)
            else:
                row_text.append("")
                row_z.append(np.nan)
        vol_text.append(row_text)
        vol_z.append(row_z)

    fig = go.Figure(data=go.Heatmap(
        z=vol_z,
        x=[e[5:] for e in expiries],  # MM-DD format
        y=[f"{s:.0f}" for s in strikes],
        text=vol_text,
        texttemplate="%{text}",
        textfont=dict(size=9, color='white'),
        colorscale=[
            [0, '#1a237e'],
            [0.3, '#283593'],
            [0.5, '#3949ab'],
            [0.7, '#e65100'],
            [1.0, '#ff6d00'],
        ],
        colorbar=dict(
            title=dict(text='IV %', font=dict(color=CS['text'], size=10)),
            tickfont=dict(color=CS['text'], size=9),
        ),
        hovertemplate='Strike: %{y}<br>Expiry: %{x}<br>IV: %{text}%<extra></extra>',
    ))

    # ATM line
    atm_idx = np.argmin(np.abs(strikes - spot))
    fig.add_hline(
        y=atm_idx, line=dict(color=CS['gold'], width=2, dash='dash'),
        annotation_text=f"ATM {spot:.0f}",
        annotation_font=dict(color=CS['gold'], size=10),
    )

    fig.update_layout(
        template='plotly_dark',
        title=dict(text='VOL SURFACE', font=dict(color=CS['text'], size=13, family='Outfit')),
        paper_bgcolor=CS['bg'],
        plot_bgcolor=CS['plot_bg'],
        font=dict(color=CS['text'], size=10, family='JetBrains Mono'),
        xaxis=dict(title='Expiry', gridcolor=CS['grid'], tickfont=dict(size=9)),
        yaxis=dict(title='Strike', gridcolor=CS['grid'], tickfont=dict(size=9)),
        margin=dict(l=60, r=20, t=40, b=40),
        height=500,
    )
    return fig


def create_changes_heatmap(changes_df, spot):
    """Create vol changes heatmap (green = IV increase, red = decrease)."""
    if changes_df.empty:
        return go.Figure()

    strikes = changes_df.index.values
    expiries = list(changes_df.columns)

    z_vals = []
    text_vals = []
    for strike in strikes:
        row_z = []
        row_t = []
        for exp in expiries:
            val = changes_df.loc[strike, exp] if exp in changes_df.columns else np.nan
            if pd.notna(val):
                row_z.append(val * 100)  # Convert to percentage points
                row_t.append(f"{val*100:+.2f}")
            else:
                row_z.append(np.nan)
                row_t.append("")
        z_vals.append(row_z)
        text_vals.append(row_t)

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=[e[5:] for e in expiries],
        y=[f"{s:.0f}" for s in strikes],
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorscale=[
            [0, '#b71c1c'],
            [0.35, '#e53935'],
            [0.5, '#1a1a2e'],
            [0.65, '#00c853'],
            [1.0, '#1b5e20'],
        ],
        zmid=0,
        colorbar=dict(
            title=dict(text='Δ IV', font=dict(color=CS['text'], size=10)),
            tickfont=dict(color=CS['text'], size=9),
            ticksuffix='%',
        ),
        hovertemplate='Strike: %{y}<br>Expiry: %{x}<br>Change: %{text}%<extra></extra>',
    ))

    atm_idx = np.argmin(np.abs(strikes - spot))
    fig.add_hline(
        y=atm_idx, line=dict(color=CS['gold'], width=2, dash='dash'),
        annotation_text=f"ATM {spot:.0f}",
        annotation_font=dict(color=CS['gold'], size=10),
    )

    fig.update_layout(
        template='plotly_dark',
        title=dict(text='VOL CHANGES', font=dict(color=CS['text'], size=13, family='Outfit')),
        paper_bgcolor=CS['bg'],
        plot_bgcolor=CS['plot_bg'],
        font=dict(color=CS['text'], size=10, family='JetBrains Mono'),
        xaxis=dict(title='Expiry', gridcolor=CS['grid'], tickfont=dict(size=9)),
        yaxis=dict(title='Strike', gridcolor=CS['grid'], tickfont=dict(size=9)),
        margin=dict(l=60, r=20, t=40, b=40),
        height=500,
    )
    return fig


def create_3d_surface(surface_df, spot):
    """Create 3D vol surface plot."""
    if surface_df.empty:
        return go.Figure()

    strikes = surface_df.index.values
    expiries = list(surface_df.columns)
    z_data = surface_df.values * 100  # Convert to percentage

    # Expiry as numeric (days from now)
    today = datetime.now()
    exp_days = [(datetime.strptime(e, '%Y-%m-%d') - today).days for e in expiries]

    fig = go.Figure(data=[go.Surface(
        z=z_data,
        x=exp_days,
        y=strikes,
        colorscale=[
            [0, '#1a237e'],
            [0.25, '#0d47a1'],
            [0.5, '#ff8f00'],
            [0.75, '#ff6d00'],
            [1.0, '#e65100'],
        ],
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor='white', project_z=True),
        ),
        hovertemplate='DTE: %{x}d<br>Strike: $%{y:.0f}<br>IV: %{z:.2f}%<extra></extra>',
    )])

    fig.update_layout(
        template='plotly_dark',
        title=dict(text='3D VOL SURFACE', font=dict(color=CS['text'], size=13, family='Outfit')),
        paper_bgcolor=CS['bg'],
        font=dict(color=CS['text'], size=9, family='JetBrains Mono'),
        scene=dict(
            xaxis=dict(title='DTE', backgroundcolor=CS['plot_bg'], gridcolor=CS['grid']),
            yaxis=dict(title='Strike', backgroundcolor=CS['plot_bg'], gridcolor=CS['grid']),
            zaxis=dict(title='IV %', backgroundcolor=CS['plot_bg'], gridcolor=CS['grid']),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=450,
    )
    return fig


def create_fixed_strike_changes(changes_df, spot, expiry_idx=0):
    """Bar chart: IV changes per strike for a single expiry."""
    if changes_df.empty or len(changes_df.columns) <= expiry_idx:
        return go.Figure()

    exp = changes_df.columns[expiry_idx]
    data = changes_df[exp].dropna() * 100  # to percentage points
    strikes = data.index.values
    values = data.values

    colors = [CS['green'] if v >= 0 else CS['red'] for v in values]

    fig = go.Figure(data=go.Bar(
        x=[f"{s:.0f}" for s in strikes],
        y=values,
        marker_color=colors,
        opacity=0.85,
        hovertemplate='Strike: %{x}<br>Δ IV: %{y:+.2f}%<extra></extra>',
    ))

    fig.update_layout(
        template='plotly_dark',
        title=dict(
            text=f'FIXED STRIKE VOL CHANGES — {exp}',
            font=dict(color=CS['text'], size=13, family='Outfit'),
        ),
        paper_bgcolor=CS['bg'],
        plot_bgcolor=CS['plot_bg'],
        font=dict(color=CS['text'], size=9, family='JetBrains Mono'),
        xaxis=dict(title='Strike', gridcolor=CS['grid'], tickangle=-45, tickfont=dict(size=8)),
        yaxis=dict(title='IV Change (%)', gridcolor=CS['grid'], zeroline=True,
                   zerolinecolor=CS['text'], zerolinewidth=0.5),
        margin=dict(l=50, r=20, t=40, b=60),
        height=350,
    )
    return fig


def create_skew_chart(current, prior, expiry_idx=0):
    """Front month vol skew: IV vs Strike, current + prior + change bars."""
    skew_now, spot, exp = build_skew(current, expiry_idx=expiry_idx)
    skew_prior, _, exp_prior = build_skew(prior, expiry_idx=expiry_idx)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if not skew_now.empty:
        fig.add_trace(go.Scatter(
            x=skew_now['strikePrice'],
            y=skew_now['volatility'] * 100,
            mode='lines',
            name='Current',
            line=dict(color=CS['gold'], width=2.5),
        ), secondary_y=False)

    if not skew_prior.empty:
        fig.add_trace(go.Scatter(
            x=skew_prior['strikePrice'],
            y=skew_prior['volatility'] * 100,
            mode='lines',
            name='Prior',
            line=dict(color='#555577', width=1.5, dash='dot'),
        ), secondary_y=False)

    # Vol change bars (if both exist)
    if not skew_now.empty and not skew_prior.empty:
        merged = pd.merge(
            skew_now.rename(columns={'volatility': 'iv_now'}),
            skew_prior.rename(columns={'volatility': 'iv_prior'}),
            on='strikePrice', how='inner',
        )
        merged['change'] = (merged['iv_now'] - merged['iv_prior']) * 100
        colors = [CS['green'] if v >= 0 else CS['red'] for v in merged['change']]

        fig.add_trace(go.Bar(
            x=merged['strikePrice'],
            y=merged['change'],
            name='Vol Change',
            marker_color=colors,
            opacity=0.5,
        ), secondary_y=True)

    # Spot line
    if spot:
        fig.add_vline(x=spot, line=dict(color=CS['cyan'], width=1, dash='dash'))

    exp_label = exp if exp else ''
    fig.update_layout(
        template='plotly_dark',
        title=dict(
            text=f'FRONT MONTH VOL SKEW — {exp_label}',
            font=dict(color=CS['text'], size=13, family='Outfit'),
        ),
        paper_bgcolor=CS['bg'],
        plot_bgcolor=CS['plot_bg'],
        font=dict(color=CS['text'], size=9, family='JetBrains Mono'),
        xaxis=dict(title='Strike', gridcolor=CS['grid']),
        margin=dict(l=50, r=50, t=40, b=40),
        height=380,
        legend=dict(orientation='h', y=-0.15, font=dict(size=9)),
        showlegend=True,
    )
    fig.update_yaxes(title_text='IV %', gridcolor=CS['grid'], secondary_y=False)
    fig.update_yaxes(title_text='Δ IV %', gridcolor=CS['grid'], secondary_y=True,
                     zeroline=True, zerolinecolor=CS['text'])

    return fig


def create_term_structure(current, prior):
    """Term structure: ATM IV vs expiry date, current + prior."""
    ts_now = build_term_structure(current)
    ts_prior = build_term_structure(prior)

    fig = go.Figure()

    if not ts_now.empty:
        fig.add_trace(go.Scatter(
            x=ts_now['expiry'],
            y=ts_now['atm_iv'] * 100,
            mode='lines+markers',
            name='Current',
            line=dict(color=CS['gold'], width=2.5),
            marker=dict(size=6, color=CS['gold']),
        ))

    if not ts_prior.empty:
        fig.add_trace(go.Scatter(
            x=ts_prior['expiry'],
            y=ts_prior['atm_iv'] * 100,
            mode='lines+markers',
            name='Prior',
            line=dict(color='#555577', width=1.5, dash='dot'),
            marker=dict(size=5, color='#555577'),
        ))

    fig.update_layout(
        template='plotly_dark',
        title=dict(text='TERM STRUCTURE', font=dict(color=CS['text'], size=13, family='Outfit')),
        paper_bgcolor=CS['bg'],
        plot_bgcolor=CS['plot_bg'],
        font=dict(color=CS['text'], size=9, family='JetBrains Mono'),
        xaxis=dict(title='Expiry', gridcolor=CS['grid'], tickangle=-30, tickfont=dict(size=9)),
        yaxis=dict(title='ATM IV %', gridcolor=CS['grid']),
        margin=dict(l=50, r=20, t=40, b=60),
        height=380,
        legend=dict(orientation='h', y=-0.2, font=dict(size=9)),
    )
    return fig


# ─────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────

current, prior = ensure_data()

if current is None:
    st.error("❌ Failed to fetch data. Check your network and try again.")
    st.stop()

spot = current['spot']
prior_spot = prior['spot'] if prior else spot
spot_change_pct = (spot / prior_spot - 1) * 100 if prior_spot else 0

# ── Title Bar ──
change_class = "spot-change-up" if spot_change_pct >= 0 else "spot-change-down"
current_date = current.get('date', 'N/A')
prior_date = prior.get('date', 'N/A') if prior else 'N/A'

# Get ET timestamp
est = pytz.timezone('US/Eastern')
now_et = datetime.now(est)

st.markdown(f"""
<div class="title-bar">
    <div>
        <span class="title-main">SPX VOLATILITY SURFACE CHANGES</span><br>
        <span class="title-sub">
            <span class="date-badge date-current">CURRENT: {current_date}</span>
            <span class="date-badge date-prior">PRIOR: {prior_date}</span>
            &nbsp;&nbsp;{now_et.strftime('%H:%M ET')}
        </span>
    </div>
    <div>
        <span class="spot-badge">
            SPOT &nbsp; ${spot:,.2f} &nbsp;
            <span class="{change_class}">{spot_change_pct:+.2f}%</span>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Controls ──
ctrl_cols = st.columns([1, 1, 1, 3])
with ctrl_cols[0]:
    strike_step = st.selectbox("Strike Step", [25, 50, 100], index=1, key="sstep")
with ctrl_cols[1]:
    pct_range = st.selectbox("Range %", [8, 10, 12, 15], index=2, key="pctrange")
with ctrl_cols[2]:
    if st.button("🔄 Refresh", use_container_width=True):
        load_live_snapshot.clear()
        st.rerun()

# ── Build data ──
surface_df, changes_df, _ = create_vol_surface_table(current, prior, strike_step=strike_step)

# ── Row 1: Vol Surface + Changes Heatmaps ──
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
row1_l, row1_r = st.columns(2)

with row1_l:
    st.markdown('<div class="chart-title">VOL SURFACE</div>', unsafe_allow_html=True)
    fig_surface = create_surface_heatmap(surface_df, changes_df, spot)
    st.plotly_chart(fig_surface, use_container_width=True, theme=None)

with row1_r:
    st.markdown('<div class="chart-title">VOL CHANGES</div>', unsafe_allow_html=True)
    fig_changes = create_changes_heatmap(changes_df, spot)
    st.plotly_chart(fig_changes, use_container_width=True, theme=None)

st.markdown('</div>', unsafe_allow_html=True)

# ── Row 2: 3D Surface + Fixed Strike Changes ──
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
row2_l, row2_r = st.columns(2)

with row2_l:
    fig_3d = create_3d_surface(surface_df, spot)
    st.plotly_chart(fig_3d, use_container_width=True, theme=None)

with row2_r:
    # Expiry selector for fixed-strike changes
    exp_options = list(changes_df.columns) if not changes_df.empty else []
    if exp_options:
        selected_exp_idx = st.selectbox(
            "Expiry for Fixed Strike",
            range(len(exp_options)),
            format_func=lambda i: exp_options[i],
            key="fsc_exp",
        )
        fig_fsc = create_fixed_strike_changes(changes_df, spot, expiry_idx=selected_exp_idx)
    else:
        fig_fsc = go.Figure()
    st.plotly_chart(fig_fsc, use_container_width=True, theme=None)

st.markdown('</div>', unsafe_allow_html=True)

# ── Row 3: Skew + Term Structure ──
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
row3_l, row3_r = st.columns(2)

with row3_l:
    fig_skew = create_skew_chart(current, prior, expiry_idx=0)
    st.plotly_chart(fig_skew, use_container_width=True, theme=None)

with row3_r:
    fig_term = create_term_structure(current, prior)
    st.plotly_chart(fig_term, use_container_width=True, theme=None)

st.markdown('</div>', unsafe_allow_html=True)

# ── Raw Data Table (collapsible) ──
with st.expander("📊 Raw Vol Surface Data"):
    if not surface_df.empty:
        display = surface_df.copy()
        display.index = [f"${s:,.0f}" for s in display.index]
        display = display.map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
        st.dataframe(display, use_container_width=True, height=400)

with st.expander("📊 Vol Changes Data"):
    if not changes_df.empty:
        display = changes_df.copy()
        display.index = [f"${s:,.0f}" for s in display.index]
        display = display.map(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "")
        st.dataframe(display, use_container_width=True, height=400)

# ── Footer ──
snapshot_dates = list_snapshots()
st.caption(f"Snapshots on disk: {', '.join(snapshot_dates) if snapshot_dates else 'none'} | "
           f"Expiries loaded: {len(current.get('expiries', {}))} | "
           f"Data source: Barchart")
