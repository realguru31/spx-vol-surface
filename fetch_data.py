"""
fetch_data.py — Barchart SPX Options Chain Fetcher
Fetches full vol surface data (multi-expiry) from Barchart.
"""

import requests
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from urllib.parse import unquote


# ═══════════════════════════════════════
# Barchart Session
# ═══════════════════════════════════════

def get_barchart_session():
    """Establish Barchart session with XSRF token."""
    page_url = 'https://www.barchart.com/stocks/quotes/$SPX/volatility-greeks'
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.9',
        'cache-control': 'max-age=0',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    session = requests.Session()
    session.get(page_url, headers=headers)
    cookies = session.cookies.get_dict()
    xsrf = unquote(cookies.get('XSRF-TOKEN', ''))
    return session, xsrf, page_url


def fetch_chain(session, xsrf, referer, expiry_date):
    """Fetch single expiry options chain from Barchart API. Returns list of dicts."""
    api_url = 'https://www.barchart.com/proxies/core-api/v1/options/get'
    headers = {
        'accept': 'application/json',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.9',
        'referer': referer,
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'x-xsrf-token': xsrf,
    }
    params = {
        'baseSymbol': '$SPX',
        'fields': 'strikePrice,lastPrice,volatility,delta,gamma,theta,vega,rho,openInterest,volume,optionType,daysToExpiration,expirationDate,bidPrice,askPrice',
        'expirationDate': expiry_date,
        'meta': 'field.shortName,field.type,field.description',
        'raw': '1',
        'limit': '500',
    }

    try:
        r = session.get(api_url, headers=headers, params=params)
        if r.status_code != 200:
            return []

        data = r.json()
        raw_data = data.get('data', [])

        # Handle grouped response (dict with Call/Put keys)
        if isinstance(raw_data, dict):
            all_rows = []
            for group_rows in raw_data.values():
                if isinstance(group_rows, list):
                    for row in group_rows:
                        if isinstance(row, dict) and 'raw' in row:
                            all_rows.append(row['raw'])
                        elif isinstance(row, dict):
                            all_rows.append(row)
            return all_rows

        # Handle flat list response
        elif isinstance(raw_data, list):
            return [
                row['raw'] if isinstance(row, dict) and 'raw' in row else row
                for row in raw_data
                if isinstance(row, dict)
            ]

        return []

    except Exception as e:
        print(f"Error fetching {expiry_date}: {e}")
        return []


# ═══════════════════════════════════════
# Snapshot Builder
# ═══════════════════════════════════════

def fetch_full_snapshot(num_expiries=8):
    """
    Fetch complete SPX vol surface snapshot from Barchart.
    Returns dict with date, spot, timestamp, and expiry chains.
    """
    # Get spot price + expiry dates from yfinance (lightweight)
    ticker = yf.Ticker("^SPX")
    expiry_dates = list(ticker.options)
    hist = ticker.history(period="5d")

    if hist.empty or not expiry_dates:
        return None

    spot = float(hist['Close'].iloc[-1])

    # Include today's expiry onwards
    today_str = datetime.now().strftime('%Y-%m-%d')
    future_expiries = [e for e in expiry_dates if e >= today_str][:num_expiries]

    if not future_expiries:
        return None

    # Barchart session
    session, xsrf, referer = get_barchart_session()

    if not xsrf:
        print("Failed to get XSRF token")
        return None

    snapshot = {
        'date': today_str,
        'spot': spot,
        'timestamp': datetime.now().isoformat(),
        'expiries': {},
    }

    for exp in future_expiries:
        rows = fetch_chain(session, xsrf, referer, exp)
        if rows:
            snapshot['expiries'][exp] = rows
            print(f"  {exp}: {len(rows)} rows")

    if not snapshot['expiries']:
        return None

    return snapshot


# ═══════════════════════════════════════
# Snapshot I/O
# ═══════════════════════════════════════

def save_snapshot(snapshot, directory='snapshots'):
    """Save snapshot as JSON file: snapshots/YYYY-MM-DD.json"""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{snapshot['date']}.json")
    with open(filepath, 'w') as f:
        json.dump(snapshot, f, default=str)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"Saved: {filepath} ({size_kb:.1f} KB)")
    return filepath


def load_snapshot(date_str, directory='snapshots'):
    """Load snapshot JSON for a given date. Returns dict or None."""
    filepath = os.path.join(directory, f"{date_str}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


def list_snapshots(directory='snapshots'):
    """List available snapshot dates, sorted descending (newest first)."""
    if not os.path.exists(directory):
        return []
    files = [f.replace('.json', '') for f in os.listdir(directory) if f.endswith('.json')]
    return sorted(files, reverse=True)


def get_prior_snapshot(current_date_str, directory='snapshots'):
    """Get the most recent snapshot BEFORE current_date_str."""
    dates = list_snapshots(directory)
    for d in dates:
        if d < current_date_str:
            return load_snapshot(d, directory)
    return None


# ═══════════════════════════════════════
# Synthetic Prior (Day-1 bootstrap)
# ═══════════════════════════════════════

def generate_synthetic_prior(today_snapshot):
    """
    Create a synthetic 'prior' snapshot by perturbing today's data.
    Used for initial deployment when no historical data exists.
    """
    if today_snapshot is None:
        return None

    np.random.seed(42)
    spot = today_snapshot['spot']

    prior = {
        'date': 'synthetic_prior',
        'spot': round(spot * 0.9816, 2),  # ~1.84% lower (matches screenshot example)
        'timestamp': 'synthetic',
        'expiries': {},
    }

    for exp, rows in today_snapshot['expiries'].items():
        prior_rows = []
        for row in rows:
            new_row = row.copy()
            iv = row.get('volatility', 0)
            if iv and iv > 0:
                # Distance-weighted noise: bigger shift for far OTM
                distance = abs(row.get('strikePrice', spot) - spot) / spot
                noise = np.random.normal(0, 0.003 + distance * 0.015)
                new_row['volatility'] = round(max(0.01, iv + noise), 6)
            # Slightly different OI and volume
            oi = row.get('openInterest', 0)
            new_row['openInterest'] = max(0, int(oi * np.random.uniform(0.8, 1.2)))
            prior_rows.append(new_row)
        prior['expiries'][exp] = prior_rows

    return prior


# ═══════════════════════════════════════
# Vol Surface Builder
# ═══════════════════════════════════════

def build_vol_surface(snapshot, strike_step=50, pct_range=0.12):
    """
    Build vol surface DataFrame from snapshot.
    Rows = strikes, Columns = expiry dates, Values = IV (decimal).
    Uses OTM options: puts below spot, calls at/above spot.
    """
    if snapshot is None:
        return pd.DataFrame(), 0

    spot = snapshot['spot']
    min_strike = spot * (1 - pct_range)
    max_strike = spot * (1 + pct_range)

    surface = {}

    for exp, rows in snapshot['expiries'].items():
        df = pd.DataFrame(rows)
        if df.empty or 'strikePrice' not in df.columns:
            continue

        # Filter to range
        df = df[(df['strikePrice'] >= min_strike) & (df['strikePrice'] <= max_strike)]

        # Use OTM: puts below spot, calls at/above
        puts = df[(df['optionType'] == 'Put') & (df['strikePrice'] < spot)]
        calls = df[(df['optionType'] == 'Call') & (df['strikePrice'] >= spot)]
        otm = pd.concat([puts, calls])

        # Round strikes to step
        otm = otm.copy()
        otm['strike_rounded'] = (otm['strikePrice'] / strike_step).round() * strike_step

        # Average IV per rounded strike (in case of duplicates)
        iv_map = otm.groupby('strike_rounded')['volatility'].mean()
        surface[exp] = iv_map

    if not surface:
        return pd.DataFrame(), spot

    surface_df = pd.DataFrame(surface)
    surface_df.index.name = 'Strike'
    surface_df = surface_df.sort_index()

    return surface_df, spot


def build_term_structure(snapshot):
    """
    Build ATM term structure: expiry → ATM IV.
    ATM = strike closest to spot.
    """
    if snapshot is None:
        return pd.DataFrame()

    spot = snapshot['spot']
    records = []

    for exp, rows in snapshot['expiries'].items():
        df = pd.DataFrame(rows)
        if df.empty:
            continue

        calls = df[df['optionType'] == 'Call']
        if calls.empty:
            continue

        # Closest strike to spot
        atm_idx = (calls['strikePrice'] - spot).abs().idxmin()
        atm_row = calls.loc[atm_idx]

        records.append({
            'expiry': exp,
            'atm_iv': atm_row['volatility'],
            'strike': atm_row['strikePrice'],
            'dte': atm_row.get('daysToExpiration', 0),
        })

    return pd.DataFrame(records)


def build_skew(snapshot, expiry_idx=0, pct_range=0.10):
    """
    Build vol skew for a single expiry: strike → IV.
    Uses OTM puts below spot, OTM calls above spot.
    """
    if snapshot is None:
        return pd.DataFrame(), 0, ''

    spot = snapshot['spot']
    expiries = list(snapshot['expiries'].keys())

    if expiry_idx >= len(expiries):
        return pd.DataFrame(), spot, ''

    exp = expiries[expiry_idx]
    rows = snapshot['expiries'][exp]
    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(), spot, exp

    min_s = spot * (1 - pct_range)
    max_s = spot * (1 + pct_range)
    df = df[(df['strikePrice'] >= min_s) & (df['strikePrice'] <= max_s)]

    puts = df[(df['optionType'] == 'Put') & (df['strikePrice'] < spot)]
    calls = df[(df['optionType'] == 'Call') & (df['strikePrice'] >= spot)]
    otm = pd.concat([puts, calls]).sort_values('strikePrice')

    return otm[['strikePrice', 'volatility']].reset_index(drop=True), spot, exp
