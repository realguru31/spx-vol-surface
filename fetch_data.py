"""
fetch_data.py — Barchart SPX Options Chain Fetcher
Fetches full vol surface data (multi-expiry) from Barchart.
"""

import requests
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

def fetch_expiry_dates_from_barchart():
    """Get available SPX option expiry dates.
    SPX has Mon/Wed/Fri weekly expirations plus monthly (3rd Friday).
    Generate known schedule + scrape Barchart page for extra dates.
    """
    import re
    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d')

    # Generate SPX expiry dates: Mon(0), Wed(2), Fri(4) for next 60 days
    generated = []
    for i in range(60):
        d = today + timedelta(days=i)
        if d.weekday() in (0, 2, 4):  # Mon, Wed, Fri
            generated.append(d.strftime('%Y-%m-%d'))

    # Also try scraping Barchart page for any additional dates
    scraped = []
    try:
        page_url = 'https://www.barchart.com/stocks/quotes/$SPX/volatility-greeks'
        headers = {
            'accept': 'text/html,application/xhtml+xml',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }
        r = requests.get(page_url, headers=headers, timeout=15)
        dates = re.findall(r'(\d{4}-\d{2}-\d{2})', r.text)
        scraped = [d for d in set(dates) if d >= today_str]
        print(f"[EXPIRY] Scraped {len(scraped)} dates from Barchart HTML")
    except Exception as e:
        print(f"[EXPIRY] Barchart scrape failed: {e}")

    # Merge and deduplicate
    all_dates = sorted(set(generated + scraped))
    future = [d for d in all_dates if d >= today_str]
    print(f"[EXPIRY] {len(future)} candidate expiry dates (generated={len(generated)}, scraped={len(scraped)})")
    return future


def fetch_full_snapshot(num_expiries=8):
    """
    Fetch complete SPX vol surface snapshot from Barchart.
    Returns dict with date, spot, timestamp, and expiry chains.
    No yfinance dependency.
    """
    today_str = datetime.now().strftime('%Y-%m-%d')

    # Get spot from tvdatafeed
    spot = fetch_tv_spot()

    # Get expiry dates from Barchart
    expiry_dates = fetch_expiry_dates_from_barchart()
    future_expiries = [e for e in expiry_dates if e >= today_str][:num_expiries * 2]  # Try extra, some will be invalid

    if not future_expiries:
        print("[SNAPSHOT] No future expiries found")
        return None

    print(f"[SNAPSHOT] Trying {len(future_expiries)} expiry dates, target {num_expiries} chains")

    # Barchart session
    session, xsrf, referer = get_barchart_session()

    if not xsrf:
        print("Failed to get XSRF token")
        return None

    # If tvdatafeed spot failed, try to get from first chain
    if spot is None:
        print("tvdatafeed spot failed, will extract from chain data")

    snapshot = {
        'date': today_str,
        'spot': spot or 0,
        'timestamp': datetime.now().isoformat(),
        'expiries': {},
    }

    for exp in future_expiries:
        rows = fetch_chain(session, xsrf, referer, exp)
        if rows:
            snapshot['expiries'][exp] = rows
            print(f"  {exp}: {len(rows)} rows")
            # If spot is still missing, estimate from ATM strikes
            if snapshot['spot'] == 0 and rows:
                strikes = [r.get('strikePrice', 0) for r in rows if r.get('strikePrice')]
                if strikes:
                    snapshot['spot'] = round(np.median(strikes), 2)
            # Stop once we have enough
            if len(snapshot['expiries']) >= num_expiries:
                break
        else:
            print(f"  {exp}: EMPTY (no data returned)")

    if not snapshot['expiries']:
        print("[SNAPSHOT] No chains fetched successfully")
        return None

    print(f"[SNAPSHOT] Got {len(snapshot['expiries'])} expiries: {list(snapshot['expiries'].keys())}")
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


# ═══════════════════════════════════════
# TvDatafeed — Price Data
# ═══════════════════════════════════════

def get_tv_connection(secrets=None):
    """Get tvDatafeed connection. Uses st.secrets if available."""
    try:
        from tvDatafeed import TvDatafeed
        if secrets and secrets.get('tv_username') and secrets.get('tv_password'):
            return TvDatafeed(username=secrets['tv_username'], password=secrets['tv_password'])
        return TvDatafeed()
    except Exception as e:
        print(f"tvDatafeed connection error: {e}")
        return None


def fetch_price_data(tv=None, n_bars=84, secrets=None):
    """
    Fetch SPX 5-min candles from TradingView (OANDA:SPX500USD).
    7 hours × 12 bars/hour = 84 bars.
    Returns DataFrame with Open, High, Low, Close columns, index in ET.
    """
    if tv is None:
        tv = get_tv_connection(secrets)
    if tv is None:
        print("fetch_price_data: no tv connection")
        return pd.DataFrame()

    try:
        from tvDatafeed import Interval
        df = tv.get_hist(
            symbol='SPX500USD',
            exchange='OANDA',
            interval=Interval.in_5_minute,
            n_bars=n_bars,
        )
        print(f"fetch_price_data: got {type(df)}, empty={df is None or (hasattr(df, 'empty') and df.empty)}")
        if df is not None and not df.empty:
            df = df.rename(columns={
                'open': 'Open', 'high': 'High',
                'low': 'Low', 'close': 'Close', 'volume': 'Volume',
            })
            # Convert to ET
            import pytz
            et = pytz.timezone('US/Eastern')
            if df.index.tz is None:
                try:
                    df.index = df.index.tz_localize('UTC')
                except Exception:
                    pass
            try:
                df.index = df.index.tz_convert(et)
            except Exception:
                pass
            print(f"fetch_price_data: returning {len(df)} bars, cols={list(df.columns)}")
            return df
    except Exception as e:
        print(f"fetch_price_data ERROR: {e}")
    return pd.DataFrame()


def fetch_tv_spot(tv=None, secrets=None):
    """Get current SPX spot price from TradingView."""
    if tv is None:
        tv = get_tv_connection(secrets)
    if tv is None:
        return None

    try:
        from tvDatafeed import Interval
        df = tv.get_hist(
            symbol='SPX500USD',
            exchange='OANDA',
            interval=Interval.in_1_minute,
            n_bars=1,
        )
        if df is not None and not df.empty:
            return float(df['close'].iloc[-1])
    except Exception:
        pass
    return None


# ═══════════════════════════════════════
# IV Change Aggregation (for combined chart)
# ═══════════════════════════════════════

def compute_iv_changes(current, prior, max_dte=0, strike_step=5, pct_range=0.05):
    """
    Compute per-strike IV changes between current live data and prior snapshot.

    max_dte: 0 = today only (0DTE), 7 = next 7 days, etc.
    Aggregation: average IV per strike across selected expiries, then diff.

    Returns DataFrame with columns: strike, iv_now, iv_prior, iv_change
    """
    if current is None or prior is None:
        return pd.DataFrame()

    spot = current['spot']
    min_strike = spot * (1 - pct_range)
    max_strike = spot * (1 + pct_range)

    today_str = datetime.now().strftime('%Y-%m-%d')

    # Select expiries within DTE range
    def select_expiries(snapshot, dte_days):
        cutoff = (datetime.now() + timedelta(days=dte_days + 1)).strftime('%Y-%m-%d')
        return [e for e in snapshot['expiries'].keys() if e <= cutoff]

    if max_dte == 0:
        # 0DTE: only today's expiry
        current_expiries = [e for e in current['expiries'].keys() if e == today_str]
        # Prior: use same date if exists, else first expiry
        prior_expiries = [e for e in prior['expiries'].keys() if e == today_str]
        if not prior_expiries:
            # Fallback: use the first (nearest) expiry in prior
            prior_expiries = list(prior['expiries'].keys())[:1]
    else:
        current_expiries = select_expiries(current, max_dte)
        prior_expiries = select_expiries(prior, max_dte)

    if not current_expiries or not prior_expiries:
        # Fallback: use all available
        current_expiries = list(current['expiries'].keys())[:max(1, max_dte // 2)]
        prior_expiries = list(prior['expiries'].keys())[:max(1, max_dte // 2)]

    def aggregate_iv(snapshot, expiries, min_s, max_s, step):
        """Average IV per rounded strike across selected expiries, using OTM."""
        all_rows = []
        spot_val = snapshot['spot']
        for exp in expiries:
            if exp not in snapshot['expiries']:
                continue
            df = pd.DataFrame(snapshot['expiries'][exp])
            if df.empty:
                continue
            df = df[(df['strikePrice'] >= min_s) & (df['strikePrice'] <= max_s)]
            # OTM: puts below spot, calls at/above
            puts = df[(df['optionType'] == 'Put') & (df['strikePrice'] < spot_val)]
            calls = df[(df['optionType'] == 'Call') & (df['strikePrice'] >= spot_val)]
            otm = pd.concat([puts, calls])
            all_rows.append(otm)

        if not all_rows:
            return pd.Series(dtype=float)

        combined = pd.concat(all_rows)
        combined['strike_rounded'] = (combined['strikePrice'] / step).round() * step
        return combined.groupby('strike_rounded')['volatility'].mean()

    iv_now = aggregate_iv(current, current_expiries, min_strike, max_strike, strike_step)
    iv_prior = aggregate_iv(prior, prior_expiries, min_strike, max_strike, strike_step)

    if iv_now.empty or iv_prior.empty:
        return pd.DataFrame()

    # Align on common strikes
    common = iv_now.index.intersection(iv_prior.index)
    if len(common) == 0:
        return pd.DataFrame()

    result = pd.DataFrame({
        'strike': common,
        'iv_now': iv_now.loc[common].values,
        'iv_prior': iv_prior.loc[common].values,
    })
    result['iv_change'] = (result['iv_now'] - result['iv_prior']) * 100  # Vol points
    result = result.sort_values('strike').reset_index(drop=True)

    return result
