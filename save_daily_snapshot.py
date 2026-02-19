#!/usr/bin/env python3
"""
save_daily_snapshot.py — Run by GitHub Actions daily after market close.
Fetches SPX vol surface from Barchart and saves as snapshots/YYYY-MM-DD.json.
"""

import sys
from datetime import datetime
from fetch_data import fetch_full_snapshot, save_snapshot

def main():
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"Fetching SPX vol surface snapshot for {today}...")

    snapshot = fetch_full_snapshot(num_expiries=8)

    if snapshot is None:
        print("❌ Failed to fetch data!")
        sys.exit(1)

    filepath = save_snapshot(snapshot)
    n_expiries = len(snapshot.get('expiries', {}))
    total_rows = sum(len(rows) for rows in snapshot['expiries'].values())
    print(f"✅ Saved: {filepath}")
    print(f"   Spot: ${snapshot['spot']:.2f}")
    print(f"   Expiries: {n_expiries}")
    print(f"   Total rows: {total_rows}")

if __name__ == "__main__":
    main()
