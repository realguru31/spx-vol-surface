# SPX Volatility Surface Changes Dashboard

Real-time SPX options volatility surface analysis powered by Barchart data, deployed on Streamlit Community Cloud.

## Features

- **Vol Surface Heatmap** — IV across strikes × expiries
- **Vol Changes Heatmap** — Day-over-day IV changes (green = up, red = down)
- **3D Volatility Surface** — Interactive 3D surface plot
- **Fixed Strike Vol Changes** — Bar chart of IV changes per strike
- **Front Month Vol Skew** — Current vs prior skew with change overlay
- **Term Structure** — ATM IV across expiry dates

## Architecture

```
spx_vol_surface/
├── app.py                    # Streamlit dashboard
├── fetch_data.py             # Barchart data fetching + snapshot logic
├── save_daily_snapshot.py    # Script for GitHub Actions cron
├── requirements.txt
├── snapshots/                # Daily JSON snapshots (auto-committed by GH Actions)
│   └── YYYY-MM-DD.json
└── .github/workflows/
    └── daily_snapshot.yml    # Runs Mon-Fri at 4:05 PM ET
```

**Data flow:**
1. GitHub Actions runs daily after market close → fetches Barchart data → commits `snapshots/YYYY-MM-DD.json`
2. Streamlit app loads → reads today's live data + yesterday's saved snapshot → computes changes
3. First day: synthetic prior is auto-generated (slight IV perturbations) so charts render immediately

## Deployment Steps

### 1. Create GitHub Repository

```bash
# Create new repo on GitHub (e.g., "spx-vol-surface"), then:
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/spx-vol-surface.git
git push -u origin main
```

### 2. Enable GitHub Actions (daily snapshots)

The workflow file is already at `.github/workflows/daily_snapshot.yml`. You need to grant it write access:

1. Go to **GitHub repo → Settings → Actions → General**
2. Scroll to **Workflow permissions**
3. Select **"Read and write permissions"**
4. Check **"Allow GitHub Actions to create and approve pull requests"**
5. Click **Save**

**Test it manually:**
- Go to **Actions** tab → **Daily SPX Vol Snapshot** → **Run workflow** → **Run**
- Verify it creates a `snapshots/YYYY-MM-DD.json` file

**Schedule:** Runs automatically Mon-Fri at 4:05 PM ET (21:05 UTC standard time). During DST it runs at 5:05 PM ET — adjust the cron to `5 20 * * 1-5` during DST if precision matters.

### 3. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your GitHub repo: `YOUR_USERNAME/spx-vol-surface`
4. Branch: `main`
5. Main file path: `app.py`
6. Click **Deploy**

The app will install dependencies from `requirements.txt` automatically.

### 4. First Run

On first load:
- App fetches live data from Barchart
- No prior snapshot exists → generates synthetic prior (slightly perturbed IVs, spot shifted ~1.8%)
- All charts render with synthetic changes
- Next day: GitHub Actions saves real snapshot → app uses real prior data

### 5. (Optional) Run first snapshot manually

If you want real data immediately instead of synthetic:

```bash
# Run locally
pip install -r requirements.txt
python save_daily_snapshot.py
git add snapshots/
git commit -m "First snapshot"
git push
```

## Configuration

| Parameter | Where | Default | Description |
|-----------|-------|---------|-------------|
| `num_expiries` | `fetch_data.py` | 8 | Number of expiry chains to fetch |
| `strike_step` | App UI | 50 | Strike interval for surface grid |
| `pct_range` | App UI | 12% | Strike range around spot |
| Cron schedule | `.github/workflows/` | 21:05 UTC M-F | Snapshot timing |
| Cache TTL | `app.py` | 300s (5 min) | Live data cache duration |

## Snapshot Size

~400-500 KB per day (8 expiries × ~400 rows each). At 252 trading days/year that's ~100-125 MB/year — well within GitHub's limits.

## Notes

- **Barchart rate limits:** The app caches live data for 5 minutes. GitHub Actions runs only once daily.
- **IV format:** Barchart returns IV in decimal (0.1448 = 14.48%). Dashboard displays as percentage.
- **OTM convention:** Vol surface uses OTM puts below spot, OTM calls at/above spot.
- **Streamlit Cloud filesystem:** Resets on redeploy. Snapshots persist because they're committed to the Git repo.
