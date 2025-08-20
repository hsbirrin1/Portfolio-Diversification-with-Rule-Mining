# Option 4 — Portfolio Diversification with Rule Mining (Buffett-style ratios)
# Companies: COST, XOM, UNH, GOOGL, BAC
# -------------------------------------------------------------------------
# What this script does
# 1) Pulls 10-K fundamentals from SEC EDGAR (companyconcept API)
# 2) Computes ratios per company:
#    ROE, ROA, Debt-to-Equity (D/E), Net Margin, ROIC,
#    FCF Margin (FCF/Revenue), and 5y EPS Growth (from SEC when available)
# 3) Builds a normalized (MinMax) Health Score in [0,1] 
# 4) Runs Apriori rule mining on factor flags
# 5) Allocates $100k, tilting toward strongest HealthScores, caps, exact rounding; saves & plots pie chart
# 6) COVID event study for GOOGL using Yahoo Chart API, t-tests for 5d/20d/30d
# 7) Exports CSVs of key tables for your report
# -------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, UTC
from pathlib import Path
import sys
import subprocess

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from scipy import stats

# -----------------------------
# Output directory (next to this .py)
# -----------------------------
OUTDIR = Path(__file__).resolve().parent
print(f"\n[INFO] Outputs will be saved to: {OUTDIR}\n")

# -----------------------------
# Config
# -----------------------------
UA = {"User-Agent": "hbirring@seattleu.edu (for academic use)"}

TICKERS = {
    "COST": "0000909832",   # Costco Wholesale Corp
    "XOM":  "0000034088",   # Exxon Mobil Corp
    "UNH":  "0000731766",   # UnitedHealth Group Inc
    "GOOGL":"0001652044",   # Alphabet Inc (Class A)
    "BAC":  "0000070858",   # Bank of America Corp
}

# Tag fallbacks (robust across filers)
TAGS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet", "Revenues", "Revenue",
        # finance-ish fallbacks for banks
        "TotalRevenueNetOfInterestExpense",
        "InterestAndNoninterestIncome",
    ],
    "net_income": ["NetIncomeLoss"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "StockholdersEquity"
    ],
    # Cash flow
    "ocf": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PurchaseOfPropertyAndEquipment",
        "CapitalExpenditures"
    ],
    # EPS (units vary)
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"],
    # ROIC pieces
    "ebit": ["OperatingIncomeLoss"],
    "pretax": ["IncomeBeforeIncomeTaxes"],
    "tax_exp": ["IncomeTaxExpenseBenefit"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    # Debt
    "debt_lt": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"],
    "debt_st": ["LongTermDebtCurrent", "ShortTermBorrowings", "DebtCurrent", "CommercialPaper"],
}

FORMS_10K = {"10-K"}

# -----------------------------
# SEC helpers
# -----------------------------

def get_company_concept(cik: str, tag: str, unit: str = "USD"):
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{str(cik).zfill(10)}/us-gaap/{tag}.json"
    r = requests.get(url, headers=UA, timeout=30)
    if r.status_code != 200:
        return []
    units = r.json().get("units", {})
    return units.get(unit, [])


def get_first_available_series(cik: str, tag_list, unit="USD") -> pd.Series:
    for tag in tag_list:
        rows = get_company_concept(cik, tag, unit=unit)
        if rows:
            # Keep 10-K only and valid end dates
            rows = [x for x in rows if x.get("form") in FORMS_10K and x.get("end")]
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["end"] = pd.to_datetime(df["end"])
            df = df.sort_values("end").drop_duplicates(subset=["end"], keep="last")
            return pd.Series(pd.to_numeric(df["val"], errors="coerce").values,
                             index=df["end"].values)
    return pd.Series(dtype=float)


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors='coerce')
    b = pd.to_numeric(b, errors='coerce')
    out = a.divide(b)
    return out.where(b != 0)


def compute_yearly_metrics(cik: str) -> pd.DataFrame:
    # Pull series
    s_revenue = get_first_available_series(cik, TAGS["revenue"])
    s_netinc  = get_first_available_series(cik, TAGS["net_income"])
    s_assets  = get_first_available_series(cik, TAGS["assets"])
    s_liab    = get_first_available_series(cik, TAGS["liabilities"])
    s_equity  = get_first_available_series(cik, TAGS["equity"])
    s_cash    = get_first_available_series(cik, TAGS["cash"])
    s_ocf     = get_first_available_series(cik, TAGS["ocf"])
    s_capex   = get_first_available_series(cik, TAGS["capex"])

    # EPS (try multiple units)
    s_eps = get_first_available_series(cik, TAGS["eps"], unit="USD/shares")
    if s_eps.empty:
        s_eps = get_first_available_series(cik, TAGS["eps"], unit="USD/share")
    if s_eps.empty:
        s_eps = get_first_available_series(cik, TAGS["eps"])  # USD fallback

    s_ebit    = get_first_available_series(cik, TAGS["ebit"])
    s_pretax  = get_first_available_series(cik, TAGS["pretax"])
    s_taxexp  = get_first_available_series(cik, TAGS["tax_exp"])

    s_debt_lt = get_first_available_series(cik, TAGS["debt_lt"])
    s_debt_st = get_first_available_series(cik, TAGS["debt_st"])

    # Index union of all time points we have
    idx = sorted(set().union(
        s_revenue.index, s_netinc.index, s_assets.index, s_liab.index,
        s_equity.index, s_cash.index, s_ocf.index, s_capex.index,
        s_eps.index, s_ebit.index, s_pretax.index, s_taxexp.index,
        s_debt_lt.index, s_debt_st.index
    ))
    if not idx:
        return pd.DataFrame()

    df = pd.DataFrame(index=idx)
    df["Revenue"]    = s_revenue.reindex(idx)
    df["NetIncome"]  = s_netinc.reindex(idx)
    df["Assets"]     = s_assets.reindex(idx)
    df["Liabilities"]= s_liab.reindex(idx)
    df["Equity"]     = s_equity.reindex(idx)
    df["Cash"]       = s_cash.reindex(idx)
    df["OCF"]        = s_ocf.reindex(idx)
    df["CapEx"]      = s_capex.reindex(idx)
    df["EPS"]        = s_eps.reindex(idx)
    df["EBIT"]       = s_ebit.reindex(idx)
    df["PreTax"]     = s_pretax.reindex(idx)
    df["TaxExp"]     = s_taxexp.reindex(idx)

    # Debt total (fallback to Liabilities if debt series missing)
    total_debt = pd.Series(0.0, index=df.index)
    if not s_debt_lt.empty:
        total_debt = total_debt.add(s_debt_lt.reindex(df.index).fillna(0.0), fill_value=0.0)
    if not s_debt_st.empty:
        total_debt = total_debt.add(s_debt_st.reindex(df.index).fillna(0.0), fill_value=0.0)
    if (total_debt.abs() < 1e-9).all() and "Liabilities" in df:
        total_debt = df["Liabilities"]
    df["TotalDebt"] = total_debt

    # Ratios
    df["NetMargin"] = safe_div(df["NetIncome"], df["Revenue"])
    df["ROE"] = safe_div(df["NetIncome"], df["Equity"])
    df["ROA"] = safe_div(df["NetIncome"], df["Assets"])
    df["DE"]  = safe_div(df["TotalDebt"], df["Equity"])

    # ROIC ≈ NOPAT / InvestedCapital
    tax_rate = safe_div(df["TaxExp"], df["PreTax"]).clip(lower=0.0, upper=0.50).fillna(0.21)
    nopat = df["EBIT"].fillna(df["PreTax"]) * (1.0 - tax_rate)
    invested_capital = (df["TotalDebt"].fillna(0.0) + df["Equity"].fillna(0.0)) - df["Cash"].fillna(0.0)
    invested_capital = invested_capital.replace(0.0, np.nan)
    df["ROIC"] = safe_div(nopat, invested_capital)

    # Free Cash Flow & margin
    df["FCF"] = df["OCF"] - df["CapEx"]
    df["FCFMargin"] = safe_div(df["FCF"], df["Revenue"])
    return df


def eps_cagr_from_sec(cik: str, years: int = 5) -> float:
    s = get_first_available_series(cik, TAGS["eps"], unit="USD/shares")
    if s.empty:
        s = get_first_available_series(cik, TAGS["eps"], unit="USD/share")
    if s.empty:
        s = get_first_available_series(cik, TAGS["eps"])  # USD fallback
    s = s.dropna().sort_index()
    if len(s) < 2:
        return np.nan
    end = s.iloc[-1]
    start_idx = max(0, len(s) - (years + 1))
    start = s.iloc[start_idx]
    n = (len(s) - 1 - start_idx)
    if start <= 0 or end <= 0 or n <= 0:
        if len(s) >= 2 and s.iloc[-2] != 0:
            return (end / s.iloc[-2]) - 1.0
        return np.nan
    return (end / start) ** (1.0 / n) - 1.0

# -----------------------------
# Collect latest snapshot per ticker + EPS growth
# -----------------------------
ratio_records = {}
for tkr, cik in TICKERS.items():
    print(f"Pulling SEC facts for {tkr}...")
    df = compute_yearly_metrics(cik).sort_index()
    if df.empty:
        print(f"  Warning: not enough data for {tkr}")
        continue

    core = ["ROE","ROA","DE","NetMargin","ROIC","FCFMargin"]
    usable = df[core].notna().any(axis=1)
    if not usable.any():
        print(f"  Warning: core ratios missing for {tkr}")
        continue
    last_dt = df.index[usable][-1]

    rec = {
        "AsOf": pd.to_datetime(last_dt).date(),
        "ROE": float(df.at[last_dt, "ROE"]) if pd.notna(df.at[last_dt, "ROE"]) else np.nan,
        "ROA": float(df.at[last_dt, "ROA"]) if pd.notna(df.at[last_dt, "ROA"]) else np.nan,
        "DE":  float(df.at[last_dt, "DE"]) if pd.notna(df.at[last_dt, "DE"]) else np.nan,
        "NetMargin": float(df.at[last_dt, "NetMargin"]) if pd.notna(df.at[last_dt, "NetMargin"]) else np.nan,
        "ROIC": float(df.at[last_dt, "ROIC"]) if pd.notna(df.at[last_dt, "ROIC"]) else np.nan,
        "FCFMargin": float(df.at[last_dt, "FCFMargin"]) if pd.notna(df.at[last_dt, "FCFMargin"]) else np.nan,
        "EPS_Growth_5y": float(eps_cagr_from_sec(cik)),
    }
    ratio_records[tkr] = rec

latest_df = pd.DataFrame(ratio_records).T
print("\nLatest ratios (raw):\n", latest_df)

# Optionally export raw ratios
raw_csv = OUTDIR / "latest_ratios_raw.csv"
latest_df.reset_index().rename(columns={"index":"Ticker"}).to_csv(raw_csv, index=False)
print(f"[SAVED] Raw ratios CSV → {raw_csv}")

# -----------------------------
# Health Score (MinMax across peers) — NaN tolerant
# -----------------------------
POS = ["ROE","ROA","NetMargin","ROIC","FCFMargin","EPS_Growth_5y"]

scaled_df = latest_df.copy()

# invert D/E (lower is better); mask exact zero as NaN (avoids fake 'perfect' leverage)
de_series = latest_df.get("DE").copy()
if de_series is not None:
    de_series = de_series.mask(de_series == 0)
scaled_df["DE_inv"] = np.where(de_series.notna(), 1.0 / (1.0 + de_series.clip(lower=0)), np.nan)

scaled_cols = []
for col in POS + ["DE_inv"]:
    s = scaled_df[col]
    col_s = col + "_S"
    if s.notna().sum() >= 2:
        vmin, vmax = s.min(skipna=True), s.max(skipna=True)
        denom = (vmax - vmin) if (vmax - vmin) != 0 else np.nan
        scaled_df[col_s] = (s - vmin) / denom
    elif s.notna().sum() == 1:
        scaled_df[col_s] = np.where(s.notna(), 1.0, np.nan)
    else:
        scaled_df[col_s] = np.nan
    scaled_cols.append(col_s)

scaled_df["HealthScore"] = scaled_df[scaled_cols].mean(axis=1, skipna=True)

def bucket(x):
    if pd.isna(x):
        return "NA"
    if x >= 2/3:
        return "High"
    if x >= 1/3:
        return "Medium"
    return "Low"

scaled_df["ScoreBucket"] = scaled_df["HealthScore"].apply(bucket)
print("\nScaled metrics + Health Score:\n", scaled_df[["AsOf","HealthScore","ScoreBucket"] + scaled_cols])

# Export scaled table
scaled_csv = OUTDIR / "scaled_metrics_healthscore.csv"
scaled_df.reset_index().rename(columns={"index":"Ticker"}).to_csv(scaled_csv, index=False)
print(f"[SAVED] Scaled metrics CSV → {scaled_csv}")

# -----------------------------
# Apriori — boolean flags & rules
# -----------------------------
flags = pd.DataFrame(index=scaled_df.index)
flags["High_ROE"]     = (scaled_df["ROE_S"] >= 2/3).fillna(False).astype(bool)
flags["High_ROA"]     = (scaled_df["ROA_S"] >= 2/3).fillna(False).astype(bool)
flags["High_Margin"]  = (scaled_df["NetMargin_S"] >= 2/3).fillna(False).astype(bool)
flags["High_ROIC"]    = (scaled_df["ROIC_S"] >= 2/3).fillna(False).astype(bool)
flags["Low_DE"]       = (scaled_df["DE_inv_S"] >= 2/3).fillna(False).astype(bool)
flags["Positive_EPS_Growth"] = (scaled_df["EPS_Growth_5y_S"] > 0.5).fillna(False).astype(bool)
flags["Positive_FCF"] = (scaled_df["FCFMargin_S"] > 0.5).fillna(False).astype(bool)
for b in ["High","Medium","Low"]:
    flags[f"Score_{b}"] = (scaled_df["ScoreBucket"] == b).fillna(False).astype(bool)

print("\nApriori input flags:\n", flags)

# Export flags
flags_csv = OUTDIR / "apriori_flags.csv"
flags.reset_index().rename(columns={"index":"Ticker"}).to_csv(flags_csv, index=False)
print(f"[SAVED] Apriori flags CSV → {flags_csv}")

itemsets = apriori(flags, min_support=0.4, use_colnames=True)
rules = association_rules(itemsets, metric="confidence", min_threshold=0.7)
if not rules.empty:
    rules = rules.sort_values(["confidence","lift"], ascending=False)
    print("\nAssociation Rules:\n", rules[["antecedents","consequents","support","confidence","lift"]].head(10))
    rules_csv = OUTDIR / "apriori_rules.csv"
    rules.to_csv(rules_csv, index=False)
    print(f"[SAVED] Apriori rules CSV → {rules_csv}")
else:
    print("\nNo strong association rules found at current thresholds.")

# -----------------------------
# $100k Portfolio Allocation — score-tilted (exact rounding) + Pie
# -----------------------------
TOTAL_INVEST = 100000
scores = scaled_df["HealthScore"].fillna(0)
N = len(scores)

if scores.sum() == 0 or N == 0:
    weights = pd.Series(1.0 / max(N, 1), index=scores.index)
else:
    equal_w = pd.Series(1.0 / N, index=scores.index)
    score_w = scores / scores.sum()
    alpha = 0.5  # 50% equal-weight, 50% score-weight
    weights = alpha * equal_w + (1 - alpha) * score_w
    # caps to avoid concentration
    weights = weights.clip(lower=0.10, upper=0.35)
    weights = weights / weights.sum()

# dollar allocations (rounded to dollars)
alloc_series = (weights * TOTAL_INVEST).round()

# exact-total rounding fix: adjust the largest weight by the residual
residual = TOTAL_INVEST - alloc_series.sum()
if residual != 0:
    top_name = weights.idxmax()
    alloc_series[top_name] = alloc_series[top_name] + residual

alloc_df = alloc_series.to_frame(name="USD")
alloc_df.loc["Total"] = alloc_df["USD"].sum()

print("\n=== FINAL $100k ALLOCATION (score-tilted, exact total) ===")
print(alloc_df)
print(f"Exact total: ${int(alloc_df.loc['Total','USD']):,}")

# Pie chart (save + show)
alloc_plot = alloc_df.drop(index=["Total"]) if "Total" in alloc_df.index else alloc_df.copy()
plt.figure(figsize=(7,7))
plt.pie(alloc_plot["USD"], labels=alloc_plot.index, autopct='%1.1f%%', startangle=90)
plt.title("Portfolio Allocation — $100k (Score-Tilted)")
plt.tight_layout()
pie_path = OUTDIR / "allocation_pie.png"
plt.savefig(pie_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"[SAVED] Allocation pie chart → {pie_path}")

# Save allocation table too
alloc_out = alloc_df.drop(index=["Total"]) if "Total" in alloc_df.index else alloc_df.copy()
alloc_csv = OUTDIR / "portfolio_allocation.csv"
alloc_out.reset_index().rename(columns={"index":"Ticker"}).to_csv(alloc_csv, index=False)
print(f"[SAVED] Portfolio allocation CSV → {alloc_csv}")

# Auto-open pie on macOS (optional)
try:
    if sys.platform == "darwin":
        subprocess.call(["open", str(pie_path)])
except Exception:
    pass

# -----------------------------
# Event Study for GOOGL (COVID) — Yahoo chart API (no yfinance)
# T-tests for 5d, 20d, 30d horizons (before vs after event)
# -----------------------------

def yahoo_prices(ticker: str, interval="1d", rng="120mo") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": interval, "range": rng}
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["chart"]["result"][0]
    ts = data["timestamp"]
    quote = data["indicators"]["quote"][0]
    df = pd.DataFrame({
        "Date": [datetime.fromtimestamp(t, UTC) for t in ts],  # timezone-aware UTC (no deprecation)
        "Close": quote["close"],
        "Volume": quote.get("volume"),
    }).dropna()
    return df.set_index("Date")


def event_ttests_google_covid_multi(event_date="2020-03-11"):
    """Compute forward-horizon returns (5d, 20d, 30d) for GOOGL and S&P 500,
    then run Welch t-tests (before vs after the event) for each horizon.
    Prints means and p-values. Also saves a price chart around the event."""
    try:
        g = yahoo_prices("GOOGL")
        m = yahoo_prices("^GSPC")
    except Exception as e:
        print(f"Yahoo chart API error: {e}. Skipping event study.")
        return

    # Align on common dates
    data = pd.DataFrame({
        "GOOGL_Close": g["Close"],
        "SPX_Close": m["Close"],
    }).dropna()

    # Build forward returns for requested horizons
    horizons = [5, 20, 30]
    for w in horizons:
        data[f"Return_{w}d"] = (data["GOOGL_Close"].shift(-w) / data["GOOGL_Close"]) - 1.0
        data[f"Market_return_{w}d"] = (data["SPX_Close"].shift(-w) / data["SPX_Close"]) - 1.0

    # Event date index (UTC-aware to match price index)
    event_dt = pd.to_datetime(event_date, utc=True)
    if event_dt < data.index.min() or event_dt > data.index.max():
        print("Event date outside price history; skipping event study.")
        return

    # For plotting context line
    d0 = data.index[data.index.searchsorted(event_dt)]

    # Run t-tests per horizon (compare last w returns before vs first w returns after)
    for w in horizons:
        dfw = data.dropna(subset=[f"Return_{w}d", f"Market_return_{w}d"])
        before = dfw.loc[dfw.index < event_dt, f"Return_{w}d"].tail(w)
        after  = dfw.loc[dfw.index > event_dt, f"Return_{w}d"].head(w)
        if len(before) < 2 or len(after) < 2:
            print(f"\nT-Test Results ({w}-day window): insufficient data.")
            continue

        t_stat, p_val = stats.ttest_ind(before, after, equal_var=False, nan_policy='omit')

        print(f"\nT-Test Results ({w}-day window, COVID event):")
        print(f"Event day used: {event_dt.date()}")
        print(f"Before mean return: {before.mean():.6f}")
        print(f"After mean return:  {after.mean():.6f}")
        print(f"T-statistic:        {t_stat:.4f}")
        print(f"P-value:            {p_val:.4f}")
        if p_val < 0.05:
            print("Statistically significant difference.")
        else:
            print("No statistically significant difference.")

    # Simple price chart with event line — save & (on macOS) auto-open
    try:
        plt.figure(figsize=(11,5))
        plt.plot(data.index, data["GOOGL_Close"], label="GOOGL Close")
        plt.axvline(d0, linestyle="--", label="COVID event (2020-03-11)")
        plt.title("GOOGL price around WHO COVID declaration")
        plt.xlabel("Date"); plt.ylabel("Price (USD)")
        plt.legend(); plt.grid(True); plt.tight_layout()
        price_path = OUTDIR / "googl_event_price.png"
        plt.savefig(price_path, dpi=200, bbox_inches="tight")
        plt.show()
        print(f"[SAVED] Event-study price chart → {price_path}")
        try:
            if sys.platform == "darwin":
                subprocess.call(["open", str(price_path)])
        except Exception:
            pass
    except Exception:
        pass

# Run multi-horizon event study
event_ttests_google_covid_multi("2020-03-11")
