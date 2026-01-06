from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class Config:
    start_date: str = "2014-01-01"
    end_date: str = "2025-12-19"
    interval: str = "1d"

    # Ticker filtering thresholds:
    min_nonnull_fraction: float = 0.95   # keep tickers with >=95% non-missing prices
    max_start_lag_days: int = 90         # must start within ~90 trading days of start_date
    require_last_within_days: int = 10   # must end within ~10 trading days of end_date

    # Date filtering threshold:
    drop_any_missing_dates: bool = True  # drop any dates where any kept ticker is missing

    out_dir: str = "data_processed"


ETFS_50: List[str] = [
    "SPY","IVV","VTI","QQQ","DIA","IWM","IJR","MDY",
    "IWF","IWD","VUG","VTV","MTUM","QUAL","USMV","VLUE","DVY","VYM",
    "XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLU","XLB","VNQ",
    "VXUS","VEA","EFA","VWO","EEM","IEMG","EWJ","EWU","EWG","FXI",
    "BND","SHY","IEF","TLT","TIP","LQD","HYG","EMB","BKLN","MUB",
    "GLD",
]


def download_prices(tickers: List[str], cfg: Config) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=cfg.start_date,
        end=cfg.end_date,
        interval=cfg.interval,
        auto_adjust=False,
        actions=False,
        group_by="column",
        threads=True,
        progress=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            prices = df["Adj Close"].copy()
        elif "Close" in df.columns.get_level_values(0):
            # fallback if Adj Close missing
            prices = df["Close"].copy()
        else:
            raise RuntimeError("Could not find Adj Close or Close in yfinance response.")
    else:
        # single ticker case
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        prices = df[col].to_frame(name=tickers[0])

    prices.index = pd.to_datetime(prices.index)
    prices.sort_index(inplace=True)
    prices = prices.replace([np.inf, -np.inf], np.nan)
    return prices


def quality_report(prices: pd.DataFrame) -> pd.DataFrame:
    total_days = len(prices.index)
    rows = []
    for t in prices.columns:
        s = prices[t]
        rows.append({
            "ticker": t,
            "first_valid_date": s.first_valid_index(),
            "last_valid_date": s.last_valid_index(),
            "nonnull_days": int(s.notna().sum()),
            "total_days": int(total_days),
            "nonnull_fraction": float(s.notna().sum() / total_days) if total_days else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["nonnull_fraction", "ticker"], ascending=[False, True])


def filter_tickers(prices: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    rep = quality_report(prices)
    start = pd.to_datetime(cfg.start_date)
    end = pd.to_datetime(cfg.end_date)

    # crude calendar-day cushions to approximate trading days
    start_ok = rep["first_valid_date"].notna() & (rep["first_valid_date"] <= start + pd.Timedelta(days=cfg.max_start_lag_days * 2))
    end_ok = rep["last_valid_date"].notna() & (rep["last_valid_date"] >= end - pd.Timedelta(days=cfg.require_last_within_days * 3))
    frac_ok = rep["nonnull_fraction"] >= cfg.min_nonnull_fraction

    keep = rep.loc[start_ok & end_ok & frac_ok, "ticker"].tolist()
    dropped = sorted(set(prices.columns) - set(keep))

    return prices[keep].copy(), rep, dropped


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    r = np.log(prices / prices.shift(1))
    return r.dropna(how="all")


def main():
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)

    print(f"Downloading {len(ETFS_50)} ETFs from {cfg.start_date} to {cfg.end_date}...")
    prices = download_prices(ETFS_50, cfg)

    filtered_prices, rep, dropped = filter_tickers(prices, cfg)

    print("\n=== Ticker filtering ===")
    print(f"Downloaded columns: {prices.shape[1]}")
    print(f"Kept tickers: {filtered_prices.shape[1]}")
    if dropped:
        print(f"Dropped ({len(dropped)}): {', '.join(dropped)}")

    if cfg.drop_any_missing_dates:
        before = len(filtered_prices)
        filtered_prices = filtered_prices.dropna(how="any")
        after = len(filtered_prices)
        print(f"\nDropped dates to make complete panel: {before - after} (kept {after} dates)")

    rets = log_returns(filtered_prices)

    prices_path = os.path.join(cfg.out_dir, "prices_adjusted.csv")
    rets_path = os.path.join(cfg.out_dir, "returns_log.csv")
    rep_path = os.path.join(cfg.out_dir, "data_quality_report.csv")

    filtered_prices.to_csv(prices_path)
    rets.to_csv(rets_path)
    rep.to_csv(rep_path, index=False)

    print("\n=== Saved ===")
    print(prices_path)
    print(rets_path)
    print(rep_path)

    print("\nPrices head:")
    print(filtered_prices.head())
    print("\nReturns head:")
    print(rets.head())


if __name__ == "__main__":
    main()
