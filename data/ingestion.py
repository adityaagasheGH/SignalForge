"""
SignalForge - Data Ingestion & Data Hygiene Engine

This module implements the DataIngestion class responsible for fetching historical market
data via yfinance and applying SignalForge data hygiene rules per blueprint.md §2:
- Deduplication of dates (keeping canonical first entry)
- Filtering invalid rows (Close <= 0, Open <= 0, High < Low, Low <= 0)
- Forward-filling strictly isolated single-day business gaps
- Guaranteeing chronological order and zero look-ahead bias
"""

from typing import Optional
import datetime
import logging
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Ingests, validates, and cleans historical equity OHLCV market data.
    """

    DEFAULT_START_DATE = "2015-01-01"

    def __init__(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: Optional[str] = None,
    ):
        """
        Initialize DataIngestion instance with date bounds.

        :param start_date: ISO date string (YYYY-MM-DD) for start of history. Default: '2015-01-01'.
        :param end_date: ISO date string (YYYY-MM-DD) for end of history. Default: current date.
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.date.today().strftime("%Y-%m-%d")

    def fetch_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetches raw market data for a ticker from yfinance.

        :param ticker: Stock symbol (e.g. 'RELIANCE.NS', 'TCS.NS', 'AAPL')
        :return: DataFrame containing raw OHLCV price series
        """
        logger.info(
            f"Fetching yfinance data for ticker '{ticker}' ({self.start_date} to {self.end_date})"
        )
        try:
            yf_ticker = yf.Ticker(ticker)
            raw_df = yf_ticker.history(
                start=self.start_date, end=self.end_date, auto_adjust=False
            )
            if raw_df.empty:
                logger.warning(f"No data retrieved for ticker '{ticker}'")
                return pd.DataFrame()
            return raw_df
        except Exception as e:
            logger.error(f"Failed to fetch data for ticker '{ticker}': {e}")
            raise RuntimeError(f"Data fetch error for '{ticker}': {e}") from e

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies SignalForge data hygiene pipeline to raw OHLCV DataFrame:
        1. Standardizes index and column naming.
        2. Deduplicates date timestamps deterministically.
        3. Sorts index in ascending chronological order (guaranteeing no look-ahead bias).
        4. Drops invalid rows (Close <= 0, Open <= 0, High < Low, Low <= 0).
        5. Forward-fills isolated single-day gaps (gaps of exactly 1 missing business day).

        :param df: Raw OHLCV DataFrame
        :return: Cleaned OHLCV DataFrame
        """
        if df.empty:
            return df.copy()

        cleaned = df.copy()

        # Handle MultiIndex columns (e.g., if yfinance returns tuple headers)
        if isinstance(cleaned.columns, pd.MultiIndex):
            cleaned.columns = cleaned.columns.get_level_values(0)

        # 1. Standardize column names
        expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        col_map = {col.lower(): col for col in expected_cols}
        rename_dict = {
            c: col_map[c.lower()] for c in cleaned.columns if c.lower() in col_map
        }
        cleaned.rename(columns=rename_dict, inplace=True)

        # 2. Standardize Date Index
        if isinstance(cleaned.index, pd.DatetimeIndex):
            cleaned.index = cleaned.index.tz_localize(None).normalize()
        elif "Date" in cleaned.columns:
            cleaned["Date"] = pd.to_datetime(cleaned["Date"]).dt.tz_localize(None).dt.normalize()
            cleaned.set_index("Date", inplace=True)

        cleaned.index.name = "Date"

        # 3. Deduplicate dates deterministically (keep first record)
        cleaned = cleaned[~cleaned.index.duplicated(keep="first")]

        # 4. Sort chronologically (ascending - enforcing zero look-ahead bias in row sequence)
        cleaned = cleaned.sort_index(ascending=True)

        # 5. Remove invalid rows
        # Rules: Close > 0, Open > 0, Low > 0, High > 0, High >= Low
        valid_mask = (
            (cleaned["Close"] > 0)
            & (cleaned["Open"] > 0)
            & (cleaned["Low"] > 0)
            & (cleaned["High"] > 0)
            & (cleaned["High"] >= cleaned["Low"])
        )

        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.warning(f"Dropping {invalid_count} invalid OHLC price rows.")
            cleaned = cleaned[valid_mask]

        # 6. Forward-fill isolated single-day business gaps
        cleaned = self._ffill_isolated_gaps(cleaned)

        return cleaned

    def _ffill_isolated_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies and forward-fills ISOLATED single-day business gaps.
        An isolated gap is defined as a missing business day strictly surrounded
        by valid data bars on both sides. Multi-day gaps (>1 day) are NOT forward-filled.

        :param df: Cleaned DataFrame with DatetimeIndex
        :return: DataFrame with single-day gaps forward-filled
        """
        if len(df) < 2:
            return df

        # Generate a full business day date range across the dataset span
        bday_range = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq="B"
        )
        full_index = df.index.union(bday_range).sort_values()

        # Reindex to full business day grid while preserving all existing dates
        reindexed = df.reindex(full_index)
        reindexed.index.name = "Date"

        # Identify missing bars (where Close is NaN)
        is_missing = reindexed["Close"].isna()

        # Check neighbor valid state using shift
        prev_valid = ~is_missing.shift(1, fill_value=False)
        next_valid = ~is_missing.shift(-1, fill_value=False)

        # An isolated gap is missing AND surrounded by valid bars
        isolated_gap_mask = is_missing & prev_valid & next_valid

        # Forward fill values across all numeric columns
        ffilled_all = reindexed.ffill()

        # Keep original valid rows AND isolated gap rows (now populated with ffilled values)
        rows_to_keep_mask = (~is_missing) | isolated_gap_mask
        result = ffilled_all[rows_to_keep_mask].copy()

        return result

    def get_historical_data(self, ticker: str) -> pd.DataFrame:
        """
        End-to-end wrapper: fetches raw data for ticker and runs data cleaning pipeline.

        :param ticker: Stock symbol
        :return: Cleaned, validated, and look-ahead safe OHLCV DataFrame
        """
        raw_df = self.fetch_data(ticker)
        cleaned_df = self.clean_data(raw_df)
        return cleaned_df
