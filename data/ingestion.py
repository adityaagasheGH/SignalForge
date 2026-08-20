from typing import Optional
import datetime
import logging
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


class DataIngestion:
    """Ingests and cleans historical equity OHLCV data."""

    DEFAULT_START_DATE = "2015-01-01"

    def __init__(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: Optional[str] = None,
    ):
        self.start_date = start_date
        self.end_date = end_date or datetime.date.today().strftime("%Y-%m-%d")

    def fetch_data(self, ticker: str) -> pd.DataFrame:
        """Fetch raw market data from yfinance."""
        logger.info(f"Fetching data for '{ticker}' ({self.start_date} to {self.end_date})")
        try:
            yf_ticker = yf.Ticker(ticker)
            raw_df = yf_ticker.history(
                start=self.start_date, end=self.end_date, auto_adjust=False
            )
            if raw_df.empty:
                logger.warning(f"No data for '{ticker}'")
                return pd.DataFrame()
            return raw_df
        except Exception as e:
            logger.error(f"Failed fetching '{ticker}': {e}")
            raise RuntimeError(f"Fetch error for '{ticker}': {e}") from e

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw OHLCV DataFrame."""
        if df.empty:
            return df.copy()

        cleaned = df.copy()

        if isinstance(cleaned.columns, pd.MultiIndex):
            cleaned.columns = cleaned.columns.get_level_values(0)

        # Standardize column names
        expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        col_map = {col.lower(): col for col in expected_cols}
        rename_dict = {
            c: col_map[c.lower()] for c in cleaned.columns if c.lower() in col_map
        }
        cleaned.rename(columns=rename_dict, inplace=True)

        # Normalize Date Index
        if isinstance(cleaned.index, pd.DatetimeIndex):
            cleaned.index = cleaned.index.tz_localize(None).normalize()
        elif "Date" in cleaned.columns:
            cleaned["Date"] = pd.to_datetime(cleaned["Date"]).dt.tz_localize(None).dt.normalize()
            cleaned.set_index("Date", inplace=True)

        cleaned.index.name = "Date"

        # Deduplicate and sort chronologically
        cleaned = cleaned[~cleaned.index.duplicated(keep="first")]
        cleaned = cleaned.sort_index(ascending=True)

        # Drop invalid rows
        valid_mask = (
            (cleaned["Close"] > 0)
            & (cleaned["Open"] > 0)
            & (cleaned["Low"] > 0)
            & (cleaned["High"] > 0)
            & (cleaned["High"] >= cleaned["Low"])
        )

        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.warning(f"Dropping {invalid_count} invalid OHLC rows.")
            cleaned = cleaned[valid_mask]

        # Forward-fill isolated gaps
        cleaned = self._ffill_isolated_gaps(cleaned)

        return cleaned

    def _ffill_isolated_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill isolated single-day business gaps."""
        if len(df) < 2:
            return df

        bday_range = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq="B"
        )
        full_index = df.index.union(bday_range).sort_values()

        reindexed = df.reindex(full_index)
        reindexed.index.name = "Date"

        is_missing = reindexed["Close"].isna()

        prev_valid = ~is_missing.shift(1, fill_value=False)
        next_valid = ~is_missing.shift(-1, fill_value=False)

        isolated_gap_mask = is_missing & prev_valid & next_valid

        ffilled_all = reindexed.ffill()

        rows_to_keep_mask = (~is_missing) | isolated_gap_mask
        result = ffilled_all[rows_to_keep_mask].copy()

        return result

    def get_historical_data(self, ticker: str) -> pd.DataFrame:
        """Fetch and clean data for ticker."""
        raw_df = self.fetch_data(ticker)
        cleaned_df = self.clean_data(raw_df)
        return cleaned_df
