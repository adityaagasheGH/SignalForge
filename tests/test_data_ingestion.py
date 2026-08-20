"""
Unit tests for data/ingestion.py (DataIngestion class)
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import pytest
from data.ingestion import DataIngestion


@pytest.fixture
def data_ingestion():
    return DataIngestion(start_date="2020-01-01", end_date="2020-01-31")


def test_clean_data_standardization_and_sorting(data_ingestion):
    """
    Test standardizing column headers, timestamp normalization, and chronological sorting.
    """
    dates = pd.to_datetime(["2020-01-05", "2020-01-02", "2020-01-03"])
    raw_df = pd.DataFrame(
        {
            "open": [100.0, 102.0, 101.0],
            "high": [105.0, 106.0, 104.0],
            "low": [99.0, 100.0, 98.0],
            "close": [104.0, 103.0, 102.0],
            "adj close": [104.0, 103.0, 102.0],
            "volume": [1000, 1200, 1100],
        },
        index=dates,
    )

    cleaned = data_ingestion.clean_data(raw_df)

    # Verify column naming standardization
    assert list(cleaned.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    # Verify chronological sorting (ascending date index)
    assert list(cleaned.index) == [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-05"),
    ]


def test_clean_data_deduplication(data_ingestion):
    """
    Test that duplicate date timestamps are deduplicated deterministically (keeping first).
    """
    dates = pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03"])
    raw_df = pd.DataFrame(
        {
            "Open": [100.0, 999.0, 101.0],
            "High": [105.0, 999.0, 104.0],
            "Low": [99.0, 999.0, 98.0],
            "Close": [104.0, 999.0, 102.0],
            "Adj Close": [104.0, 999.0, 102.0],
            "Volume": [1000, 9999, 1100],
        },
        index=dates,
    )

    cleaned = data_ingestion.clean_data(raw_df)

    assert len(cleaned) == 2
    assert cleaned.loc["2020-01-02", "Open"] == 100.0


def test_clean_data_invalid_rows_filtered(data_ingestion):
    """
    Test filtering out invalid rows: Close <= 0, Open <= 0, High < Low.
    """
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    raw_df = pd.DataFrame(
        {
            "Open": [100.0, 100.0, -10.0, 100.0],
            "High": [105.0, 90.0, 105.0, 105.0],  # row 1: High < Low (90 < 95)
            "Low": [98.0, 95.0, 95.0, 95.0],
            "Close": [102.0, 98.0, 100.0, 0.0],  # row 3: Close == 0
            "Adj Close": [102.0, 98.0, 100.0, 0.0],
            "Volume": [1000, 1000, 1000, 1000],
        },
        index=dates,
    )

    cleaned = data_ingestion.clean_data(raw_df)

    # Only row 0 ("2020-01-02") is valid
    assert len(cleaned) == 1
    assert cleaned.index[0] == pd.Timestamp("2020-01-02")


def test_clean_data_ffill_isolated_gap(data_ingestion):
    """
    Test that an isolated single-day business gap is forward-filled.
    Mon (Jan 6), Wed (Jan 8) -> Tue (Jan 7) missing.
    """
    dates = pd.to_datetime(["2020-01-06", "2020-01-08"])  # Mon and Wed (Tue Jan 7 missing)
    raw_df = pd.DataFrame(
        {
            "Open": [100.0, 110.0],
            "High": [105.0, 115.0],
            "Low": [98.0, 108.0],
            "Close": [102.0, 112.0],
            "Adj Close": [102.0, 112.0],
            "Volume": [1000, 1200],
        },
        index=dates,
    )

    cleaned = data_ingestion.clean_data(raw_df)

    # Jan 7 (Tuesday) should be forward-filled from Jan 6
    assert len(cleaned) == 3
    assert pd.Timestamp("2020-01-07") in cleaned.index
    assert cleaned.loc["2020-01-07", "Close"] == 102.0


def test_clean_data_multi_day_gap_not_ffilled(data_ingestion):
    """
    Test that multi-day business gaps (>1 day) are NOT forward-filled.
    Mon (Jan 6), Thu (Jan 9) -> Tue & Wed missing (2 business days gap).
    """
    dates = pd.to_datetime(["2020-01-06", "2020-01-09"])  # Mon and Thu
    raw_df = pd.DataFrame(
        {
            "Open": [100.0, 110.0],
            "High": [105.0, 115.0],
            "Low": [98.0, 108.0],
            "Close": [102.0, 112.0],
            "Adj Close": [102.0, 112.0],
            "Volume": [1000, 1200],
        },
        index=dates,
    )

    cleaned = data_ingestion.clean_data(raw_df)

    # Multi-day gap should not be filled
    assert len(cleaned) == 2
    assert pd.Timestamp("2020-01-07") not in cleaned.index
    assert pd.Timestamp("2020-01-08") not in cleaned.index


@patch("data.ingestion.yf.Ticker")
def test_fetch_data_mock(mock_ticker_cls, data_ingestion):
    """
    Test fetch_data using a mocked yfinance Ticker.
    """
    mock_ticker = MagicMock()
    mock_df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [98.0],
            "Close": [102.0],
            "Adj Close": [102.0],
            "Volume": [1000],
        },
        index=pd.to_datetime(["2020-01-02"]),
    )
    mock_ticker.history.return_value = mock_df
    mock_ticker_cls.return_value = mock_ticker

    result = data_ingestion.fetch_data("RELIANCE.NS")

    mock_ticker_cls.assert_called_once_with("RELIANCE.NS")
    mock_ticker.history.assert_called_once_with(
        start="2020-01-01", end="2020-01-31", auto_adjust=False
    )
    assert len(result) == 1
    assert result.iloc[0]["Close"] == 102.0
