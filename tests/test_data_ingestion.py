from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import pytest
from data.ingestion import DataIngestion


@pytest.fixture
def data_ingestion():
    return DataIngestion(start_date="2020-01-01", end_date="2020-01-31")


def test_clean_data_standardization_and_sorting(data_ingestion):
    """Test column standardization, date normalization, and sorting."""
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

    assert list(cleaned.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    assert list(cleaned.index) == [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-05"),
    ]


def test_clean_data_deduplication(data_ingestion):
    """Test deduplication keeping the first record."""
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
    """Test filtering of invalid OHLC rows."""
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    raw_df = pd.DataFrame(
        {
            "Open": [100.0, 100.0, -10.0, 100.0],
            "High": [105.0, 90.0, 105.0, 105.0],
            "Low": [98.0, 95.0, 95.0, 95.0],
            "Close": [102.0, 98.0, 100.0, 0.0],
            "Adj Close": [102.0, 98.0, 100.0, 0.0],
            "Volume": [1000, 1000, 1000, 1000],
        },
        index=dates,
    )

    cleaned = data_ingestion.clean_data(raw_df)

    assert len(cleaned) == 1
    assert cleaned.index[0] == pd.Timestamp("2020-01-02")


def test_clean_data_ffill_isolated_gap(data_ingestion):
    """Test forward-filling isolated single-day gaps."""
    dates = pd.to_datetime(["2020-01-06", "2020-01-08"])
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

    assert len(cleaned) == 3
    assert pd.Timestamp("2020-01-07") in cleaned.index
    assert cleaned.loc["2020-01-07", "Close"] == 102.0


def test_clean_data_multi_day_gap_not_ffilled(data_ingestion):
    """Test multi-day business gaps are not forward-filled."""
    dates = pd.to_datetime(["2020-01-06", "2020-01-09"])
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

    assert len(cleaned) == 2
    assert pd.Timestamp("2020-01-07") not in cleaned.index
    assert pd.Timestamp("2020-01-08") not in cleaned.index


def test_clean_data_corporate_action_adjustment(data_ingestion):
    """Test that OHLC prices are scaled proportionally when corporate actions occur."""
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    # 1:1 bonus issue scenario: bar 1 has raw Close 2000 but Adj Close 1000 (0.5 ratio)
    raw_df = pd.DataFrame(
        {
            "Open": [2800.0, 1400.0],
            "High": [2900.0, 1500.0],
            "Low": [2700.0, 1300.0],
            "Close": [2800.0, 1400.0],
            "Adj Close": [1400.0, 1400.0],  # Bar 1 raw Close 2800 -> Adj Close 1400
            "Volume": [1000, 2000],
        },
        index=dates,
    )

    cleaned = data_ingestion.clean_data(raw_df)

    # Bar 1 Open (2800) should be adjusted by 0.5 ratio -> 1400
    assert cleaned.loc["2020-01-02", "Open"] == pytest.approx(1400.0)
    assert cleaned.loc["2020-01-02", "Close"] == pytest.approx(1400.0)
    assert cleaned.loc["2020-01-03", "Close"] == pytest.approx(1400.0)


@patch("data.ingestion.yf.Ticker")
def test_fetch_data_mock(mock_ticker_cls, data_ingestion):
    """Test fetch_data with yfinance mock."""
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
