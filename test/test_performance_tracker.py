from pathlib import Path

import pandas as pd
import pytest

from script.performance_tracker import PerformanceTracker


def read_local_database(file_name="base_dados.xlsx") -> pd.DataFrame:
    data_path = Path(__file__).parent.parent.resolve().joinpath(
        "data", file_name)
    data = pd.read_excel(data_path)
    return data


def get_returns_dataframe(raw_data):
    raw_data['Date'] = pd.to_datetime(raw_data['Date'])
    raw_data = raw_data.set_index('Date').sort_index()
    data = raw_data.pct_change()
    data = data.dropna()
    return data


class TestPerformanceTracker:

    @pytest.fixture
    def performance_tracker(self):
        data = read_local_database(file_name="base_dados.xlsx")
        returns_data = get_returns_dataframe(data)
        stock_data = returns_data["B3SA3"]
        market_data = returns_data.mean(axis=1)
        performance_tracker = PerformanceTracker(stock_data,
                                                 market_data,
                                                 annual_risk_free=0,
                                                 alpha_var=0.05,
                                                 period="weekly")
        return performance_tracker

    def test_annualized_return(self, performance_tracker):
        assert performance_tracker.annualized_return() == pytest.approx(
            10.540206952938934)

    def test_std_return(self, performance_tracker):
        assert performance_tracker.annualized_std_return() == pytest.approx(
            36.72333065917448)

    def test_portfolio_beta(self, performance_tracker):
        assert performance_tracker.portfolio_beta() == pytest.approx(
            0.9330059186787968)

    def test_sharpe(self, performance_tracker):
        assert performance_tracker.sharpe_ratio() == pytest.approx(
            0.28701663938822786)

    def test_modified_sharpe_ratio(self, performance_tracker):
        assert performance_tracker.modified_sharpe_ratio() == pytest.approx(
            0.007815648369479346)

    def test_treynor(self, performance_tracker):
        assert performance_tracker.treynor_ratio() == pytest.approx(
            11.297041896437937)

    def test_modigliani_ratio(self, performance_tracker):
        assert performance_tracker.modigliani_ratio() == pytest.approx(
            18.46904453359593)

    def test_max_drawdown(self, performance_tracker):
        assert performance_tracker.max_drawdown() == pytest.approx(
            -48.83045927949462)

    def test_value_at_risk(self, performance_tracker):
        assert performance_tracker.portfolio_value_at_risk() == pytest.approx(
            -8.038060257111454)

    def test_media_perda_esperada(self, performance_tracker):
        assert performance_tracker.portfolio_media_perda_esperada(
        ) == pytest.approx(-11.977865327382458)

    def test_expected_return_capm(self, performance_tracker):
        assert performance_tracker.expected_return_capm() == pytest.approx(
            8.996945801089486)

    def test_calculate_portfolio_alpha(self, performance_tracker):
        assert performance_tracker.calculate_portfolio_alpha(
        ) == pytest.approx(1.543261151849448)
