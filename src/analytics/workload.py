"""Timestamp-only workload forecasting, backtesting and surge detection."""

import numpy as np
import pandas as pd
from scipy.stats import poisson


class WorkloadForecaster:
    def __init__(self, tickets):
        dates = pd.to_datetime(tickets.created_at).dt.floor("D")
        if dates.empty:
            raise ValueError("At least one ticket is required")
        counts = dates.value_counts().sort_index()
        self.daily = counts.reindex(pd.date_range(counts.index.min(), counts.index.max(), freq="D"), fill_value=0).astype(float)

    @staticmethod
    def _forecast(history, horizon, method):
        dates = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        if method == "recent_mean":
            values = np.repeat(history.tail(28).mean(), horizon)
        elif method == "weekday_mean":
            recent = history.tail(84)
            weekday = recent.groupby(recent.index.dayofweek).mean()
            values = [weekday.get(day.dayofweek, recent.mean()) for day in dates]
        elif method == "adaptive_mean":
            values = np.repeat(history.ewm(span=14, adjust=False).mean().iloc[-1], horizon)
        else:
            raise ValueError("Unknown forecast method")
        return pd.Series(np.maximum(values, 0.01), index=dates)

    def analyze(self, horizon=7):
        if not 1 <= horizon <= 30:
            raise ValueError("Forecast horizon must be between 1 and 30 days")
        if len(self.daily) < 90:
            raise ValueError("At least 90 calendar days are required for a backtest")
        # Select only before the final 28-day backtest window.
        validation_end = len(self.daily)-28
        methods = ["recent_mean", "weekday_mean", "adaptive_mean"]
        validation = {}
        for method in methods:
            errors = []
            for end in range(max(28, validation_end-56), validation_end, 7):
                actual = self.daily.iloc[end:min(end+7, validation_end)]
                predicted = self._forecast(self.daily.iloc[:end], len(actual), method)
                errors.extend(np.abs(actual.to_numpy()-predicted.to_numpy()))
            validation[method] = float(np.mean(errors))
        selected = min(validation, key=validation.get)
        backtest_rows = []
        for end in range(validation_end, len(self.daily), 7):
            actual = self.daily.iloc[end:min(end+7, len(self.daily))]
            predicted = self._forecast(self.daily.iloc[:end], len(actual), selected)
            baseline = self._forecast(self.daily.iloc[:end], len(actual), "recent_mean")
            for date, observed, estimate, reference in zip(actual.index, actual, predicted, baseline):
                backtest_rows.append({"date": date, "actual": observed, "forecast": estimate, "baseline": reference})
        backtest = pd.DataFrame(backtest_rows)
        future = self._forecast(self.daily, horizon, selected)
        forecast = pd.DataFrame({"date": future.index, "expected_tickets": future.to_numpy()})
        forecast["lower_80"] = poisson.ppf(0.1, future.to_numpy()).astype(int)
        forecast["upper_80"] = poisson.ppf(0.9, future.to_numpy()).astype(int)
        prior_mean = self.daily.shift(1).rolling(28, min_periods=14).mean()
        prior_std = self.daily.shift(1).rolling(28, min_periods=14).std().clip(lower=1)
        anomaly_score = (self.daily-prior_mean)/prior_std
        anomalies = pd.DataFrame({"date": self.daily.index, "tickets": self.daily.to_numpy(), "prior_mean": prior_mean.to_numpy(), "surge_score": anomaly_score.to_numpy()})
        anomalies = anomalies[anomalies.surge_score >= 3].sort_values("date", ascending=False)
        return {
            "method": selected, "origin": self.daily.index[-1],
            "validation_mae": validation,
            "backtest_mae": float((backtest.actual-backtest.forecast).abs().mean()),
            "baseline_mae": float((backtest.actual-backtest.baseline).abs().mean()),
            "forecast": forecast, "backtest": backtest, "anomalies": anomalies,
            "daily": self.daily.rename("tickets").rename_axis("date").reset_index(),
            "limitations": "Poisson intervals assume independent counts and omit model uncertainty. Abrupt incident-driven surges can exceed them. Forecast dates follow the dataset snapshot, not the current date.",
        }
