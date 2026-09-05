"""Vietnam market placeholders for V1+ extensions."""

from __future__ import annotations


class VietnamMarket:
    """Execution-rule placeholder for VN market."""

    def __init__(self, exchange: str = "HOSE") -> None:
        self.exchange = exchange

    def normalize_ticker(self, ticker: str) -> str:
        return str(ticker).upper().strip()

    def supports(self, _ticker: str) -> bool:
        return True

    def execution_constraints(self) -> dict:
        """Return known/placeholder constraints."""
        return {
            "lot_size": None,
            "floor_ceiling": None,
            "short_selling": False,
            "trading_hours": "HOSE/HNX/UPCoM regular sessions (placeholder)",
            "corporate_actions_placeholder": True,
        }

    def benchmark(self, _ticker: str = "VNINDEX"):
        """V1+: replace with a VNINDEX benchmark fetcher."""
        raise NotImplementedError(
            "Benchmark integration is planned in V1+. Use strategy-vs-cash in V0."
        )
