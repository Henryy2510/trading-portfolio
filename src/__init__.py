"""Institutional Quantitative Research & Portfolio Management (VNQuant)."""

from src.data import (
    FundamentalData,
    MarketData,
    RealisticSyntheticMarketGenerator,
    load_dataset,
)
from src.features import AlphaEvaluationReport, AlphaFactory
from src.risk import (
    CovarianceEstimator,
    EconometricAuditReport,
    EconometricTester,
    GJRGARCHRiskEngine,
    MacroRegimeModel,
    MarketRegime,
    RiskForecast,
)
from src.portfolio import AllocationResult, GoalBasedPortfolioOptimizer
from src.backtest import BacktestMetrics, InstitutionalBacktester
from src.visualization import (
    InvestorDashboardVisualizer,
    create_drawdown_garch_chart,
    create_factor_ic_chart,
    create_sector_allocation_chart,
    create_wealth_chart,
)

__all__ = [
    "MarketData",
    "FundamentalData",
    "RealisticSyntheticMarketGenerator",
    "load_dataset",
    "AlphaFactory",
    "AlphaEvaluationReport",
    "CovarianceEstimator",
    "EconometricAuditReport",
    "EconometricTester",
    "GJRGARCHRiskEngine",
    "MacroRegimeModel",
    "MarketRegime",
    "RiskForecast",
    "GoalBasedPortfolioOptimizer",
    "AllocationResult",
    "InstitutionalBacktester",
    "BacktestMetrics",
    "create_wealth_chart",
    "create_sector_allocation_chart",
    "create_drawdown_garch_chart",
    "create_factor_ic_chart",
    "InvestorDashboardVisualizer",
]
