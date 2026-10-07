def main():
    """Run a minimal local workflow from config-driven data if available."""
    from pathlib import Path

    from config import LOCAL_STOCK_CLOSE_CSV, REBALANCE_FREQUENCY_DAYS, SPLIT_DATE
    from src.data.provider import load_price_matrix
    from src.pipeline import run_dynamic_workflow

    source = Path(LOCAL_STOCK_CLOSE_CSV)
    if not source.exists():
        print("VNQuant V0 scaffold ready.")
        print("Drop your price CSV at src/data/stock_data.csv to run the default workflow.")
        return

    prices = load_price_matrix(source=source)
    result = run_dynamic_workflow(
        prices,
        split_date=SPLIT_DATE,
        lookback_window=63,
        rebalance_freq=REBALANCE_FREQUENCY_DAYS,
        run_backtest=False,
    )

    print("Best model:", result.best_model)
    print(result.ranked_models[["model", "Final_Rank", "Sharpe_Ratio", "Annual_Return"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
