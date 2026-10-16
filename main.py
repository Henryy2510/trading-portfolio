import argparse


def main():
    """Fetch prices from vnstock and run the configured dynamic workflow."""
    from pathlib import Path

    from config import (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        DEFAULT_TICKERS,
        END_DATE,
        FEES,
        INITIAL_CASH,
        REBALANCE_FREQUENCY_DAYS,
        SLIPPAGE,
        SPLIT_DATE,
        START_DATE,
    )
    from src.data.provider import load_price_matrix, vndataprovider
    from src.pipeline import run_dynamic_workflow

    parser = argparse.ArgumentParser(description="Run the VNQuant dynamic workflow.")
    parser.add_argument(
        "--csv",
        type=Path,
        help="Use a local close-price matrix CSV instead of downloading from vnstock.",
    )
    args = parser.parse_args()

    if args.csv is not None:
        prices = load_price_matrix(source=args.csv, tickers=DEFAULT_TICKERS)
        print(f"Loaded local prices from {args.csv}")
    else:
        provider = vndataprovider(
            raw_dir=RAW_DATA_DIR,
            processed_dir=PROCESSED_DATA_DIR,
        )
        prices = load_price_matrix(
            tickers=DEFAULT_TICKERS,
            start=START_DATE,
            end=END_DATE,
            provider=provider,
        )
        print(f"Downloaded {len(prices.columns)} tickers from vnstock")

    result = run_dynamic_workflow(
        prices,
        split_date=SPLIT_DATE,
        lookback_window=63,
        rebalance_freq=REBALANCE_FREQUENCY_DAYS,
        run_backtest=True,
        fees=FEES,
        slippage=SLIPPAGE,
        initial_cash=INITIAL_CASH,
    )

    print("Best model:", result.best_model)
    print(result.ranked_models[["model", "Final_Rank", "Sharpe_Ratio", "Annual_Return"]].head(3).to_string(index=False))
    if result.backtest_report is not None:
        print("Out-of-sample backtest:")
        for metric, value in result.backtest_report.items():
            print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
