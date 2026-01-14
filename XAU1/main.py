#!/usr/bin/env python3
"""
XAU1 Trading Bot - Main Entry Point
SMC + Order Flow Strategy for XAUUSDT

Usage:
    python main.py --mode [backtest|live] --config [config_path]
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from xau1.backtest.backtester import XAU1Backtester
from xau1.exchange.binance import BinanceClient
from xau1.utils.data import DataManager, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_data(config: dict, days: int = 365) -> str:
    """Download historical data from Binance"""
    logger.info(f"Downloading {days} days of XAUUSDT data...")
    
    try:
        # Initialize Binance client
        binance = BinanceClient()
        
        # Fetch data
        df = binance.fetch_recent_ohlcv(
            timeframe="15m",
            days=days
        )
        
        if df.empty:
            logger.error("No data downloaded")
            return ""
        
        # Save data
        data_manager = DataManager()
        filename = f"XAUUSDT_15m_{days}days.csv"
        
        if data_manager.save_csv(df, filename):
            logger.info(f"Data saved to data/{filename}")
            return filename
        else:
            logger.error("Failed to save data")
            return ""
            
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return ""


def run_backtest(config: dict, data_file: str) -> dict:
    """Run backtest on historical data"""
    logger.info("Running backtest...")
    
    try:
        # Load data
        data_manager = DataManager()
        df = data_manager.load_csv(data_file)
        
        if df is None or df.empty:
            logger.error("No data available for backtest")
            return {}
        
        # Initialize backtester
        backtest_config = load_config("src/xau1/config/backtest_params.yaml")
        strategy_config = load_config("src/xau1/config/strategy_params.yaml")
        
        backtester = XAU1Backtester(strategy_config, backtest_config)
        
        # Run backtest
        result = backtester.run_backtest(df)
        
        # Save results
        results_dir = Path("backtest_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trade history
        trades_df = backtester.get_trades_dataframe()
        if not trades_df.empty:
            trades_file = results_dir / f"trades_{timestamp}.csv"
            trades_df.to_csv(trades_file)
            logger.info(f"Trades saved to {trades_file}")
        
        # Save equity curve
        equity_df = backtester.get_equity_curve()
        if not equity_df.empty:
            equity_file = results_dir / f"equity_{timestamp}.csv"
            equity_df.to_csv(equity_file)
            logger.info(f"Equity curve saved to {equity_file}")
        
        # Print results summary
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Period: {result.start_date} to {result.end_date}")
        print(f"Total Trades: {result.total_trades}")
        print(f"Win Rate: {result.win_rate:.1%}")
        print(f"Total PnL: ${result.total_pnl:.2f}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"Max Drawdown: {result.max_drawdown_pct:.1f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Trades/Week: {result.trades_per_week:.1f}")
        print("="*50)
        
        # Validate targets
        targets = backtest_config.get("metrics", {})
        
        print("\nTARGET VALIDATION:")
        print(f"Win Rate: {result.win_rate:.1%} (target: >{targets.get('target_win_rate', 0.55):.1%}) {'✅' if result.win_rate > targets.get('target_win_rate', 0.55) else '❌'}")
        print(f"Profit Factor: {result.profit_factor:.2f} (target: >{targets.get('target_profit_factor', 1.8):.1f}) {'✅' if result.profit_factor > targets.get('target_profit_factor', 1.8) else '❌'}")
        print(f"Max Drawdown: {result.max_drawdown_pct:.1f}% (target: <{targets.get('target_max_drawdown', 0.12):.1%}) {'✅' if result.max_drawdown_pct < targets.get('target_max_drawdown', 0.12)*100 else '❌'}")
        print(f"Trades/Week: {result.trades_per_week:.1f} (target: {targets.get('target_trades_per_week', 3.0):.1f}) {'✅' if result.trades_per_week >= 3 else '❌'}")
        
        return result.model_dump()
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return {}


def setup_data() -> str:
    """Setup and download required data"""
    logger.info("Setting up data...")
    
    # Download 12 months of data
    data_file = download_data({}, days=365)
    
    if not data_file:
        # Try with default data file if exists
        default_file = "data/XAUUSDT_15m_12months.csv"
        if Path(default_file).exists():
            logger.info(f"Using existing data file: {default_file}")
            return "XAUUSDT_15m_12months.csv"
        else:
            logger.error("No data available. Please check internet connection.")
            return ""
    
    return data_file


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="XAU1 Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["backtest", "live"],
        default="backtest",
        help="Run mode: backtest or live (default: backtest)"
    )
    parser.add_argument(
        "--config",
        default="src/xau1/config/strategy_params.yaml",
        help="Strategy configuration file"
    )
    parser.add_argument(
        "--data",
        help="Data file to use (default: auto-download)"
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Download data and exit"
    )
    
    args = parser.parse_args()
    
    # Setup data
    if args.data:
        data_file = args.data
    else:
        data_file = setup_data()
    
    if not data_file:
        logger.error("No data available")
        sys.exit(1)
    
    if args.setup_only:
        logger.info("Data setup completed.")
        return
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        sys.exit(1)
    
    # Run based on mode
    if args.mode == "backtest":
        results = run_backtest(config, data_file)
        
        # Exit with error if targets not met
        if results:
            if results['win_rate'] < 0.5:  # Basic validation
                logger.warning("Strategy did not meet minimum performance targets")
                sys.exit(1)
        else:
            logger.error("Backtest failed to produce results")
            sys.exit(1)
    
    elif args.mode == "live":
        logger.warning("Live trading not yet implemented. Use backtest mode.")
        sys.exit(1)
    
    logger.info("XAU1 bot execution completed.")


if __name__ == "__main__":
    main()