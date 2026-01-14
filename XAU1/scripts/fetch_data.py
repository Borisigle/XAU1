#!/usr/bin/env python3
"""
XAUUSDT Data Fetcher
Downloads historical data from Binance for backtesting
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from xau1.exchange.binance import BinanceClient
from xau1.utils.data import DataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_data(days: int = 365, timeframe: str = "15m", symbol: str = "XAUUSDT"):
    """Fetch historical data from Binance"""
    
    logger.info(f"Fetching {days} days of {symbol} {timeframe} data from Binance...")
    
    try:
        # Initialize Binance client
        binance = BinanceClient()
        
        # Calculate start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Fetch data
        df = binance.fetch_ohlcv(
            timeframe=timeframe,
            since=start_date,
            to_date=end_date
        )
        
        if df.empty:
            logger.error("No data fetched")
            return None
        
        logger.info(f"Successfully fetched {len(df)} candles")
        
        # Display data info
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None


def validate_data(df, required_candles: int = 33000):
    """Validate downloaded data quality"""
    
    logger.info("Validating data quality...")
    
    issues = []
    
    # Check size
    if len(df) < required_candles:
        issues.append(f"Insufficient data: {len(df)} candles, expected {required_candles}")
    
    # Check for gaps
    expected_interval = pd.Timedelta(minutes=15)
    time_diffs = df.index.to_series().diff().dropna()
    gaps = time_diffs[time_diffs > expected_interval * 2]  # 2x interval = potential gap
    
    if len(gaps) > 0:
        issues.append(f"Found {len(gaps)} potential data gaps")
        logger.warning(f"Data gaps found at: {gaps.index[:5].tolist()}")
    
    # Check for zero volume
    zero_volume = (df['volume'] == 0).sum()
    if zero_volume > len(df) * 0.05:  # More than 5%
        issues.append(f"High zero-volume bars: {zero_volume} ({zero_volume/len(df)*100:.1f}%)")
    
    # Check for outliers
    for col in ['open', 'high', 'low', 'close']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = ((df[col] < (q1 - 3 * iqr)) | (df[col] > (q3 + 3 * iqr))).sum()
        
        if outliers > 0:
            logger.warning(f"Found {outliers} outliers in {col}")
    
    if issues:
        logger.warning("Data validation issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Data validation passed")
    
    return len(issues) == 0


def save_data(df, filename: str = None):
    """Save data to CSV"""
    
    if filename is None:
        # Generate default filename
        start_date = df.index.min().strftime("%Y%m%d")
        end_date = df.index.max().strftime("%Y%m%d")
        filename = f"XAUUSDT_15m_{start_date}_{end_date}.csv"
    
    try:
        # Ensure data directory exists
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Save data
        filepath = data_dir / filename
        df.to_csv(filepath)
        
        logger.info(f"Data saved to {filepath}")
        logger.info(f"File size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")
        
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        return None


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description="Fetch XAUUSDT historical data from Binance"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days to fetch (default: 365)"
    )
    
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15m",
        choices=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"],
        help="Candle timeframe (default: 15m)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: auto-generated)"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate data quality after download"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if file already exists"
    )
    
    args = parser.parse_args()
    
    # Check if file already exists
    if args.skip_existing:
        data_dir = Path("data")
        if data_dir.exists():
            existing_files = list(data_dir.glob("XAUUSDT*.csv"))
            if existing_files:
                logger.info(f"Found existing data file: {existing_files[0]}")
                logger.info("Skipping download (--skip-existing flag)")
                sys.exit(0)
    
    # Fetch data
    df = fetch_data(days=args.days, timeframe=args.timeframe)
    
    if df is None:
        logger.error("Failed to fetch data")
        sys.exit(1)
    
    # Validate data
    if args.validate:
        is_valid = validate_data(df)
        if not is_valid:
            logger.warning("Data validation issues found")
    
    # Save data
    output_path = save_data(df, args.output)
    
    if output_path is None:
        logger.error("Failed to save data")
        sys.exit(1)
    
    # Summary
    logger.info("=" * 50)
    logger.info("DATA FETCH SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Symbol: XAUUSDT")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Period: {args.days} days")
    logger.info(f"Candles: {len(df):,}")
    logger.info(f"Date Range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Output: {output_path}")
    logger.info(f"File Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info("=" * 50)
    
    logger.info("Data fetch completed successfully!")


if __name__ == "__main__":
    main()