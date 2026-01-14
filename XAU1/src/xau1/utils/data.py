"""
Data utilities for loading, saving, and validating market data
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class DataManager:
    """Manage market data loading, saving, and validation"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def load_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Load CSV data file"""
        try:
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                logger.error(f"File not found: {filepath}")
                return None
            
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return None
            
            logger.info(f"Loaded {len(df)} rows from {filename}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return None
    
    def save_csv(self, df: pd.DataFrame, filename: str) -> bool:
        """Save DataFrame to CSV"""
        try:
            filepath = self.data_dir / filename
            df.to_csv(filepath)
            logger.info(f"Saved {len(df)} rows to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
            return False
    
    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """Validate data quality and completeness"""
        if df is None or df.empty:
            logger.error("Empty DataFrame")
            return False
        
        # Check for null values
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            logger.warning(f"Found {null_count} null values in data")
        
        # Check for zero volume bars
        zero_volume = (df['volume'] == 0).sum()
        if zero_volume > len(df) * 0.1:  # More than 10%
            logger.warning(f"Found {zero_volume} bars with zero volume")
        
        # Check for outlier prices
        for col in ['open', 'high', 'low', 'close']:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((df[col] < (q1 - 3 * iqr)) | (df[col] > (q3 + 3 * iqr))).sum()
            if outliers > 0:
                logger.warning(f"Found {outliers} outliers in {col}")
        
        # Check chronological order
        time_diffs = df.index.to_series().diff().dropna()
        if (time_diffs <= pd.Timedelta(0)).any():
            logger.error("Data is not in chronological order")
            return False
        
        return True
    
    def get_data_range(self, df: pd.DataFrame) -> tuple:
        """Get data date range"""
        if df is None or df.empty:
            return None, None
        
        return df.index.min(), df.index.max()


def load_config(filepath: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {filepath}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def save_config(config: dict, filepath: str) -> bool:
    """Save configuration to YAML file"""
    try:
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved config to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False