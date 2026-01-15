#!/usr/bin/env python3
"""
XAU1 Utility Functions
Common utility functions for the XAU1 trading system
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional
import yaml
import json

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def setup_logging(name: str = "xau1", level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration for XAU1 system"""
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_file = f"{log_dir}/{name}_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(name)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML or JSON file"""
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            elif config_path.endswith('.json'):
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        return {}
    except Exception as e:
        logging.error(f"Error loading config {config_path}: {e}")
        return {}


def save_config(config: Dict, config_path: str) -> bool:
    """Save configuration to YAML or JSON file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif config_path.endswith('.json'):
                json.dump(config, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
        
        logging.info(f"Config saved to {config_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving config to {config_path}: {e}")
        return False


def validate_config(config: Dict, required_keys: List[str]) -> bool:
    """Validate that configuration contains required keys"""
    missing_keys = []
    
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    if missing_keys:
        logging.error(f"Missing required config keys: {missing_keys}")
        return False
    
    return True


def calculate_pips(price_diff: float, symbol: str = "XAUUSDT") -> float:
    """Calculate pips from price difference"""
    if symbol == "XAUUSDT":
        return price_diff / 0.01  # 1 pip = 0.01 for XAU
    else:
        return price_diff  # Default assumption


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount for display"""
    if currency == "USD":
        return f"${amount:,.2f}"
    elif currency == "XAU":
        return f"{amount:.4f} XAU"
    else:
        return f"{amount:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage value for display"""
    return f"{value * 100:.{decimals}f}%"


def format_pips(pips: float, decimals: int = 1) -> str:
    """Format pips value for display"""
    return f"{pips:.{decimals}f} pips"


def get_trading_session(timestamp: datetime) -> str:
    """Get current trading session based on UTC time"""
    hour = timestamp.hour
    
    # London session: 13:00-20:00 UTC
    if 13 <= hour < 20:
        return "london"
    # New York session: 13:30-21:00 UTC  
    elif 13.5 <= hour < 21:
        return "newyork"
    else:
        return "none"


def is_trading_hours(timestamp: datetime) -> bool:
    """Check if current time is within trading hours"""
    session = get_trading_session(timestamp)
    return session != "none"


def is_friday_cutoff(timestamp: datetime) -> bool:
    """Check if it's Friday after 18:00 UTC (avoid Friday volatility)"""
    return timestamp.weekday() == 4 and timestamp.hour >= 18


def calculate_position_size(capital: float, risk_pct: float, entry_price: float, stop_loss: float) -> float:
    """Calculate position size based on risk management"""
    risk_amount = capital * (risk_pct / 100)
    risk_per_unit = abs(entry_price - stop_loss)
    
    if risk_per_unit == 0:
        return 0
    
    return risk_amount / risk_per_unit


def calculate_risk_reward(entry_price: float, stop_loss: float, take_profit: float, direction: str) -> float:
    """Calculate risk-reward ratio"""
    if direction.lower() == 'long':
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
    else:  # short
        risk = abs(entry_price - stop_loss)
        reward = abs(entry_price - take_profit)
    
    if risk == 0:
        return 0
    
    return reward / risk


def validate_price_data(df) -> bool:
    """Validate price data integrity"""
    try:
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return False
        
        # Check for missing values
        if df[required_cols].isnull().any().any():
            return False
        
        # Check OHLC logic (High >= all, Low <= all)
        if not (df['high'] >= df['open']).all() or not (df['high'] >= df['close']).all():
            return False
        if not (df['low'] <= df['open']).all() or not (df['low'] <= df['close']).all():
            return False
        
        # Check for negative values
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        if (df[numeric_cols] < 0).any().any():
            return False
        
        return True
        
    except Exception:
        return False


def calculate_drawdown(equity_curve: List[float]) -> Dict:
    """Calculate drawdown metrics from equity curve"""
    try:
        if len(equity_curve) < 2:
            return {'max_drawdown': 0, 'current_drawdown': 0}
        
        # Convert to pandas Series for easier calculation
        import pandas as pd
        equity = pd.Series(equity_curve)
        
        # Calculate running maximum
        running_max = equity.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity - running_max) / running_max
        
        # Return metrics
        return {
            'max_drawdown': abs(drawdown.min()) * 100,
            'current_drawdown': abs(drawdown.iloc[-1]) * 100,
            'drawdown_series': drawdown.tolist()
        }
        
    except Exception as e:
        logging.error(f"Error calculating drawdown: {e}")
        return {'max_drawdown': 0, 'current_drawdown': 0}


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from returns"""
    try:
        if len(returns) < 2:
            return 0
        
        import numpy as np
        returns_array = np.array(returns)
        
        # Calculate mean return and standard deviation
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0
        
        # Calculate Sharpe ratio
        excess_return = mean_return - risk_free_rate
        return excess_return / std_return * np.sqrt(252)  # Annualized
        
    except Exception as e:
        logging.error(f"Error calculating Sharpe ratio: {e}")
        return 0


def create_directories(dirs: List[str]):
    """Create directories if they don't exist"""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def get_system_info() -> Dict:
    """Get system information for debugging"""
    import platform
    import sys
    
    return {
        'python_version': sys.version,
        'platform': platform.platform(),
        'working_directory': os.getcwd(),
        'environment_variables': {
            'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
            'PATH': os.environ.get('PATH', '')
        }
    }


def backup_file(file_path: str, backup_dir: str = "backups") -> str:
    """Create backup of a file with timestamp"""
    try:
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)
        
        # Generate backup filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.basename(file_path)
        backup_path = os.path.join(backup_dir, f"{base_name}.backup_{timestamp}")
        
        # Copy file
        import shutil
        shutil.copy2(file_path, backup_path)
        
        logging.info(f"File backed up to {backup_path}")
        return backup_path
        
    except Exception as e:
        logging.error(f"Error backing up file {file_path}: {e}")
        return ""


def clean_old_logs(log_dir: str = "logs", days_to_keep: int = 30):
    """Clean old log files"""
    try:
        import glob
        import time
        
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        log_files = glob.glob(os.path.join(log_dir, "*.log"))
        
        cleaned_count = 0
        for log_file in log_files:
            if os.path.getmtime(log_file) < cutoff_time:
                os.remove(log_file)
                cleaned_count += 1
        
        logging.info(f"Cleaned {cleaned_count} old log files from {log_dir}")
        return cleaned_count
        
    except Exception as e:
        logging.error(f"Error cleaning old logs: {e}")
        return 0


def main():
    """Test utility functions"""
    print("Testing XAU1 utilities...")
    
    # Test logging setup
    logger = setup_logging("test")
    logger.info("Testing logging setup")
    
    # Test config loading
    config = load_config("src/xau1/config/strategy_params.yaml")
    print(f"Config loaded: {len(config)} keys")
    
    # Test trading session
    now = datetime.now()
    session = get_trading_session(now)
    print(f"Current session: {session}")
    
    # Test trading hours
    trading = is_trading_hours(now)
    print(f"Is trading hours: {trading}")
    
    # Test Friday cutoff
    friday_cutoff = is_friday_cutoff(now)
    print(f"Is Friday cutoff: {friday_cutoff}")
    
    # Test position sizing
    pos_size = calculate_position_size(10000, 1.0, 2025.50, 2000.50)
    print(f"Position size: {pos_size}")
    
    # Test risk-reward
    rr = calculate_risk_reward(2025.50, 2000.50, 2075.50, 'long')
    print(f"Risk-reward ratio: {rr}")
    
    # Test system info
    info = get_system_info()
    print(f"System info: {info['platform']}")
    
    print("✅ All utility tests completed")


if __name__ == "__main__":
    main()