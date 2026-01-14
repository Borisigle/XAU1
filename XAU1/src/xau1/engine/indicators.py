"""
Smart Money Concepts (SMC) Indicators
Implements SMC-specific indicators for XAUUSDT trading
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

logger = logging.getLogger(__name__)


class SMCIndicators:
    """Smart Money Concepts indicator calculator"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._validate_dataframe()
    
    def _validate_dataframe(self):
        """Validate DataFrame has required columns"""
        required_columns = ['high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")
    
    def calculate_swing_points(
        self,
        left_bars: int = 2,
        right_bars: int = 2,
    ) -> pd.DataFrame:
        """
        Calculate Swing Highs and Swing Lows
        
        Args:
            left_bars: Number of bars to the left for comparison
            right_bars: Number of bars to the right for comparison
            
        Returns:
            DataFrame with swing points
        """
        df = self.df.copy()
        
        # Swing Highs: Higher than left and right bars
        swing_high = (
            df['high'] == df['high'].rolling(window=left_bars+right_bars+1, center=True).max()
        )
        
        # Swing Lows: Lower than left and right bars
        swing_low = (
            df['low'] == df['low'].rolling(window=left_bars+right_bars+1, center=True).min()
        )
        
        # Create swing level columns
        df['swing_high'] = np.where(swing_high, df['high'], np.nan)
        df['swing_low'] = np.where(swing_low, df['low'], np.nan)
        
        # Fill forward swing levels for reference
        df['swing_high_level'] = df['swing_high'].fillna(method='ffill')
        df['swing_low_level'] = df['swing_low'].fillna(method='ffill')
        
        logger.debug(f"Calculated {swing_high.sum()} swing highs and {swing_low.sum()} swing lows")
        
        return df
    
    def calculate_fair_value_gaps(
        self,
        merge_consecutive: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate Fair Value Gaps (FVG)
        
        Args:
            merge_consecutive: Whether to merge consecutive FVGs
            
        Returns:
            DataFrame with FVG markers
        """
        df = self.df.copy()
        
        # Bullish FVG: Current low > previous high 2 candles ago
        df['fvg_bullish'] = (
            df['low'] > df['high'].shift(2)
        )
        
        # Bearish FVG: Current high < previous low 2 candles ago
        df['fvg_bearish'] = (
            df['high'] < df['low'].shift(2)
        )
        
        # FVG levels
        df['fvg_level_bullish'] = np.where(df['fvg_bullish'], df['low'], np.nan)
        df['fvg_level_bearish'] = np.where(df['fvg_bearish'], df['high'], np.nan)
        
        # FVG midpoints for plotting
        df['fvg_mid_bullish'] = np.where(
            df['fvg_bullish'],
            (df['low'] + df['high'].shift(2)) / 2,
            np.nan
        )
        df['fvg_mid_bearish'] = np.where(
            df['fvg_bearish'],
            (df['high'] + df['low'].shift(2)) / 2,
            np.nan
        )
        
        # Merge consecutive FVGs if requested
        if merge_consecutive:
            df['fvg_bullish'] = self._merge_consecutive_gaps(df['fvg_bullish'])
            df['fvg_bearish'] = self._merge_consecutive_gaps(df['fvg_bearish'])
        
        logger.debug(f"Calculated fair value gaps")
        
        return df
    
    def _merge_consecutive_gaps(self, series: pd.Series, max_gap: int = 5) -> pd.Series:
        """Merge consecutive gaps"""
        mask = series.copy()
        for i in range(1, len(series)):
            if mask.iloc[i]:
                # Check if previous gap exists within max_gap
                recent_gaps = mask.iloc[max(0, i-max_gap):i]
                if recent_gaps.any():
                    mask.iloc[i] = False
        return mask
    
    def calculate_market_structure(
        self,
        bos_lookback: int = 20,
        choch_lookback: int = 30,
    ) -> pd.DataFrame:
        """
        Calculate Market Structure (BOS and CHoCH)
        
        Args:
            bos_lookback: Lookback period for Break of Structure
            choch_lookback: Lookback period for Change of Character
            
        Returns:
            DataFrame with market structure markers
        """
        df = self.df.copy()
        
        # Get swing points
        df = self.calculate_swing_points()
        
        # Initialize structure tracking
        df['market_trend'] = 'neutral'
        df['bos_bullish'] = False
        df['bos_bearish'] = False
        df['choch_bullish'] = False
        df['choch_bearish'] = False
        
        current_trend = 'neutral'
        last_swing_high = 0
        last_swing_low = float('inf')
        
        for i in range(len(df)):
            # Update swing levels
            if not pd.isna(df['swing_high'].iloc[i]):
                last_swing_high = df['swing_high'].iloc[i]
            if not pd.isna(df['swing_low'].iloc[i]):
                last_swing_low = df['swing_low'].iloc[i]
            
            # BOS Bullish: Price breaks above last swing high
            if df['close'].iloc[i] > last_swing_high:
                df.loc[df.index[i], 'bos_bullish'] = True
                df.loc[df.index[i], 'market_trend'] = 'bullish'
                current_trend = 'bullish'
            
            # BOS Bearish: Price breaks below last swing low
            elif df['close'].iloc[i] < last_swing_low:
                df.loc[df.index[i], 'bos_bearish'] = True
                df.loc[df.index[i], 'market_trend'] = 'bearish'
                current_trend = 'bearish'
            else:
                df.loc[df.index[i], 'market_trend'] = current_trend
        
        logger.debug(f"Calculated market structure")
        
        return df
    
    def calculate_order_blocks(
        self,
        lookback_period: int = 50,
        volume_threshold: float = 1.5,
    ) -> pd.DataFrame:
        """
        Calculate Order Blocks
        
        Args:
            lookback_period: Period to look for order blocks
            volume_threshold: Volume multiplier threshold
            
        Returns:
            DataFrame with order block markers
        """
        df = self.df.copy()
        
        # Calculate average volume
        df['avg_volume'] = df['volume'].rolling(window=lookback_period).mean()
        
        # High volume candles
        df['high_volume'] = df['volume'] > (df['avg_volume'] * volume_threshold)
        
        # Bullish order block: High volume up candle before swing high
        df['ob_bullish'] = (
            df['high_volume'] &
            (df['close'] > df['open']) &
            (df['high'].shift(-1) > df['high'])
        )
        
        # Bearish order block: High volume down candle before swing low
        df['ob_bearish'] = (
            df['high_volume'] &
            (df['close'] < df['open']) &
            (df['low'].shift(-1) < df['low'])
        )
        
        # Order block levels
        df['ob_level_bullish'] = np.where(df['ob_bullish'], df['low'], np.nan)
        df['ob_level_bearish'] = np.where(df['ob_bearish'], df['high'], np.nan)
        
        logger.debug(f"Calculated order blocks")
        
        return df
    
    def calculate_liquidity_zones(
        self,
        lookback_period: int = 100,
    ) -> pd.DataFrame:
        """
        Calculate Liquidity Zones (equal highs/lows)
        
        Args:
            lookback_period: Period to look for liquidity
            
        Returns:
            DataFrame with liquidity zones
        """
        df = self.df.copy()
        
        # Find equal highs within tolerance
        tolerance = df['high'].rolling(lookback_period).std() * 0.001
        
        df['equal_highs'] = False
        df['equal_lows'] = False
        
        for i in range(len(df)):
            # Check for equal highs in lookback period
            lookback_start = max(0, i - lookback_period)
            tolerance_range = (df['high'].iloc[i] - tolerance.iloc[i], df['high'].iloc[i] + tolerance.iloc[i])
            
            matching_highs = (
                (df['high'].iloc[lookback_start:i] >= tolerance_range[0]) &
                (df['high'].iloc[lookback_start:i] <= tolerance_range[1])
            ).sum()
            
            if matching_highs >= 2:  # At least 2 equal highs
                df.loc[df.index[i], 'equal_highs'] = True
            
            # Check for equal lows
            tolerance_range_low = (df['low'].iloc[i] - tolerance.iloc[i], df['low'].iloc[i] + tolerance.iloc[i])
            
            matching_lows = (
                (df['low'].iloc[lookback_start:i] >= tolerance_range_low[0]) &
                (df['low'].iloc[lookback_start:i] <= tolerance_range_low[1])
            ).sum()
            
            if matching_lows >= 2:  # At least 2 equal lows
                df.loc[df.index[i], 'equal_lows'] = True
        
        logger.debug(f"Calculated liquidity zones")
        
        return df
    
    def calculate_rsi_divergence(
        self,
        rsi_period: int = 14,
    ) -> pd.DataFrame:
        """
        Calculate RSI and detect divergences
        
        Args:
            rsi_period: RSI calculation period
            
        Returns:
            DataFrame with RSI and divergence signals
        """
        df = self.df.copy()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Get swing points
        df = self.calculate_swing_points()
        
        # Detect divergences
        df['rsi_div_bullish'] = False
        df['rsi_div_bearish'] = False
        
        for i in range(20, len(df)):  # Start from 20 to have enough history
            # Bullish divergence: Lower lows in price, higher lows in RSI
            if not pd.isna(df['swing_low'].iloc[i]):
                previous_swings = df['swing_low'].iloc[:i].dropna()
                if len(previous_swings) >= 2:
                    last_swing = previous_swings.iloc[-1]
                    prev_swing = previous_swings.iloc[-2]
                    
                    # Check divergence
                    if (last_swing < prev_swing and  # Price making lower lows
                        df['rsi'].iloc[i] > df['rsi'].loc[previous_swings.index[-1]]):  # RSI higher
                        df.loc[df.index[i], 'rsi_div_bullish'] = True
            
            # Bearish divergence: Higher highs in price, lower highs in RSI
            if not pd.isna(df['swing_high'].iloc[i]):
                previous_swings = df['swing_high'].iloc[:i].dropna()
                if len(previous_swings) >= 2:
                    last_swing = previous_swings.iloc[-1]
                    prev_swing = previous_swings.iloc[-2]
                    
                    if (last_swing > prev_swing and  # Price making higher highs
                        df['rsi'].iloc[i] < df['rsi'].loc[previous_swings.index[-1]]):  # RSI lower
                        df.loc[df.index[i], 'rsi_div_bearish'] = True
        
        logger.debug(f"Calculated RSI and divergences")
        
        return df
    
    def calculate_atr(
        self,
        period: int = 14,
    ) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR)
        
        Args:
            period: ATR calculation period
            
        Returns:
            DataFrame with ATR
        """
        df = self.df.copy()
        
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate ATR
        df['atr'] = true_range.rolling(window=period).mean()
        
        logger.debug(f"Calculated ATR({period})")
        
        return df
    
    def calculate_trading_sessions(
        self,
        london_start: str = "13:00",
        london_end: str = "20:00",
        nyc_start: str = "13:30",
        nyc_end: str = "21:00",
    ) -> pd.DataFrame:
        """
        Identify trading sessions
        
        Returns:
            DataFrame with session indicators
        """
        df = self.df.copy()
        
        def time_in_range(start, end, current):
            """Check if current time is in range"""
            return start <= current <= end
        
        # Extract time from datetime index
        df['time'] = df.index.time
        
        # London session: 13:00-20:00 UTC
        london_start_t = pd.to_datetime(london_start).time()
        london_end_t = pd.to_datetime(london_end).time()
        df['london_session'] = df['time'].apply(
            lambda x: time_in_range(london_start_t, london_end_t, x)
        )
        
        # NYC session: 13:30-21:00 UTC
        nyc_start_t = pd.to_datetime(nyc_start).time()
        nyc_end_t = pd.to_datetime(nyc_end).time()
        df['nyc_session'] = df['time'].apply(
            lambda x: time_in_range(nyc_start_t, nyc_end_t, x)
        )
        
        # Active session (either London or NYC)
        df['active_session'] = df['london_session'] | df['nyc_session']
        
        # Skip Friday after 18:00 UTC
        df['is_friday_after_18'] = (
            (df.index.dayofweek == 4) &  # Friday
            (df['time'] >= pd.to_datetime("18:00").time())
        )
        
        logger.debug(f"Calculated trading sessions")
        
        return df
    
    def calculate_all_indicators(self) -> pd.DataFrame:
        """
        Calculate all SMC indicators
        
        Returns:
            DataFrame with all indicators
        """
        logger.info("Calculating all SMC indicators...")
        
        df = self.df.copy()
        
        # Swing points
        df = self.calculate_swing_points()
        
        # Fair Value Gaps
        df = self.calculate_fair_value_gaps()
        
        # Market Structure (BOS/CHoCH)
        df = self.calculate_market_structure()
        
        # Order Blocks
        df = self.calculate_order_blocks()
        
        # Liquidity zones
        df = self.calculate_liquidity_zones()
        
        # RSI + Divergence
        df = self.calculate_rsi_divergence()
        
        # ATR
        df = self.calculate_atr()
        
        # Trading sessions
        df = self.calculate_trading_sessions()
        
        logger.info("All SMC indicators calculated successfully")
        
        return df