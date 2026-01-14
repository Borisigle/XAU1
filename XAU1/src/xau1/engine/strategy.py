"""
XAU1 Trading Strategy
SMC + Order Flow entry logic and risk management
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from xau1.engine.indicators import SMCIndicators

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trade signal types"""
    TYPE1_BOS_FVG_RSI = "type1_bos_fvg_rsi"
    TYPE2_OB_LIQUIDITY = "type2_ob_liquidity"
    TYPE3_RSI_DIVERGENCE = "type3_rsi_divergence"


class TradeDirection(Enum):
    """Trade direction"""
    LONG = "long"
    SHORT = "short"


class TradeSetup(BaseModel):
    """Trade setup data"""
    signal_type: SignalType
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit1: float
    take_profit2: float
    confluence_score: int
    setup_timestamp: datetime
    reason: str


class PortfolioState(BaseModel):
    """Current portfolio state"""
    equity: float
    max_position_size: float
    active_positions: int
    daily_loss: float
    max_daily_loss: float


class XAU1Strategy:
    """XAU1 Trading Strategy"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_per_trade = config.get("risk_management", {}).get("position_size_percentage", 1.0)
        self.sl_pips = config.get("risk_management", {}).get("stop_loss_pips", 30)
        self.tp1_pips = config.get("risk_management", {}).get("take_profit1_pips", 50)
        self.tp2_pips = config.get("risk_management", {}).get("take_profit2_pips", 100)
        self.min_rr = config.get("risk_management", {}).get("min_risk_reward_ratio", 2.0)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from price data
        
        Args:
            df: DataFrame with OHLCV and indicator data
            
        Returns:
            DataFrame with trade signals
        """
        logger.info("Generating trading signals...")
        
        df = df.copy()
        df['signal_long'] = False
        df['signal_short'] = False
        df['signal_type'] = None
        df['signal_reason'] = ''
        df['confluence_score'] = 0
        
        # Check each candle for signals
        for i in range(20, len(df)):
            current_candle = df.iloc[i]
            
            # Skip if not during active session or Friday after 18:00
            if not current_candle['active_session'] or current_candle['is_friday_after_18']:
                continue
            
            # Check each signal type
            setup = None
            
            # Type 1: BOS + FVG + RSI extremo
            if self.config.get("entry_rules", {}).get("type1_bos_fvg_rsi", {}).get("enabled", True):
                setup = self._check_type1_setup(df, i)
            
            # Type 2: Order Block + liquidity sweep
            if setup is None and self.config.get("entry_rules", {}).get("type2_ob_liquidity", {}).get("enabled", True):
                setup = self._check_type2_setup(df, i)
            
            # Type 3: Divergencia RSI + BOS
            if setup is None and self.config.get("entry_rules", {}).get("type3_rsi_divergence", {}).get("enabled", True):
                setup = self._check_type3_setup(df, i)
            
            # Record signal if found
            if setup:
                if setup.direction == TradeDirection.LONG:
                    df.loc[df.index[i], 'signal_long'] = True
                else:
                    df.loc[df.index[i], 'signal_short'] = True
                
                df.loc[df.index[i], 'signal_type'] = setup.signal_type.value
                df.loc[df.index[i], 'signal_reason'] = setup.reason
                df.loc[df.index[i], 'confluence_score'] = setup.confluence_score
        
        logger.info(f"Generated {df['signal_long'].sum()} long signals and {df['signal_short'].sum()} short signals")
        
        return df
    
    def _check_type1_setup(self, df: pd.DataFrame, index: int) -> Optional[TradeSetup]:
        """
        Type 1: BOS + FVG + RSI extremo
        High RR setup
        """
        current = df.iloc[index]
        
        # Need recent structure data
        if index < 30:
            return None
        
        lookback_data = df.iloc[index-20:index]
        
        # Bullish setup
        if (current['bos_bullish'] and current['fvg_bullish'] and 
            current['rsi'] < 35):  # RSI oversold
            
            confluence = self._calculate_confluence_score(df, index, 'bullish')
            
            if confluence >= self.config.get("entry_rules", {}).get("type1_bos_fvg_rsi", {}).get("min_confluence", 3):
                entry_price = current['close']
                sl = entry_price - (self.sl_pips * 0.01)  # 25-35 pips
                tp1 = entry_price + (self.tp1_pips * 0.01)
                tp2 = entry_price + (self.tp2_pips * 0.01)
                
                return TradeSetup(
                    signal_type=SignalType.TYPE1_BOS_FVG_RSI,
                    direction=TradeDirection.LONG,
                    entry_price=entry_price,
                    stop_loss=sl,
                    take_profit1=tp1,
                    take_profit2=tp2,
                    confluence_score=confluence,
                    setup_timestamp=current.name,
                    reason=f"BOS Bullish + FVG + RSI({current['rsi']:.1f})"
                )
        
        # Bearish setup
        elif (current['bos_bearish'] and current['fvg_bearish'] and 
              current['rsi'] > 65):  # RSI overbought
            
            confluence = self._calculate_confluence_score(df, index, 'bearish')
            
            if confluence >= self.config.get("entry_rules", {}).get("type1_bos_fvg_rsi", {}).get("min_confluence", 3):
                entry_price = current['close']
                sl = entry_price + (self.sl_pips * 0.01)
                tp1 = entry_price - (self.tp1_pips * 0.01)
                tp2 = entry_price - (self.tp2_pips * 0.01)
                
                return TradeSetup(
                    signal_type=SignalType.TYPE1_BOS_FVG_RSI,
                    direction=TradeDirection.SHORT,
                    entry_price=entry_price,
                    stop_loss=sl,
                    take_profit1=tp1,
                    take_profit2=tp2,
                    confluence_score=confluence,
                    setup_timestamp=current.name,
                    reason=f"BOS Bearish + FVG + RSI({current['rsi']:.1f})"
                )
        
        return None
    
    def _check_type2_setup(self, df: pd.DataFrame, index: int) -> Optional[TradeSetup]:
        """
        Type 2: Order Block + liquidity sweep
        """
        current = df.iloc[index]
        
        if index < 30:
            return None
        
        # Bullish setup: Liquidity sweep + retest of order block
        if (current['ob_bullish'] and
            self._liquidity_sweep_detected(df, index-10, index, direction='bearish')):
            
            confluence = self._calculate_confluence_score(df, index, 'bullish')
            
            if confluence >= self.config.get("entry_rules", {}).get("type2_ob_liquidity", {}).get("min_confluence", 3):
                entry_price = current['close']
                sl = entry_price - (self.sl_pips * 0.01)
                tp1 = entry_price + (self.tp1_pips * 0.01)
                tp2 = entry_price + (self.tp2_pips * 0.01)
                
                return TradeSetup(
                    signal_type=SignalType.TYPE2_OB_LIQUIDITY,
                    direction=TradeDirection.LONG,
                    entry_price=entry_price,
                    stop_loss=sl,
                    take_profit1=tp1,
                    take_profit2=tp2,
                    confluence_score=confluence,
                    setup_timestamp=current.name,
                    reason="Order Block + Liquidity Sweep"
                )
        
        # Bearish setup
        elif (current['ob_bearish'] and
              self._liquidity_sweep_detected(df, index-10, index, direction='bullish')):
            
            confluence = self._calculate_confluence_score(df, index, 'bearish')
            
            if confluence >= self.config.get("entry_rules", {}).get("type2_ob_liquidity", {}).get("min_confluence", 3):
                entry_price = current['close']
                sl = entry_price + (self.sl_pips * 0.01)
                tp1 = entry_price - (self.tp1_pips * 0.01)
                tp2 = entry_price - (self.tp2_pips * 0.01)
                
                return TradeSetup(
                    signal_type=SignalType.TYPE2_OB_LIQUIDITY,
                    direction=TradeDirection.SHORT,
                    entry_price=entry_price,
                    stop_loss=sl,
                    take_profit1=tp1,
                    take_profit2=tp2,
                    confluence_score=confluence,
                    setup_timestamp=current.name,
                    reason="Order Block + Liquidity Sweep"
                )
        
        return None
    
    def _check_type3_setup(self, df: pd.DataFrame, index: int) -> Optional[TradeSetup]:
        """
        Type 3: Divergencia RSI + BOS
        """
        current = df.iloc[index]
        
        if index < 30:
            return None
        
        # Bullish divergence + BOS
        if (current['bos_bullish'] and 
            current['rsi_div_bullish'] and 
            current['rsi'] < 50):
            
            confluence = self._calculate_confluence_score(df, index, 'bullish')
            
            if confluence >= self.config.get("entry_rules", {}).get("type3_rsi_divergence", {}).get("min_confluence", 2):
                entry_price = current['close']
                sl = entry_price - (self.sl_pips * 0.01)
                tp1 = entry_price + (self.tp1_pips * 0.01)
                tp2 = entry_price + (self.tp2_pips * 0.01)
                
                return TradeSetup(
                    signal_type=SignalType.TYPE3_RSI_DIVERGENCE,
                    direction=TradeDirection.LONG,
                    entry_price=entry_price,
                    stop_loss=sl,
                    take_profit1=tp1,
                    take_profit2=tp2,
                    confluence_score=confluence,
                    setup_timestamp=current.name,
                    reason=f"RSI Divergence + BOS (RSI={current['rsi']:.1f})"
                )
        
        # Bearish divergence + BOS
        elif (current['bos_bearish'] and 
              current['rsi_div_bearish'] and 
              current['rsi'] > 50):
            
            confluence = self._calculate_confluence_score(df, index, 'bearish')
            
            if confluence >= self.config.get("entry_rules", {}).get("type3_rsi_divergence", {}).get("min_confluence", 2):
                entry_price = current['close']
                sl = entry_price + (self.sl_pips * 0.01)
                tp1 = entry_price - (self.tp1_pips * 0.01)
                tp2 = entry_price - (self.tp2_pips * 0.01)
                
                return TradeSetup(
                    signal_type=SignalType.TYPE3_RSI_DIVERGENCE,
                    direction=TradeDirection.SHORT,
                    entry_price=entry_price,
                    stop_loss=sl,
                    take_profit1=tp1,
                    take_profit2=tp2,
                    confluence_score=confluence,
                    setup_timestamp=current.name,
                    reason=f"RSI Divergence + BOS (RSI={current['rsi']:.1f})"
                )
        
        return None
    
    def _calculate_confluence_score(self, df: pd.DataFrame, index: int, direction: str) -> int:
        """
        Calculate confluence score (0-5)
        More factors = higher probability
        """
        current = df.iloc[index]
        score = 0
        
        if direction == 'bullish':
            # RSI extreme
            if current['rsi'] < 35:
                score += 1
            
            # FVG present
            if current['fvg_bullish']:
                score += 1
            
            # Order block
            if current['ob_bullish']:
                score += 1
            
            # Liquidity
            if current['equal_lows']:
                score += 1
        
        else:  # bearish
            # RSI extreme
            if current['rsi'] > 65:
                score += 1
            
            # FVG present
            if current['fvg_bearish']:
                score += 1
            
            # Order block
            if current['ob_bearish']:
                score += 1
            
            # Liquidity
            if current['equal_highs']:
                score += 1
        
        return min(score, 5)  # Max score of 5
    
    def _liquidity_sweep_detected(self, df: pd.DataFrame, start_idx: int, end_idx: int, direction: str) -> bool:
        """Detect liquidity sweep in range"""
        lookback_data = df.iloc[start_idx:end_idx]
        
        if direction == 'bullish':
            # Bullish liquidity sweep = broke above equal highs then retraced
            return lookback_data['equal_highs'].any() and (lookback_data['high'].max() > lookback_data['close'].iloc[-1] * 1.001)
        else:
            # Bearish liquidity sweep = broke below equal lows then retraced
            return lookback_data['equal_lows'].any() and (lookback_data['low'].min() < lookback_data['close'].iloc[-1] * 0.999)
    
    def calculate_position_size(self, equity: float, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk"""
        risk_amount = equity * (self.risk_per_trade / 100)
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0
        
        position_size = risk_amount / risk_per_unit
        return position_size
    
    def validate_setup(self, setup: TradeSetup, portfolio: PortfolioState) -> bool:
        """Validate if setup meets criteria"""
        # Check max positions
        if portfolio.active_positions >= self.config.get("risk_management", {}).get("max_positions", 2):
            logger.debug("Max positions reached, skipping setup")
            return False
        
        # Check R/R ratio
        risk = abs(setup.entry_price - setup.stop_loss)
        reward = abs(setup.take_profit1 - setup.entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        if rr_ratio < self.min_rr:
            logger.debug(f"R/R ratio {rr_ratio:.2f} below minimum {self.min_rr}")
            return False
        
        # Check daily loss limit
        if portfolio.daily_loss >= portfolio.max_daily_loss:
            logger.debug("Daily loss limit reached")
            return False
        
        # Check confluence minimum
        min_confluence = min([
            self.config.get("entry_rules", {}).get("type1_bos_fvg_rsi", {}).get("min_confluence", 3),
            self.config.get("entry_rules", {}).get("type2_ob_liquidity", {}).get("min_confluence", 3),
            self.config.get("entry_rules", {}).get("type3_rsi_divergence", {}).get("min_confluence", 2),
        ])
        
        if setup.confluence_score < min_confluence:
            logger.debug(f"Confluence score {setup.confluence_score} below minimum")
            return False
        
        return True