"""
XAU1 Backtesting Engine
Realistic backtesting with slippage, commissions, and spread
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from xau1.engine.indicators import SMCIndicators
from xau1.engine.strategy import (
    PortfolioState,
    TradeDirection,
    TradeSetup,
    XAU1Strategy,
)

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """Position status"""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


class TradeRecord(BaseModel):
    """Trade execution record"""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: TradeDirection
    signal_type: str
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit1: float
    take_profit2: float
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    risk_reward: Optional[float] = None
    confluence_score: int
    commissions: float = 0.0
    slippage: float = 0.0


class BacktestResult(BaseModel):
    """Backtest results summary"""
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    recovery_factor: float
    avg_trade_pnl: float
    avg_winning_trade: float
    avg_losing_trade: float
    trades_per_week: float
    avg_risk_reward: float
    initial_capital: float
    final_capital: float
    total_return_pct: float


class XAU1Backtester:
    """XAU1 Backtesting Engine"""
    
    def __init__(self, strategy_config: Dict, backtest_config: Dict):
        self.strategy_config = strategy_config
        self.backtest_config = backtest_config
        self.strategy = XAU1Strategy(strategy_config)
        
        # Simulation parameters
        self.initial_capital = backtest_config.get("simulation", {}).get("initial_capital", 10000)
        self.commission_rate = backtest_config.get("simulation", {}).get("commission_rate", 0.0002)
        self.slippage_pips = backtest_config.get("simulation", {}).get("slippage_pips", 1.0)
        self.spread_pips = backtest_config.get("simulation", {}).get("spread_pips", 0.5)
        
        # State tracking
        self.equity_curve = []
        self.trades = []
        self.active_position = None
        
        logger.info(f"Backtester initialized with ${self.initial_capital:.2f} initial capital")
    
    def run_backtest(self, df: pd.DataFrame) -> BacktestResult:
        """
        Run complete backtest on historical data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            BacktestResult with performance metrics
        """
        logger.info("Starting backtest...")
        
        # Step 1: Calculate indicators
        logger.info("Calculating indicators...")
        smc = SMCIndicators(df)
        df_indicators = smc.calculate_all_indicators()
        
        # Step 2: Generate signals
        logger.info("Generating signals...")
        df_signals = self.strategy.generate_signals(df_indicators)
        
        # Step 3: Run simulation
        logger.info("Running simulation...")
        result = self._run_simulation(df_signals)
        
        logger.info("Backtest completed successfully")
        return result
    
    def _run_simulation(self, df: pd.DataFrame) -> BacktestResult:
        """Run tick-by-tick simulation"""
        capital = self.initial_capital
        daily_loss = 0.0
        max_daily_loss = self.strategy_config.get("risk_management", {}).get("max_daily_loss", 0.05) * capital
        
        portfolio = PortfolioState(
            equity=capital,
            max_position_size=capital * (self.strategy.risk_per_trade / 100),
            active_positions=0,
            daily_loss=daily_loss,
            max_daily_loss=max_daily_loss
        )
        
        # Track equity
        self.equity_curve = [(df.index[0], capital)]
        
        # Process each bar
        for i in range(len(df)):
            current_bar = df.iloc[i]
            current_time = df.index[i]
            
            # Reset daily loss at start of day
            if i > 0 and current_time.date() != df.index[i-1].date():
                portfolio.daily_loss = 0.0
            
            # Check for new signals only if no active position
            if self.active_position is None:
                if current_bar['signal_long']:
                    setup = self._get_setup_from_signal(df, i, TradeDirection.LONG)
                    if setup and self.strategy.validate_setup(setup, portfolio):
                        self._open_position(setup, current_bar, portfolio)
                
                elif current_bar['signal_short']:
                    setup = self._get_setup_from_signal(df, i, TradeDirection.SHORT)
                    if setup and self.strategy.validate_setup(setup, portfolio):
                        self._open_position(setup, current_bar, portfolio)
            
            # Manage active position
            else:
                self._manage_position(current_bar, portfolio)
            
            # Update equity curve
            if i % 96 == 0:  # Every day (96 15-min bars)
                current_equity = portfolio.equity
                if self.active_position:
                    # Add unrealized PnL
                    current_price = current_bar['close']
                    unrealized_pnl = self._calculate_unrealized_pnl(current_price)
                    current_equity += unrealized_pnl
                
                self.equity_curve.append((current_time, current_equity))
        
        # Close any remaining position
        if self.active_position:
            final_bar = df.iloc[-1]
            self._close_position(final_bar, "End of data", portfolio)
        
        # Calculate results
        return self._calculate_results()
    
    def _get_setup_from_signal(self, df: pd.DataFrame, index: int, direction: TradeDirection) -> Optional[TradeSetup]:
        """Extract setup from signal data"""
        current = df.iloc[index]
        
        entry_price = current['close']
        
        if direction == TradeDirection.LONG:
            sl = entry_price - (self.strategy.sl_pips * 0.01)
            tp1 = entry_price + (self.strategy.tp1_pips * 0.01)
            tp2 = entry_price + (self.strategy.tp2_pips * 0.01)
        else:
            sl = entry_price + (self.strategy.sl_pips * 0.01)
            tp1 = entry_price - (self.strategy.tp1_pips * 0.01)
            tp2 = entry_price - (self.strategy.tp2_pips * 0.01)
        
        signal_type_str = current.get('signal_type', 'unknown')
        
        try:
            if 'type1' in signal_type_str:
                signal_type = 'TYPE1_BOS_FVG_RSI'
            elif 'type2' in signal_type_str:
                signal_type = 'TYPE2_OB_LIQUIDITY'
            else:
                signal_type = 'TYPE3_RSI_DIVERGENCE'
        except:
            signal_type = 'UNKNOWN'
        
        return TradeSetup(
            signal_type=signal_type,
            direction=direction,
            entry_price=entry_price,
            stop_loss=sl,
            take_profit1=tp1,
            take_profit2=tp2,
            confluence_score=int(current.get('confluence_score', 0)),
            setup_timestamp=current.name,
            reason=current.get('signal_reason', 'Signal generated')
        )
    
    def _open_position(self, setup: TradeSetup, bar: pd.Series, portfolio: PortfolioState):
        """Open a new position"""
        # Apply slippage and spread
        slippage_cost = self.slippage_pips * 0.01
        spread_cost = self.spread_pips * 0.01
        
        if setup.direction == TradeDirection.LONG:
            entry_price = setup.entry_price + slippage_cost + spread_cost
        else:
            entry_price = setup.entry_price - slippage_cost - spread_cost
        
        # Calculate position size
        position_size = self.strategy.calculate_position_size(
            portfolio.equity, entry_price, setup.stop_loss
        )
        
        if position_size <= 0:
            logger.warning(f"Invalid position size: {position_size}")
            return
        
        # Calculate commissions
        commission = entry_price * position_size * self.commission_rate
        
        # Update portfolio
        portfolio.equity -= commission
        portfolio.active_positions += 1
        
        # Create trade record
        trade = TradeRecord(
            entry_time=bar.name,
            direction=setup.direction,
            signal_type=setup.signal_type,
            entry_price=entry_price,
            position_size=position_size,
            stop_loss=setup.stop_loss,
            take_profit1=setup.take_profit1,
            take_profit2=setup.take_profit2,
            confluence_score=setup.confluence_score,
            commissions=commission,
            slippage=slippage_cost
        )
        
        self.active_position = {
            'trade': trade,
            'partial_tp1_closed': False,
            'units_closed': 0
        }
        
        logger.info(f"Opened {setup.direction.value} position at {entry_price:.2f}")
    
    def _manage_position(self, bar: pd.Series, portfolio: PortfolioState):
        """Manage active position"""
        position = self.active_position
        trade = position['trade']
        
        high = bar['high']
        low = bar['low']
        
        # Check stop loss
        if trade.direction == TradeDirection.LONG:
            if low <= trade.stop_loss:
                exit_price = trade.stop_loss - (self.slippage_pips * 0.01)  # Slippage on exit
                self._close_position(bar, "Stop Loss", portfolio, exit_price)
                return
        else:  # SHORT
            if high >= trade.stop_loss:
                exit_price = trade.stop_loss + (self.slippage_pips * 0.01)
                self._close_position(bar, "Stop Loss", portfolio, exit_price)
                return
        
        # Check TP1 (first 50%)
        if not position['partial_tp1_closed']:
            if trade.direction == TradeDirection.LONG:
                if high >= trade.take_profit1:
                    self._partial_close(bar, portfolio, 0.5, "TP1")
                    position['partial_tp1_closed'] = True
                    position['units_closed'] = 0.5
            else:
                if low <= trade.take_profit1:
                    self._partial_close(bar, portfolio, 0.5, "TP1")
                    position['partial_tp1_closed'] = True
                    position['units_closed'] = 0.5
        
        # Check TP2 (second 50%)
        elif position['partial_tp1_closed']:
            if trade.direction == TradeDirection.LONG:
                if high >= trade.take_profit2:
                    self._close_position(bar, "TP2", portfolio, trade.take_profit2)
            else:
                if low <= trade.take_profit2:
                    self._close_position(bar, "TP2", portfolio, trade.take_profit2)
    
    def _partial_close(self, bar: pd.Series, portfolio: PortfolioState, fraction: float, reason: str):
        """Partially close position"""
        position = self.active_position
        trade = position['trade']
        
        # Calculate exit with slippage
        target_price = trade.take_profit1
        if trade.direction == TradeDirection.LONG:
            exit_price = target_price - (self.slippage_pips * 0.01)
        else:
            exit_price = target_price + (self.slippage_pips * 0.01)
        
        # Calculate PnL for closed portion
        closed_size = trade.position_size * fraction
        entry_value = trade.entry_price * closed_size
        exit_value = exit_price * closed_size
        
        if trade.direction == TradeDirection.LONG:
            pnl = exit_value - entry_value
        else:
            pnl = entry_value - exit_value
        
        # Apply commissions
        commission = exit_value * self.commission_rate
        pnl -= commission
        
        # Update portfolio
        portfolio.equity += pnl
        portfolio.daily_loss += min(pnl, 0)
        
        logger.info(f"Partial close ({fraction*100:.0f}%): PnL=${pnl:.2f}")
    
    def _close_position(self, bar: pd.Series, reason: str, portfolio: PortfolioState, forced_exit_price: Optional[float] = None):
        """Close active position"""
        position = self.active_position
        trade = position['trade']
        
        # Determine exit price
        if forced_exit_price:
            exit_price = forced_exit_price
        else:
            if trade.direction == TradeDirection.LONG:
                exit_price = bar['close'] - (self.slippage_pips * 0.01)
            else:
                exit_price = bar['close'] + (self.slippage_pips * 0.01)
        
        # Calculate PnL
        entry_value = trade.entry_price * trade.position_size
        exit_value = exit_price * trade.position_size
        
        if trade.direction == TradeDirection.LONG:
            pnl = exit_value - entry_value
        else:
            pnl = entry_value - exit_value
        
        # Apply commissions
        commission = exit_value * self.commission_rate
        pnl -= commission
        
        # Calculate risk/reward
        risk = abs(trade.entry_price - trade.stop_loss) * trade.position_size
        reward = max(pnl, 0)
        risk_reward = reward / risk if risk > 0 else 0
        
        # Update portfolio
        portfolio.equity += pnl
        portfolio.active_positions -= 1
        portfolio.daily_loss += min(pnl, 0)
        
        # Create closed trade record
        closed_trade = TradeRecord(
            entry_time=trade.entry_time,
            exit_time=bar.name,
            direction=trade.direction,
            signal_type=trade.signal_type,
            entry_price=trade.entry_price,
            position_size=trade.position_size,
            stop_loss=trade.stop_loss,
            take_profit1=trade.take_profit1,
            take_profit2=trade.take_profit2,
            exit_price=exit_price,
            exit_reason=reason,
            pnl=pnl,
            pnl_percentage=(pnl / entry_value) * 100,
            risk_reward=risk_reward,
            confluence_score=trade.confluence_score,
            commissions=trade.commissions + commission,
            slippage=trade.slippage
        )
        
        self.trades.append(closed_trade)
        self.active_position = None
        
        logger.info(f"Closed {trade.direction.value} position: PnL=${pnl:.2f} ({reason})")
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL for active position"""
        if not self.active_position:
            return 0
        
        trade = self.active_position['trade']
        
        entry_value = trade.entry_price * trade.position_size
        current_value = current_price * trade.position_size
        
        if trade.direction == TradeDirection.LONG:
            return current_value - entry_value
        else:
            return entry_value - current_value
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest metrics"""
        if not self.trades:
            logger.warning("No trades executed in backtest")
            return self._empty_result()
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame([trade.model_dump() for trade in self.trades])
        
        # Basic stats
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL stats
        total_pnl = trades_df['pnl'].sum()
        winning_pnl = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        losing_pnl = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')
        
        # Calculate equity curve
        equity_data = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        equity_data['drawdown'] = equity_data['equity'] / equity_data['equity'].cummax() - 1
        max_drawdown = equity_data['drawdown'].min()
        max_drawdown_pct = abs(max_drawdown) * 100
        
        # Average metrics
        avg_trade_pnl = trades_df['pnl'].mean()
        avg_winning_trade = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
        avg_losing_trade = trades_df[trades_df['pnl'] < 0]['pnl'].mean()
        avg_risk_reward = trades_df['risk_reward'].mean()
        
        # Trades per week
        start_date = trades_df['entry_time'].min()
        end_date = trades_df['entry_time'].max()
        weeks = (end_date - start_date).days / 7
        trades_per_week = total_trades / weeks if weeks > 0 else 0
        
        # Sharpe ratio (assuming risk-free rate = 0)
        returns = pd.Series([trade.pnl for trade in self.trades if trade.pnl is not None])
        if len(returns) > 1 and returns.std() != 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Recovery factor
        recovery_factor = total_pnl / (max_drawdown_pct * 100) if max_drawdown_pct > 0 else 0
        
        # Final capital
        final_capital = self.initial_capital + total_pnl
        total_return_pct = (final_capital / self.initial_capital - 1) * 100
        
        return BacktestResult(
            start_date=trades_df['entry_time'].min(),
            end_date=trades_df['entry_time'].max(),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown_pct * 100,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            recovery_factor=recovery_factor,
            avg_trade_pnl=avg_trade_pnl,
            avg_winning_trade=avg_winning_trade,
            avg_losing_trade=avg_losing_trade,
            trades_per_week=trades_per_week,
            avg_risk_reward=avg_risk_reward,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return_pct=total_return_pct
        )
    
    def _empty_result(self) -> BacktestResult:
        """Return empty backtest result"""
        current_time = datetime.now()
        return BacktestResult(
            start_date=current_time,
            end_date=current_time,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            total_pnl=0,
            profit_factor=0,
            max_drawdown=0,
            max_drawdown_pct=0,
            sharpe_ratio=0,
            recovery_factor=0,
            avg_trade_pnl=0,
            avg_winning_trade=0,
            avg_losing_trade=0,
            trades_per_week=0,
            avg_risk_reward=0,
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_return_pct=0
        )
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades as DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([trade.model_dump() for trade in self.trades])
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        if not self.equity_curve:
            return pd.DataFrame()
        
        return pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])