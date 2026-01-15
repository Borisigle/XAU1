"""
XAU1 Paper Trading Engine
Real-time paper trading with live Binance data simulation
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from pydantic import BaseModel

from xau1.engine.strategy import (
    TradeSetup, 
    PortfolioState, 
    TradeDirection, 
    SignalType
)

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Open position tracking"""
    trade_id: str
    symbol: str
    direction: TradeDirection
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit1: float
    take_profit2: float
    entry_time: datetime
    partial_tp1_closed: bool = False
    units_closed: float = 0.0
    unrealized_pnl: float = 0.0
    status: str = "OPEN"


@dataclass
class TradeRecord:
    """Completed trade record"""
    trade_id: str
    symbol: str
    direction: TradeDirection
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    entry_time: datetime
    exit_time: Optional[datetime]
    stop_loss: float
    take_profit1: float
    take_profit2: float
    pnl: Optional[float]
    pnl_percentage: Optional[float]
    exit_reason: str
    commissions: float = 0.0
    slippage: float = 0.0
    confluence_score: int = 0


class PaperTrader:
    """Paper trading engine with live market simulation"""
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 commission_rate: float = 0.0002,
                 slippage_pips: float = 1.0,
                 spread_pips: float = 0.5):
        
        # Capital and accounts
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_capital = initial_capital
        
        # Trading parameters
        self.commission_rate = commission_rate
        self.slippage_pips = slippage_pips
        self.spread_pips = spread_pips
        
        # State tracking
        self.positions: Dict[str, Position] = {}
        self.trades_history: List[TradeRecord] = []
        self.daily_pnl: Dict[str, float] = {}
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # Portfolio state
        self.portfolio = PortfolioState(
            equity=initial_capital,
            max_position_size=initial_capital * 0.02,  # 2% max per position
            active_positions=0,
            daily_loss=0.0,
            max_daily_loss=initial_capital * 0.05  # 5% max daily loss
        )
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        self._setup_logging()
        
        logger.info(f"PaperTrader initialized with ${initial_capital:,.2f} initial capital")
    
    def _setup_logging(self):
        """Setup detailed paper trading logging"""
        log_file = f"logs/paper_trading_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        paper_trader_logger = logging.getLogger('paper_trader')
        paper_trader_logger.setLevel(logging.INFO)
        paper_trader_logger.addHandler(file_handler)
        paper_trader_logger.addHandler(console_handler)
    
    async def on_signal(self, signal: TradeSetup) -> bool:
        """
        Process incoming trade signal
        
        Args:
            signal: TradeSetup with entry details
            
        Returns:
            True if signal was executed, False if rejected
        """
        try:
            logger.info(f"📊 SIGNAL RECEIVED: {signal.direction.value.upper()} {signal.signal_type}")
            logger.info(f"   Entry: {signal.entry_price:.2f}")
            logger.info(f"   SL: {signal.stop_loss:.2f} ({abs(signal.entry_price - signal.stop_loss):.1f} pips)")
            logger.info(f"   TP1: {signal.take_profit1:.2f} | TP2: {signal.take_profit2:.2f}")
            logger.info(f"   Confluence Score: {signal.confluence_score}")
            logger.info(f"   Reason: {signal.reason}")
            
            # Check if signal meets criteria
            if not await self._validate_signal(signal):
                logger.info(f"   ❌ Signal REJECTED by validation rules")
                return False
            
            # Execute signal
            success = await self._execute_signal(signal)
            
            if success:
                logger.info(f"   ✅ Signal EXECUTED successfully")
                await self._log_execution_details(signal)
            else:
                logger.info(f"   ❌ Signal EXECUTION FAILED")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return False
    
    async def _validate_signal(self, signal: TradeSetup) -> bool:
        """Validate signal against risk management rules"""
        
        # Check max positions
        if len(self.positions) >= 2:
            logger.debug(f"Max positions reached ({len(self.positions)}/2)")
            return False
        
        # Check daily loss limit
        if self.portfolio.daily_loss >= self.portfolio.max_daily_loss:
            logger.debug(f"Daily loss limit reached: ${self.portfolio.daily_loss:.2f}")
            return False
        
        # Check capital availability
        estimated_cost = signal.entry_price * 0.1  # Rough estimate
        if self.available_capital < estimated_cost:
            logger.debug(f"Insufficient capital: ${self.available_capital:.2f}")
            return False
        
        # Check R/R ratio
        risk_pips = abs(signal.entry_price - signal.stop_loss)
        reward_pips = abs(signal.take_profit1 - signal.entry_price)
        rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
        
        if rr_ratio < 1.8:
            logger.debug(f"R/R ratio too low: {rr_ratio:.2f}")
            return False
        
        # Check confluence score
        if signal.confluence_score < 2:
            logger.debug(f"Confluence score too low: {signal.confluence_score}")
            return False
        
        return True
    
    async def _execute_signal(self, signal: TradeSetup) -> bool:
        """Execute the trading signal"""
        try:
            # Apply slippage and spread to entry price
            if signal.direction == TradeDirection.LONG:
                entry_price = signal.entry_price + (self.slippage_pips * 0.01) + (self.spread_pips * 0.01)
            else:
                entry_price = signal.entry_price - (self.slippage_pips * 0.01) - (self.spread_pips * 0.01)
            
            # Calculate position size (1% risk)
            position_size = self._calculate_position_size(entry_price, signal.stop_loss)
            
            if position_size <= 0:
                logger.error("Invalid position size calculated")
                return False
            
            # Generate trade ID
            trade_id = f"PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create position
            position = Position(
                trade_id=trade_id,
                symbol="XAUUSDT",
                direction=signal.direction,
                entry_price=entry_price,
                quantity=position_size,
                stop_loss=signal.stop_loss,
                take_profit1=signal.take_profit1,
                take_profit2=signal.take_profit2,
                entry_time=datetime.now()
            )
            
            # Calculate commission
            commission = entry_price * position_size * self.commission_rate
            
            # Update portfolio
            self.current_capital -= commission
            self.available_capital -= commission
            self.portfolio.active_positions += 1
            
            # Store position
            self.positions[trade_id] = position
            
            # Update statistics
            self.total_trades += 1
            
            logger.info(f"💰 POSITION OPENED: {position.direction.value.upper()} {trade_id}")
            logger.info(f"   Entry Price: {entry_price:.2f}")
            logger.info(f"   Quantity: {position_size:.4f}")
            logger.info(f"   Commission: ${commission:.2f}")
            logger.info(f"   Available Capital: ${self.available_capital:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False
    
    def _calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on 1% risk per trade"""
        risk_amount = self.current_capital * 0.01  # 1% risk
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0
        
        position_size = risk_amount / risk_per_unit
        
        # Cap at 20% of available capital
        max_position_value = self.available_capital * 0.2
        max_size = max_position_value / entry_price
        
        return min(position_size, max_size)
    
    async def on_tick(self, current_price: float, timestamp: datetime):
        """
        Process live price tick and update positions
        
        Args:
            current_price: Current market price
            timestamp: Current timestamp
        """
        try:
            # Update all open positions
            positions_to_close = []
            
            for trade_id, position in list(self.positions.items()):
                # Update unrealized P&L
                self._update_unrealized_pnl(position, current_price)
                
                # Check stop loss and take profit conditions
                exit_info = self._check_exit_conditions(position, current_price)
                
                if exit_info['should_close']:
                    positions_to_close.append((trade_id, exit_info))
            
            # Close positions that hit exit conditions
            for trade_id, exit_info in positions_to_close:
                await self._close_position(trade_id, exit_info, current_price, timestamp)
            
            # Update equity curve (daily)
            current_date = timestamp.date()
            if current_date not in self.daily_pnl:
                self.daily_pnl[current_date] = 0.0
            
            # Update equity curve every hour or if significant change
            if (not self.equity_curve or 
                (timestamp - self.equity_curve[-1][0]).seconds >= 3600 or
                abs(self._calculate_total_equity(current_price) - self.equity_curve[-1][1]) > 50):
                
                total_equity = self._calculate_total_equity(current_price)
                self.equity_curve.append((timestamp, total_equity))
                
                # Keep only last 30 days of equity curve
                cutoff_date = timestamp - timedelta(days=30)
                self.equity_curve = [(ts, eq) for ts, eq in self.equity_curve if ts >= cutoff_date]
            
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
    
    def _update_unrealized_pnl(self, position: Position, current_price: float):
        """Update unrealized P&L for position"""
        if position.direction == TradeDirection.LONG:
            unrealized = (current_price - position.entry_price) * position.quantity
        else:
            unrealized = (position.entry_price - current_price) * position.quantity
        
        position.unrealized_pnl = unrealized
    
    def _check_exit_conditions(self, position: Position, current_price: float) -> Dict:
        """Check if position should be closed"""
        
        # Check stop loss
        if position.direction == TradeDirection.LONG:
            if current_price <= position.stop_loss:
                return {
                    'should_close': True,
                    'exit_price': position.stop_loss,
                    'exit_reason': 'Stop Loss'
                }
        else:
            if current_price >= position.stop_loss:
                return {
                    'should_close': True,
                    'exit_price': position.stop_loss,
                    'exit_reason': 'Stop Loss'
                }
        
        # Check take profit levels
        if position.direction == TradeDirection.LONG:
            # TP1 hit
            if current_price >= position.take_profit1 and not position.partial_tp1_closed:
                return {
                    'should_close': False,  # Partial close
                    'exit_price': position.take_profit1,
                    'exit_reason': 'TP1 Partial'
                }
            
            # TP2 hit (or continue if TP1 already hit)
            if current_price >= position.take_profit2:
                if position.partial_tp1_closed:
                    return {
                        'should_close': True,
                        'exit_price': position.take_profit2,
                        'exit_reason': 'TP2 Full Close'
                    }
                else:
                    return {
                        'should_close': False,  # Full close without TP1
                        'exit_price': position.take_profit2,
                        'exit_reason': 'TP2 Direct'
                    }
        
        else:  # SHORT
            # TP1 hit
            if current_price <= position.take_profit1 and not position.partial_tp1_closed:
                return {
                    'should_close': False,
                    'exit_price': position.take_profit1,
                    'exit_reason': 'TP1 Partial'
                }
            
            # TP2 hit
            if current_price <= position.take_profit2:
                if position.partial_tp1_closed:
                    return {
                        'should_close': True,
                        'exit_price': position.take_profit2,
                        'exit_reason': 'TP2 Full Close'
                    }
                else:
                    return {
                        'should_close': False,
                        'exit_price': position.take_profit2,
                        'exit_reason': 'TP2 Direct'
                    }
        
        return {'should_close': False}
    
    async def _close_position(self, trade_id: str, exit_info: Dict, current_price: float, timestamp: datetime):
        """Close position and record trade"""
        try:
            position = self.positions[trade_id]
            
            # Calculate P&L
            if position.direction == TradeDirection.LONG:
                exit_price = position.entry_price if exit_info['exit_price'] > current_price else exit_info['exit_price']
                pnl = (exit_price - position.entry_price) * position.quantity
            else:
                exit_price = position.entry_price if exit_info['exit_price'] < current_price else exit_info['exit_price']
                pnl = (position.entry_price - exit_price) * position.quantity
            
            # Apply slippage on exit
            if exit_info['exit_reason'] in ['Stop Loss', 'TP2 Full Close', 'TP2 Direct']:
                if position.direction == TradeDirection.LONG:
                    exit_price -= self.slippage_pips * 0.01
                else:
                    exit_price += self.slippage_pips * 0.01
            
            # Recalculate P&L with adjusted price
            if position.direction == TradeDirection.LONG:
                pnl = (exit_price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - exit_price) * position.quantity
            
            # Apply commission on exit
            commission = exit_price * position.quantity * self.commission_rate
            pnl -= commission
            
            # Calculate P&L percentage
            entry_value = position.entry_price * position.quantity
            pnl_percentage = (pnl / entry_value) * 100
            
            # Update portfolio
            self.current_capital += pnl
            self.available_capital += pnl
            self.portfolio.active_positions -= 1
            self.portfolio.daily_loss += min(pnl, 0)
            
            # Update statistics
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Create trade record
            trade_record = TradeRecord(
                trade_id=trade_id,
                symbol=position.symbol,
                direction=position.direction,
                entry_price=position.entry_price,
                exit_price=exit_price,
                quantity=position.quantity,
                entry_time=position.entry_time,
                exit_time=timestamp,
                stop_loss=position.stop_loss,
                take_profit1=position.take_profit1,
                take_profit2=position.take_profit2,
                pnl=pnl,
                pnl_percentage=pnl_percentage,
                exit_reason=exit_info['exit_reason'],
                commissions=commission,
                confluence_score=0  # This would come from the original signal
            )
            
            self.trades_history.append(trade_record)
            
            # Remove position
            del self.positions[trade_id]
            
            # Log trade completion
            result_icon = "✅" if pnl > 0 else "❌"
            logger.info(f"{result_icon} POSITION CLOSED: {position.direction.value.upper()} {trade_id}")
            logger.info(f"   Exit Price: {exit_price:.2f}")
            logger.info(f"   P&L: ${pnl:.2f} ({pnl_percentage:.2f}%)")
            logger.info(f"   Reason: {exit_info['exit_reason']}")
            logger.info(f"   Current Capital: ${self.current_capital:.2f}")
            
        except Exception as e:
            logger.error(f"Error closing position {trade_id}: {e}")
    
    def _calculate_total_equity(self, current_price: float) -> float:
        """Calculate total account equity including unrealized P&L"""
        total_unrealized = 0
        
        for position in self.positions.values():
            if position.direction == TradeDirection.LONG:
                unrealized = (current_price - position.entry_price) * position.quantity
            else:
                unrealized = (position.entry_price - current_price) * position.quantity
            total_unrealized += unrealized
        
        return self.current_capital + total_unrealized
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_equity = self.current_capital + total_unrealized
        
        # Calculate win rate
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Calculate profit factor
        winning_pnl = sum(t.pnl for t in self.trades_history if t.pnl and t.pnl > 0)
        losing_pnl = abs(sum(t.pnl for t in self.trades_history if t.pnl and t.pnl < 0))
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_equity': total_equity,
            'total_return_pct': (total_equity / self.initial_capital - 1) * 100,
            'unrealized_pnl': total_unrealized,
            'available_capital': self.available_capital,
            'active_positions': len(self.positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'daily_pnl': dict(self.daily_pnl),
            'positions': [asdict(pos) for pos in self.positions.values()],
            'recent_trades': [asdict(trade) for trade in self.trades_history[-10:]]  # Last 10 trades
        }
    
    def save_portfolio_state(self, filepath: str = None):
        """Save current portfolio state to file"""
        if filepath is None:
            filepath = f"logs/portfolio_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        portfolio_data = self.get_portfolio_summary()
        
        with open(filepath, 'w') as f:
            json.dump(portfolio_data, f, indent=2, default=str)
        
        logger.info(f"Portfolio state saved to {filepath}")
    
    async def close_all_positions(self, reason: str = "Manual Close All"):
        """Close all open positions"""
        logger.info(f"🛑 CLOSING ALL POSITIONS: {reason}")
        
        positions_to_close = list(self.positions.keys())
        current_price = self.current_capital  # This would come from market data
        timestamp = datetime.now()
        
        for trade_id in positions_to_close:
            position = self.positions[trade_id]
            exit_info = {
                'should_close': True,
                'exit_price': current_price,
                'exit_reason': reason
            }
            await self._close_position(trade_id, exit_info, current_price, timestamp)
        
        logger.info(f"✅ All positions closed. Reason: {reason}")


def main():
    """Test the paper trading engine"""
    async def test_paper_trader():
        # Create paper trader
        trader = PaperTrader(initial_capital=10000)
        
        # Create test signal
        signal = TradeSetup(
            signal_type=SignalType.TYPE1_BOS_FVG_RSI,
            direction=TradeDirection.LONG,
            entry_price=2025.50,
            stop_loss=2000.50,
            take_profit1=2075.50,
            take_profit2=2125.50,
            confluence_score=4,
            setup_timestamp=datetime.now(),
            reason="Test signal"
        )
        
        # Execute signal
        await trader.on_signal(signal)
        
        # Simulate price movements
        prices = [2025.50, 2030.00, 2035.00, 2040.00, 2075.50, 2100.00, 2125.50]
        
        for price in prices:
            await trader.on_tick(price, datetime.now())
            print(f"Price: ${price:.2f}, P&L: ${trader.positions['PAPER_TEST'].unrealized_pnl:.2f}")
        
        # Get summary
        summary = trader.get_portfolio_summary()
        print("\nPortfolio Summary:")
        print(json.dumps(summary, indent=2, default=str))
    
    # Run test
    asyncio.run(test_paper_trader())


if __name__ == "__main__":
    main()