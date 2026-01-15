"""
XAU1 Live Risk Manager
Real-time risk management for paper trading
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk management limits configuration"""
    max_daily_loss_pct: float = 2.0  # 2% max daily loss
    max_weekly_loss_pct: float = 6.0  # 6% max weekly loss
    max_drawdown_pct: float = 10.0   # 10% max drawdown
    min_win_rate_window: int = 10     # Minimum trades for win rate calculation
    min_win_rate_threshold: float = 0.35  # 35% minimum win rate
    max_consecutive_losses: int = 4   # Maximum consecutive losses
    min_time_between_trades: int = 120  # Minimum minutes between trades
    max_positions: int = 2             # Maximum concurrent positions
    position_size_pct: float = 1.0    # 1% per trade


class LiveRiskManager:
    """Live risk management system"""
    
    def __init__(self, initial_capital: float = 10000, risk_limits: RiskLimits = None):
        """
        Initialize live risk manager
        
        Args:
            initial_capital: Starting capital amount
            risk_limits: Risk limits configuration
        """
        self.initial_capital = initial_capital
        self.risk_limits = risk_limits or RiskLimits()
        
        # Portfolio tracking
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
        
        # Daily tracking
        self.daily_start_capital = initial_capital
        self.daily_pnl = 0.0
        self.daily_start_time = datetime.now().date()
        
        # Weekly tracking
        self.week_start_capital = initial_capital
        self.weekly_pnl = 0.0
        self.week_start_date = datetime.now().date()
        
        # Trade tracking
        self.trades_history = []
        self.consecutive_losses = 0
        self.last_trade_time = None
        
        # Risk state flags
        self.risk_flags = {
            'daily_loss_limit': False,
            'weekly_loss_limit': False,
            'max_drawdown': False,
            'low_win_rate': False,
            'too_many_losses': False,
            'insufficient_capital': False,
            'too_many_positions': False,
            'too_frequent_trades': False
        }
        
        logger.info(f"LiveRiskManager initialized with ${initial_capital:,.2f} capital")
    
    def check_position_limits(self, new_signal: Dict = None, current_positions: int = 0) -> Dict:
        """
        Check if new position meets risk limits
        
        Args:
            new_signal: Signal data for new position
            current_positions: Current number of open positions
            
        Returns:
            Dict with check results
        """
        try:
            # Reset risk flags
            for flag in self.risk_flags:
                self.risk_flags[flag] = False
            
            # Update current capital
            self._update_portfolio_metrics()
            
            # Check 1: Maximum positions
            if current_positions >= self.risk_limits.max_positions:
                self.risk_flags['too_many_positions'] = True
                return {
                    'approved': False,
                    'reason': f"Maximum positions reached ({current_positions}/{self.risk_limits.max_positions})",
                    'risk_flags': self.risk_flags.copy()
                }
            
            # Check 2: Daily loss limit
            if self.daily_pnl <= -(self.daily_start_capital * self.risk_limits.max_daily_loss_pct / 100):
                self.risk_flags['daily_loss_limit'] = True
                return {
                    'approved': False,
                    'reason': f"Daily loss limit reached: ${self.daily_pnl:.2f}",
                    'risk_flags': self.risk_flags.copy()
                }
            
            # Check 3: Weekly loss limit
            if self.weekly_pnl <= -(self.week_start_capital * self.risk_limits.max_weekly_loss_pct / 100):
                self.risk_flags['weekly_loss_limit'] = True
                return {
                    'approved': False,
                    'reason': f"Weekly loss limit reached: ${self.weekly_pnl:.2f}",
                    'risk_flags': self.risk_flags.copy()
                }
            
            # Check 4: Maximum drawdown
            if self.current_drawdown >= self.risk_limits.max_drawdown_pct:
                self.risk_flags['max_drawdown'] = True
                return {
                    'approved': False,
                    'reason': f"Maximum drawdown exceeded: {self.current_drawdown:.2f}%",
                    'risk_flags': self.risk_flags.copy()
                }
            
            # Check 5: Win rate threshold
            if len(self.trades_history) >= self.risk_limits.min_win_rate_window:
                recent_trades = self.trades_history[-self.risk_limits.min_win_rate_window:]
                win_rate = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0) / len(recent_trades)
                
                if win_rate < self.risk_limits.min_win_rate_threshold:
                    self.risk_flags['low_win_rate'] = True
                    return {
                        'approved': False,
                        'reason': f"Win rate below threshold: {win_rate:.2%} (min: {self.risk_limits.min_win_rate_threshold:.2%})",
                        'risk_flags': self.risk_flags.copy()
                    }
            
            # Check 6: Consecutive losses
            if self.consecutive_losses >= self.risk_limits.max_consecutive_losses:
                self.risk_flags['too_many_losses'] = True
                return {
                    'approved': False,
                    'reason': f"Too many consecutive losses: {self.consecutive_losses}",
                    'risk_flags': self.risk_flags.copy()
                }
            
            # Check 7: Time between trades
            if (self.last_trade_time and 
                (datetime.now() - self.last_trade_time).seconds < self.risk_limits.min_time_between_trades * 60):
                self.risk_flags['too_frequent_trades'] = True
                return {
                    'approved': False,
                    'reason': f"Too soon since last trade: {self.last_trade_time}",
                    'risk_flags': self.risk_flags.copy()
                }
            
            # Check 8: Capital requirements
            estimated_cost = self._estimate_position_cost(new_signal)
            if self.current_capital < estimated_cost:
                self.risk_flags['insufficient_capital'] = True
                return {
                    'approved': False,
                    'reason': f"Insufficient capital: ${self.current_capital:.2f} < ${estimated_cost:.2f}",
                    'risk_flags': self.risk_flags.copy()
                }
            
            # All checks passed
            return {
                'approved': True,
                'reason': "All risk checks passed",
                'risk_flags': self.risk_flags.copy(),
                'current_drawdown': self.current_drawdown,
                'daily_pnl': self.daily_pnl,
                'weekly_pnl': self.weekly_pnl
            }
            
        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            return {
                'approved': False,
                'reason': f"Risk check error: {str(e)}",
                'risk_flags': {'error': True}
            }
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, direction: str = 'long') -> Dict:
        """
        Calculate optimal position size based on risk management
        
        Args:
            entry_price: Entry price for the position
            stop_loss: Stop loss price
            direction: Position direction ('long' or 'short')
            
        Returns:
            Dict with position size information
        """
        try:
            # Calculate risk amount (1% of current capital)
            risk_amount = self.current_capital * (self.risk_limits.position_size_pct / 100)
            
            # Calculate risk per unit
            risk_per_unit = abs(entry_price - stop_loss)
            
            if risk_per_unit == 0:
                return {
                    'position_size': 0,
                    'risk_amount': 0,
                    'reason': 'Invalid stop loss distance'
                }
            
            # Calculate position size based on risk
            position_size = risk_amount / risk_per_unit
            
            # Apply position size limits
            max_position_value = self.current_capital * 0.20  # Max 20% per position
            max_size = max_position_value / entry_price
            
            final_size = min(position_size, max_size)
            
            # Calculate actual risk
            actual_risk = final_size * risk_per_unit
            actual_risk_pct = (actual_risk / self.current_capital) * 100
            
            logger.info(f"Position size calculated: {final_size:.4f} units")
            logger.info(f"Risk: ${actual_risk:.2f} ({actual_risk_pct:.2f}% of capital)")
            
            return {
                'position_size': final_size,
                'risk_amount': actual_risk,
                'risk_percentage': actual_risk_pct,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'max_loss': actual_risk,
                'max_position_value': final_size * entry_price,
                'position_pct_of_capital': (final_size * entry_price / self.current_capital) * 100
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {
                'position_size': 0,
                'error': str(e)
            }
    
    def update_trade_result(self, trade_result: Dict):
        """
        Update risk manager with completed trade result
        
        Args:
            trade_result: Dict with trade result information
        """
        try:
            # Add to history
            trade_result['timestamp'] = datetime.now()
            self.trades_history.append(trade_result)
            
            # Update P&L
            if 'pnl' in trade_result and trade_result['pnl'] is not None:
                pnl = trade_result['pnl']
                
                # Update daily P&L
                self.daily_pnl += pnl
                
                # Update weekly P&L
                self.weekly_pnl += pnl
                
                # Update capital
                self.current_capital += pnl
                
                # Update peak capital and drawdown
                if self.current_capital > self.peak_capital:
                    self.peak_capital = self.current_capital
                
                self.current_drawdown = ((self.peak_capital - self.current_capital) / self.peak_capital) * 100
                
                # Update consecutive losses
                if pnl < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0
                
                # Update last trade time
                self.last_trade_time = datetime.now()
                
                # Reset daily/weekly counters if needed
                self._check_time_periods()
                
                logger.info(f"Trade result updated: PnL=${pnl:.2f}, Daily PnL=${self.daily_pnl:.2f}")
                logger.info(f"Current capital: ${self.current_capital:.2f}, Drawdown: {self.current_drawdown:.2f}%")
            
        except Exception as e:
            logger.error(f"Error updating trade result: {e}")
    
    def get_risk_status(self) -> Dict:
        """Get current risk status"""
        self._update_portfolio_metrics()
        
        # Calculate recent win rate
        recent_trades = self.trades_history[-10:] if self.trades_history else []
        recent_win_rate = 0
        if recent_trades:
            recent_win_rate = sum(1 for t in recent_trades if t.get('pnl', 0) > 0) / len(recent_trades)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'capital': {
                'initial': self.initial_capital,
                'current': self.current_capital,
                'peak': self.peak_capital,
                'total_return_pct': ((self.current_capital / self.initial_capital) - 1) * 100,
                'current_drawdown_pct': self.current_drawdown
            },
            'daily': {
                'start_capital': self.daily_start_capital,
                'pnl': self.daily_pnl,
                'pnl_pct': (self.daily_pnl / self.daily_start_capital) * 100,
                'loss_limit': self.daily_start_capital * self.risk_limits.max_daily_loss_pct / 100
            },
            'weekly': {
                'start_capital': self.week_start_capital,
                'pnl': self.weekly_pnl,
                'pnl_pct': (self.weekly_pnl / self.week_start_capital) * 100,
                'loss_limit': self.week_start_capital * self.risk_limits.max_weekly_loss_pct / 100
            },
            'trading': {
                'total_trades': len(self.trades_history),
                'consecutive_losses': self.consecutive_losses,
                'recent_win_rate': recent_win_rate,
                'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None
            },
            'risk_flags': self.risk_flags.copy(),
            'limits': {
                'max_daily_loss_pct': self.risk_limits.max_daily_loss_pct,
                'max_weekly_loss_pct': self.risk_limits.max_weekly_loss_pct,
                'max_drawdown_pct': self.risk_limits.max_drawdown_pct,
                'min_win_rate_threshold': self.risk_limits.min_win_rate_threshold,
                'max_consecutive_losses': self.risk_limits.max_consecutive_losses,
                'min_time_between_trades': self.risk_limits.min_time_between_trades,
                'max_positions': self.risk_limits.max_positions
            }
        }
    
    def should_trail_stop_loss(self, position: Dict, current_price: float) -> Dict:
        """
        Determine if stop loss should be moved to break-even or higher
        
        Args:
            position: Position information
            current_price: Current market price
            
        Returns:
            Dict with trailing stop recommendation
        """
        try:
            entry_price = position['entry_price']
            direction = position['direction']
            current_sl = position['stop_loss']
            
            # Calculate unrealized profit
            if direction == 'long':
                unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
                profit_pips = (current_price - entry_price) / 0.01
            else:
                unrealized_pnl_pct = ((entry_price - current_price) / entry_price) * 100
                profit_pips = (entry_price - current_price) / 0.01
            
            # Trailing stop rules
            trail_decision = {
                'should_trail': False,
                'new_stop_loss': None,
                'reason': ''
            }
            
            # Rule 1: Move to break-even after +30 pips profit
            if profit_pips >= 30:
                if direction == 'long':
                    new_sl = entry_price + 2  # +2 pips buffer
                else:
                    new_sl = entry_price - 2
                
                if (direction == 'long' and new_sl > current_sl) or (direction == 'short' and new_sl < current_sl):
                    trail_decision['should_trail'] = True
                    trail_decision['new_stop_loss'] = new_sl
                    trail_decision['reason'] = f'Move to break-even (+{profit_pips:.1f} pips profit)'
            
            # Rule 2: Move to TP1 after +50 pips profit (if using partial close)
            if profit_pips >= 50:
                if direction == 'long':
                    new_sl = entry_price + 15  # Move closer to entry
                else:
                    new_sl = entry_price - 15
                
                if (direction == 'long' and new_sl > current_sl) or (direction == 'short' and new_sl < current_sl):
                    # Only trail if we haven't already moved to break-even
                    if not trail_decision['should_trail']:
                        trail_decision['should_trail'] = True
                        trail_decision['new_stop_loss'] = new_sl
                        trail_decision['reason'] = f'Tighten stop after +{profit_pips:.1f} pips profit'
            
            return trail_decision
            
        except Exception as e:
            logger.error(f"Error in trailing stop calculation: {e}")
            return {
                'should_trail': False,
                'error': str(e)
            }
    
    def _update_portfolio_metrics(self):
        """Update portfolio metrics"""
        # Update drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        self.current_drawdown = ((self.peak_capital - self.current_capital) / self.peak_capital) * 100
    
    def _check_time_periods(self):
        """Check if daily/weekly periods should be reset"""
        current_date = datetime.now().date()
        current_time = datetime.now()
        
        # Check daily reset
        if current_date > self.daily_start_time:
            self.daily_start_time = current_date
            self.daily_start_capital = self.current_capital
            self.daily_pnl = 0.0
        
        # Check weekly reset (Monday)
        if current_date > self.week_start_date and current_date.weekday() == 0:
            self.week_start_date = current_date
            self.week_start_capital = self.current_capital
            self.weekly_pnl = 0.0
    
    def _estimate_position_cost(self, signal: Dict) -> float:
        """Estimate total cost of opening position"""
        if not signal:
            return 1000.0  # Conservative estimate
        
        entry_price = signal.get('entry_price', 2000)
        
        # Estimate 10% of position value as buffer
        estimated_position_value = 1000  # Conservative estimate
        estimated_cost = estimated_position_value * 0.1  # 10% buffer
        
        return estimated_cost
    
    def reset_risk_flags(self):
        """Manually reset all risk flags"""
        for flag in self.risk_flags:
            self.risk_flags[flag] = False
        
        logger.info("Risk flags manually reset")
    
    def update_risk_limits(self, new_limits: RiskLimits):
        """Update risk limits"""
        self.risk_limits = new_limits
        logger.info(f"Risk limits updated: {new_limits}")


def main():
    """Test the live risk manager"""
    # Create risk manager
    risk_manager = LiveRiskManager(initial_capital=10000)
    
    print("Testing Live Risk Manager...")
    
    # Test position approval
    test_signal = {
        'entry_price': 2025.50,
        'direction': 'long',
        'confluence_score': 4
    }
    
    result = risk_manager.check_position_limits(test_signal, current_positions=1)
    print(f"Position approval: {result}")
    
    # Test position sizing
    position_info = risk_manager.calculate_position_size(2025.50, 2000.50, 'long')
    print(f"Position sizing: {position_info}")
    
    # Get risk status
    status = risk_manager.get_risk_status()
    print(f"Risk status: {status}")
    
    # Simulate a trade
    trade_result = {
        'trade_id': 'TEST_001',
        'pnl': 125.50,
        'exit_reason': 'TP1'
    }
    
    risk_manager.update_trade_result(trade_result)
    
    # Get updated status
    updated_status = risk_manager.get_risk_status()
    print(f"Updated status: {updated_status}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()