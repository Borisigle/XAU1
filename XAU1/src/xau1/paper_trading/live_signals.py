"""
XAU1 Live Signal Generator
Real-time signal generation using live market data
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from xau1.engine.strategy import XAU1Strategy
from xau1.engine.indicators import SMCIndicators
from xau1.paper_trading.binance_connector import PaperBinanceConnector
from xau1.utils.data import prepare_market_data

logger = logging.getLogger(__name__)


class LiveSignalGenerator:
    """Real-time signal generation engine"""
    
    def __init__(self, strategy_config: Dict):
        """
        Initialize live signal generator
        
        Args:
            strategy_config: Strategy configuration dictionary
        """
        self.strategy_config = strategy_config
        self.strategy = XAU1Strategy(strategy_config)
        self.binance = PaperBinanceConnector()
        
        # Data storage
        self.price_data_15m = []
        self.price_data_1h = []
        self.price_data_4h = []
        
        # Signal tracking
        self.last_signal_time = None
        self.signals_today = 0
        self.daily_signal_limit = 5  # Max signals per day
        
        # State tracking
        self.is_running = False
        self.last_update_time = None
        
        # Market session tracking
        self.current_session = None
        self.friday_cutoff_passed = False
        
        logger.info("LiveSignalGenerator initialized")
    
    async def start_live_signals(self):
        """Start live signal generation"""
        logger.info("🚀 Starting live signal generation...")
        self.is_running = True
        
        try:
            while self.is_running:
                await self._process_market_update()
                await asyncio.sleep(60)  # Check every minute
                
        except Exception as e:
            logger.error(f"Error in live signal loop: {e}")
        finally:
            self.is_running = False
            logger.info("Live signal generation stopped")
    
    def stop_live_signals(self):
        """Stop live signal generation"""
        logger.info("🛑 Stopping live signal generation...")
        self.is_running = False
    
    async def _process_market_update(self):
        """Process market data update and check for signals"""
        try:
            # Get latest market data
            current_bar = await self.binance.get_latest_15m_bar()
            
            if not current_bar:
                logger.debug("No new market data available")
                return
            
            # Check if new bar (every 15 minutes)
            if not self._is_new_bar(current_bar):
                return
            
            logger.info(f"📊 Processing new 15m bar: {current_bar['timestamp']}")
            
            # Add to price data
            self.price_data_15m.append(current_bar)
            
            # Keep only recent data
            self._trim_price_data()
            
            # Update market sessions
            self._update_market_sessions(current_bar['timestamp'])
            
            # Check daily signal limit
            if self._should_reset_daily_counters(current_bar['timestamp']):
                self.signals_today = 0
            
            # Only process signals during active sessions
            if not self._is_trading_session_active(current_bar['timestamp']):
                return
            
            # Generate signal if we have enough data
            if len(self.price_data_15m) >= 100:  # Need sufficient history
                signal = await self._generate_signal()
                
                if signal:
                    await self._handle_new_signal(signal)
            
            self.last_update_time = current_bar['timestamp']
            
        except Exception as e:
            logger.error(f"Error processing market update: {e}")
    
    def _is_new_bar(self, current_bar: Dict) -> bool:
        """Check if this is a new 15-minute bar"""
        if not self.last_update_time:
            return True
        
        last_bar_time = self.last_update_time
        current_bar_time = current_bar['timestamp']
        
        # Check if it's been at least 15 minutes
        return (current_bar_time - last_bar_time).seconds >= 900
    
    def _trim_price_data(self):
        """Keep only recent price data to manage memory"""
        # Keep last 1000 bars for each timeframe
        max_bars = 1000
        
        for data_list in [self.price_data_15m, self.price_data_1h, self.price_data_4h]:
            if len(data_list) > max_bars:
                del data_list[:len(data_list) - max_bars]
    
    def _update_market_sessions(self, timestamp: datetime):
        """Update current trading session"""
        hour = timestamp.hour
        
        # London session: 13:00-20:00 UTC
        if 13 <= hour < 20:
            self.current_session = 'london'
        # New York session: 13:30-21:00 UTC  
        elif 13.5 <= hour < 21:
            self.current_session = 'newyork'
        else:
            self.current_session = None
        
        # Check Friday cutoff (18:00 UTC)
        if timestamp.weekday() == 4 and hour >= 18:  # Friday and after 18:00
            self.friday_cutoff_passed = True
        elif timestamp.weekday() < 4:  # Monday-Thursday
            self.friday_cutoff_passed = False
    
    def _is_trading_session_active(self, timestamp: datetime) -> bool:
        """Check if trading session is active"""
        # Skip if Friday after 18:00
        if self.friday_cutoff_passed:
            return False
        
        # Check if in active trading session
        return self.current_session is not None
    
    def _should_reset_daily_counters(self, timestamp: datetime) -> bool:
        """Check if daily counters should be reset"""
        if not self.last_update_time:
            return False
        
        return timestamp.date() != self.last_update_time.date()
    
    async def _generate_signal(self) -> Optional[Dict]:
        """Generate trading signal based on current market data"""
        try:
            # Convert price data to DataFrame
            df_15m = self._prepare_dataframe(self.price_data_15m)
            
            if df_15m is None or len(df_15m) < 100:
                return None
            
            # Calculate indicators
            smc_indicators = SMCIndicators(df_15m)
            df_with_indicators = smc_indicators.calculate_all_indicators()
            
            # Add session information
            df_with_indicators = self._add_session_data(df_with_indicators)
            
            # Generate signals using strategy
            df_signals = self.strategy.generate_signals(df_with_indicators)
            
            # Get latest signal
            latest_signal = self._extract_latest_signal(df_signals)
            
            if latest_signal:
                logger.info(f"🎯 Signal detected: {latest_signal}")
                return latest_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    def _prepare_dataframe(self, price_data: List[Dict]) -> Optional[pd.DataFrame]:
        """Convert price data to DataFrame"""
        try:
            if not price_data:
                return None
            
            df = pd.DataFrame(price_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.warning("Missing required columns in price data")
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing DataFrame: {e}")
            return None
    
    def _add_session_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trading session information to DataFrame"""
        df = df.copy()
        
        # Add session flags
        df['active_session'] = False
        df['is_friday_after_18'] = False
        
        for idx, timestamp in enumerate(df.index):
            # Check active session
            if self._is_trading_session_active(timestamp):
                df.iloc[idx, df.columns.get_loc('active_session')] = True
            
            # Check Friday cutoff
            if timestamp.weekday() == 4 and timestamp.hour >= 18:
                df.iloc[idx, df.columns.get_loc('is_friday_after_18')] = True
        
        return df
    
    def _extract_latest_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Extract the latest signal from DataFrame"""
        try:
            if df is None or len(df) == 0:
                return None
            
            # Get latest row
            latest = df.iloc[-1]
            
            # Check for signals
            signal_data = {
                'timestamp': df.index[-1],
                'signal_long': latest.get('signal_long', False),
                'signal_short': latest.get('signal_short', False),
                'signal_type': latest.get('signal_type'),
                'signal_reason': latest.get('signal_reason'),
                'confluence_score': latest.get('confluence_score', 0),
                'entry_price': latest['close'],
                'rsi': latest.get('rsi', 50),
                'bos_bullish': latest.get('bos_bullish', False),
                'bos_bearish': latest.get('bos_bearish', False),
                'fvg_bullish': latest.get('fvg_bullish', False),
                'fvg_bearish': latest.get('fvg_bearish', False),
                'ob_bullish': latest.get('ob_bullish', False),
                'ob_bearish': latest.get('ob_bearish', False)
            }
            
            # Only return if there's an actual signal
            if signal_data['signal_long'] or signal_data['signal_short']:
                return signal_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting signal: {e}")
            return None
    
    async def _handle_new_signal(self, signal_data: Dict):
        """Handle new signal detection"""
        try:
            # Check daily signal limit
            if self.signals_today >= self.daily_signal_limit:
                logger.info(f"Daily signal limit reached ({self.signals_today}/{self.daily_signal_limit})")
                return
            
            # Check time since last signal (minimum 2 hours)
            if (self.last_signal_time and 
                (datetime.now() - self.last_signal_time).seconds < 7200):  # 2 hours
                logger.info("Too soon since last signal (minimum 2 hours required)")
                return
            
            # Create signal object
            signal = self._create_trade_signal(signal_data)
            
            if signal:
                logger.info(f"✅ New signal generated and ready for execution")
                logger.info(f"   Direction: {signal.direction.value.upper()}")
                logger.info(f"   Entry: {signal.entry_price:.2f}")
                logger.info(f"   Reason: {signal.reason}")
                
                # Update counters
                self.signals_today += 1
                self.last_signal_time = datetime.now()
                
                # Emit signal for execution
                await self._emit_signal(signal)
            
        except Exception as e:
            logger.error(f"Error handling new signal: {e}")
    
    def _create_trade_signal(self, signal_data: Dict):
        """Create TradeSetup object from signal data"""
        try:
            from xau1.engine.strategy import TradeSetup, TradeDirection, SignalType
            
            # Determine direction
            if signal_data['signal_long']:
                direction = TradeDirection.LONG
                sl_pips = self.strategy.sl_pips
                tp1_pips = self.strategy.tp1_pips
                tp2_pips = self.strategy.tp2_pips
                
                stop_loss = signal_data['entry_price'] - (sl_pips * 0.01)
                take_profit1 = signal_data['entry_price'] + (tp1_pips * 0.01)
                take_profit2 = signal_data['entry_price'] + (tp2_pips * 0.01)
                
            elif signal_data['signal_short']:
                direction = TradeDirection.SHORT
                sl_pips = self.strategy.sl_pips
                tp1_pips = self.strategy.tp1_pips
                tp2_pips = self.strategy.tp2_pips
                
                stop_loss = signal_data['entry_price'] + (sl_pips * 0.01)
                take_profit1 = signal_data['entry_price'] - (tp1_pips * 0.01)
                take_profit2 = signal_data['entry_price'] - (tp2_pips * 0.01)
            
            else:
                return None
            
            # Create signal type
            if 'type1' in str(signal_data['signal_type']):
                signal_type = SignalType.TYPE1_BOS_FVG_RSI
            elif 'type2' in str(signal_data['signal_type']):
                signal_type = SignalType.TYPE2_OB_LIQUIDITY
            elif 'type3' in str(signal_data['signal_type']):
                signal_type = SignalType.TYPE3_RSI_DIVERGENCE
            else:
                signal_type = SignalType.TYPE1_BOS_FVG_RSI  # Default
            
            signal = TradeSetup(
                signal_type=signal_type,
                direction=direction,
                entry_price=signal_data['entry_price'],
                stop_loss=stop_loss,
                take_profit1=take_profit1,
                take_profit2=take_profit2,
                confluence_score=signal_data['confluence_score'],
                setup_timestamp=signal_data['timestamp'],
                reason=signal_data['signal_reason']
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error creating trade signal: {e}")
            return None
    
    async def _emit_signal(self, signal):
        """Emit signal for execution (override this method)"""
        # This method should be overridden by the main application
        # to handle signal execution with the paper trader
        logger.info(f"📡 SIGNAL EMITTED: {signal.direction.value.upper()} at {signal.entry_price:.2f}")
        logger.info(f"   Stop Loss: {signal.stop_loss:.2f}")
        logger.info(f"   Take Profit 1: {signal.take_profit1:.2f}")
        logger.info(f"   Take Profit 2: {signal.take_profit2:.2f}")
        logger.info(f"   Confluence Score: {signal.confluence_score}")
        logger.info(f"   Reason: {signal.reason}")
    
    def get_status(self) -> Dict:
        """Get current status of the signal generator"""
        return {
            'is_running': self.is_running,
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
            'current_session': self.current_session,
            'signals_today': self.signals_today,
            'daily_limit': self.daily_signal_limit,
            'friday_cutoff_passed': self.friday_cutoff_passed,
            'data_points_15m': len(self.price_data_15m),
            'connection_status': self.binance.get_connection_status()
        }
    
    async def get_latest_market_data(self) -> Optional[Dict]:
        """Get latest market data snapshot"""
        try:
            if not self.price_data_15m:
                return None
            
            latest = self.price_data_15m[-1]
            
            # Add recent signal info
            latest['has_recent_signal'] = (
                self.last_signal_time and 
                (datetime.now() - self.last_signal_time).seconds < 3600  # Last hour
            )
            
            return latest
            
        except Exception as e:
            logger.error(f"Error getting latest market data: {e}")
            return None


# Example usage and testing
async def main():
    """Test the live signal generator"""
    # Example strategy config
    strategy_config = {
        'strategy': {
            'name': 'XAU1 SMC + Order Flow',
            'symbol': 'XAUUSDT'
        },
        'risk_management': {
            'position_size_percentage': 1.0,
            'stop_loss_pips': 30,
            'take_profit1_pips': 50,
            'take_profit2_pips': 100,
            'min_risk_reward_ratio': 2.0
        },
        'entry_rules': {
            'type1_bos_fvg_rsi': {'enabled': True, 'min_confluence': 3},
            'type2_ob_liquidity': {'enabled': True, 'min_confluence': 3},
            'type3_rsi_divergence': {'enabled': True, 'min_confluence': 2}
        }
    }
    
    # Create signal generator
    generator = LiveSignalGenerator(strategy_config)
    
    print("Testing Live Signal Generator...")
    print("Starting signal generation for 5 minutes...")
    
    # Start signal generation
    task = asyncio.create_task(generator.start_live_signals())
    
    # Run for 5 minutes then stop
    await asyncio.sleep(300)  # 5 minutes
    
    # Stop
    generator.stop_live_signals()
    task.cancel()
    
    # Get status
    status = generator.get_status()
    print(f"\nFinal Status: {status}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(main())