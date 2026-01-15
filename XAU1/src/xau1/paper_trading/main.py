#!/usr/bin/env python3
"""
XAU1 Paper Trading Main Script
Main entry point for live paper trading system
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, Optional

import yaml

# Import XAU1 modules
from xau1.paper_trading.paper_trader import PaperTrader
from xau1.paper_trading.live_signals import LiveSignalGenerator
from xau1.paper_trading.risk_manager import LiveRiskManager
from xau1.paper_trading.binance_connector import PaperBinanceConnector


class XAU1PaperTradingSystem:
    """Main paper trading system orchestrator"""
    
    def __init__(self, config_path: str = "src/xau1/config/optimized_strategy_params.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.paper_trader: Optional[PaperTrader] = None
        self.signal_generator: Optional[LiveSignalGenerator] = None
        self.risk_manager: Optional[LiveRiskManager] = None
        self.binance_connector: Optional[PaperBinanceConnector] = None
        
        # System state
        self.is_running = False
        self.system_tasks = []
        
        # Setup logging
        self._setup_logging()
        
        logger.info("XAU1 Paper Trading System initialized")
    
    def _load_config(self) -> Dict:
        """Load strategy configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Fallback to default config
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
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
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/xau1_paper_trading_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    async def start_system(self):
        """Start the complete paper trading system"""
        try:
            logger.info("🚀 Starting XAU1 Paper Trading System...")
            
            # Initialize components
            await self._initialize_components()
            
            # Start the main trading loop
            self.is_running = True
            await self._run_trading_loop()
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            await self.stop_system()
    
    async def _initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing system components...")
        
        # Initialize Binance connector
        self.binance_connector = PaperBinanceConnector()
        logger.info("✅ Binance connector initialized")
        
        # Initialize paper trader
        self.paper_trader = PaperTrader(
            initial_capital=10000,
            commission_rate=0.0002,
            slippage_pips=1.0,
            spread_pips=0.5
        )
        logger.info("✅ Paper trader initialized")
        
        # Initialize risk manager
        self.risk_manager = LiveRiskManager(initial_capital=10000)
        logger.info("✅ Risk manager initialized")
        
        # Initialize signal generator
        self.signal_generator = LiveSignalGenerator(self.config)
        
        # Override the signal emission to use our paper trader
        original_emit = self.signal_generator._emit_signal
        
        async def emit_signal_with_paper_trading(signal):
            """Emit signal and execute with paper trader"""
            logger.info(f"📡 Signal received: {signal.direction.value.upper()} at {signal.entry_price:.2f}")
            
            # Check with risk manager first
            risk_check = self.risk_manager.check_position_limits(
                current_positions=len(self.paper_trader.positions)
            )
            
            if not risk_check['approved']:
                logger.warning(f"Signal rejected by risk manager: {risk_check['reason']}")
                return
            
            # Execute signal with paper trader
            success = await self.paper_trader.on_signal(signal)
            
            if success:
                logger.info("✅ Signal executed successfully")
                # Update risk manager with the trade (simulated)
                trade_result = {
                    'signal': signal,
                    'status': 'executed'
                }
                # Note: Real P&L will be updated when position closes
            else:
                logger.warning("❌ Signal execution failed")
        
        # Replace the emit method
        self.signal_generator._emit_signal = emit_signal_with_paper_trading
        
        logger.info("✅ Signal generator initialized with paper trading integration")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("✅ All components initialized successfully")
    
    async def _run_trading_loop(self):
        """Main trading loop"""
        logger.info("🔄 Starting main trading loop...")
        
        # Create tasks
        self.system_tasks = [
            asyncio.create_task(self._signal_generation_loop()),
            asyncio.create_task(self._price_monitoring_loop()),
            asyncio.create_task(self._risk_monitoring_loop()),
            asyncio.create_task(self._portfolio_saving_loop())
        ]
        
        try:
            # Run all tasks concurrently
            await asyncio.gather(*self.system_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            logger.info("Main trading loop stopped")
    
    async def _signal_generation_loop(self):
        """Signal generation loop"""
        logger.info("📊 Starting signal generation loop...")
        
        try:
            await self.signal_generator.start_live_signals()
        except Exception as e:
            logger.error(f"Error in signal generation loop: {e}")
    
    async def _price_monitoring_loop(self):
        """Price monitoring and position management loop"""
        logger.info("💰 Starting price monitoring loop...")
        
        while self.is_running:
            try:
                # Get current price
                price_data = await self.binance_connector.get_live_prices()
                current_price = price_data['price']
                timestamp = price_data['timestamp']
                
                # Update paper trader with new price
                await self.paper_trader.on_tick(current_price, timestamp)
                
                # Check for trailing stops
                await self._check_trailing_stops(current_price)
                
                # Update portfolio metrics
                self.risk_manager._update_portfolio_metrics()
                
                # Log status every 10 minutes
                if int(timestamp.strftime("%M")) % 10 == 0:
                    portfolio_summary = self.paper_trader.get_portfolio_summary()
                    logger.info(f"Portfolio Status: Equity=${portfolio_summary['total_equity']:.2f}, "
                              f"Positions={portfolio_summary['active_positions']}, "
                              f"Win Rate={portfolio_summary['win_rate']:.1%}")
                
                # Sleep for 1 second (price updates)
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in price monitoring loop: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    async def _risk_monitoring_loop(self):
        """Risk monitoring loop"""
        logger.info("🛡️ Starting risk monitoring loop...")
        
        while self.is_running:
            try:
                # Get risk status
                risk_status = self.risk_manager.get_risk_status()
                
                # Check for risk violations
                if any(risk_status['risk_flags'].values()):
                    logger.warning(f"⚠️ Risk violations detected: {risk_status['risk_flags']}")
                    
                    # If daily loss limit exceeded, stop trading
                    if risk_status['risk_flags'].get('daily_loss_limit', False):
                        logger.warning("🛑 Daily loss limit exceeded - pausing trading")
                        if self.signal_generator:
                            self.signal_generator.stop_live_signals()
                
                # Log risk status every hour
                current_minute = datetime.now().minute
                if current_minute == 0:
                    logger.info(f"Risk Status: DD={risk_status['capital']['current_drawdown_pct']:.2f}%, "
                              f"Daily P&L=${risk_status['daily']['pnl']:.2f}")
                
                # Sleep for 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _portfolio_saving_loop(self):
        """Portfolio state saving loop"""
        logger.info("💾 Starting portfolio saving loop...")
        
        while self.is_running:
            try:
                # Save portfolio state every 5 minutes
                await asyncio.sleep(300)  # 5 minutes
                
                if self.paper_trader:
                    self.paper_trader.save_portfolio_state()
                    logger.info("✅ Portfolio state saved")
                
            except Exception as e:
                logger.error(f"Error in portfolio saving loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_trailing_stops(self, current_price: float):
        """Check and update trailing stops"""
        try:
            for trade_id, position in self.paper_trader.positions.items():
                # Get trailing stop recommendation
                position_dict = {
                    'entry_price': position.entry_price,
                    'direction': position.direction.value,
                    'stop_loss': position.stop_loss
                }
                
                trail_decision = self.risk_manager.should_trail_stop_loss(position_dict, current_price)
                
                if trail_decision['should_trail']:
                    # Update stop loss
                    old_sl = position.stop_loss
                    position.stop_loss = trail_decision['new_stop_loss']
                    
                    logger.info(f"📈 Trailing stop updated for {trade_id}: "
                              f"${old_sl:.2f} → ${position.stop_loss:.2f} "
                              f"({trail_decision['reason']})")
                
        except Exception as e:
            logger.error(f"Error checking trailing stops: {e}")
    
    async def stop_system(self):
        """Stop the paper trading system"""
        logger.info("🛑 Stopping XAU1 Paper Trading System...")
        
        self.is_running = False
        
        # Stop signal generator
        if self.signal_generator:
            self.signal_generator.stop_live_signals()
        
        # Cancel all tasks
        for task in self.system_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.system_tasks:
            await asyncio.gather(*self.system_tasks, return_exceptions=True)
        
        # Close any open positions
        if self.paper_trader:
            await self.paper_trader.close_all_positions("System shutdown")
            # Save final state
            self.paper_trader.save_portfolio_state()
        
        logger.info("✅ XAU1 Paper Trading System stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        if self.is_running:
            # Create task to stop system
            asyncio.create_task(self.stop_system())
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        status = {
            'is_running': self.is_running,
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Paper trader status
        if self.paper_trader:
            status['components']['paper_trader'] = self.paper_trader.get_portfolio_summary()
        
        # Signal generator status
        if self.signal_generator:
            status['components']['signal_generator'] = self.signal_generator.get_status()
        
        # Risk manager status
        if self.risk_manager:
            status['components']['risk_manager'] = self.risk_manager.get_risk_status()
        
        # Binance connector status
        if self.binance_connector:
            status['components']['binance_connector'] = self.binance_connector.get_connection_status()
        
        return status


async def main():
    """Main entry point"""
    # Check for command line arguments
    config_path = sys.argv[1] if len(sys.argv) > 1 else "src/xau1/config/optimized_strategy_params.yaml"
    
    # Create and start system
    system = XAU1PaperTradingSystem(config_path)
    
    try:
        await system.start_system()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await system.stop_system()


if __name__ == "__main__":
    # Ensure logs directory exists
    import os
    os.makedirs('logs', exist_ok=True)
    
    # Run the main system
    asyncio.run(main())