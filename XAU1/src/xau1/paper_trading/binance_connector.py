"""
XAU1 Paper Trading Binance Connector
Simulated live connection to Binance for paper trading
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class PaperBinanceConnector:
    """Simulated Binance connector for paper trading"""
    
    def __init__(self, 
                 api_key: str = None, 
                 api_secret: str = None,
                 sandbox: bool = True):
        """
        Initialize Binance connector for paper trading
        
        Args:
            api_key: Binance API key (not used in paper trading)
            api_secret: Binance API secret (not used in paper trading)
            sandbox: Use sandbox mode (True for paper trading)
        """
        self.exchange = None
        self.symbol = "XAU/USDT"
        self.timeframe = "15m"
        self.is_sandbox = sandbox
        
        # Price simulation
        self.current_price = 2025.50  # Initial XAU price
        self.price_history = []
        self.last_price_update = datetime.now()
        
        # Connection status
        self.connected = False
        self.last_error = None
        
        # Data storage
        self.ohlcv_data = []
        self.last_bar_time = None
        
        # Volatility simulation
        self.volatility = 0.015  # 1.5% typical XAU volatility
        self.trend_direction = 1  # 1 for up, -1 for down
        
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize exchange connection"""
        try:
            # For paper trading, we don't need real API keys
            self.exchange = ccxt.binance({
                'sandbox': self.is_sandbox,
                'enableRateLimit': True,
            })
            
            # Test connection
            self.exchange.load_markets()
            self.connected = True
            logger.info(f"Binance connector initialized (sandbox: {self.is_sandbox})")
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to initialize Binance connector: {e}")
            self.connected = False
    
    async def get_live_prices(self, symbol: str = "XAU/USDT") -> Dict:
        """
        Get live price data from Binance
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with current price data
        """
        try:
            if not self.connected:
                return self._get_simulated_price()
            
            # Get ticker data
            ticker = self.exchange.fetch_ticker(symbol)
            
            price_data = {
                'symbol': symbol,
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000),
                'volume_24h': ticker['quoteVolume'],
                'change_24h': ticker['percentage'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low']
            }
            
            # Update current price for simulation
            self.current_price = price_data['price']
            self.last_price_update = datetime.now()
            
            return price_data
            
        except Exception as e:
            logger.warning(f"Error fetching live prices: {e}. Using simulated data.")
            return self._get_simulated_price()
    
    def _get_simulated_price(self) -> Dict:
        """Get simulated price data for paper trading"""
        # Simulate realistic price movement
        price_change = self._generate_price_movement()
        self.current_price += price_change
        
        # Keep price within realistic bounds for XAU
        self.current_price = max(1800, min(2500, self.current_price))
        
        # Calculate bid/ask (spread simulation)
        spread = 0.50  # 50 cents spread
        bid = self.current_price - spread / 2
        ask = self.current_price + spread / 2
        
        price_data = {
            'symbol': self.symbol,
            'price': self.current_price,
            'bid': bid,
            'ask': ask,
            'timestamp': self.last_price_update,
            'volume_24h': 15000 + (self.current_price - 2000) * 100,  # Simulated volume
            'change_24h': ((self.current_price - 2025.50) / 2025.50) * 100,
            'high_24h': self.current_price + abs(price_change) * 10,
            'low_24h': self.current_price - abs(price_change) * 10,
            'simulated': True
        }
        
        return price_data
    
    def _generate_price_movement(self) -> float:
        """Generate realistic price movement for simulation"""
        import random
        
        # Random walk with trend component
        random_change = random.gauss(0, self.volatility * 0.1)
        trend_change = self.trend_direction * self.volatility * 0.05
        
        # Occasionally change trend direction
        if random.random() < 0.05:  # 5% chance each update
            self.trend_direction *= -1
        
        # Limit maximum change
        max_change = self.current_price * self.volatility * 0.1
        total_change = random_change + trend_change
        
        return max(-max_change, min(max_change, total_change))
    
    async def get_latest_15m_bar(self, symbol: str = "XAU/USDT") -> Optional[Dict]:
        """
        Get latest 15-minute OHLCV bar
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with OHLCV data or None if no data
        """
        try:
            if not self.connected:
                return self._generate_simulated_bar()
            
            # Fetch latest OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=2)
            
            if len(ohlcv) < 1:
                return None
            
            latest_bar = ohlcv[-1]
            
            bar_data = {
                'timestamp': datetime.fromtimestamp(latest_bar[0] / 1000),
                'open': latest_bar[1],
                'high': latest_bar[2],
                'low': latest_bar[3],
                'close': latest_bar[4],
                'volume': latest_bar[5],
                'symbol': symbol,
                'timeframe': self.timeframe
            }
            
            # Update OHLCV data storage
            if (not self.ohlcv_data or 
                bar_data['timestamp'] > self.ohlcv_data[-1]['timestamp']):
                self.ohlcv_data.append(bar_data)
                
                # Keep only last 1000 bars
                if len(self.ohlcv_data) > 1000:
                    self.ohlcv_data = self.ohlcv_data[-1000:]
            
            return bar_data
            
        except Exception as e:
            logger.warning(f"Error fetching OHLCV data: {e}. Using simulated data.")
            return self._generate_simulated_bar()
    
    def _generate_simulated_bar(self) -> Optional[Dict]:
        """Generate simulated OHLCV bar"""
        try:
            current_time = datetime.now()
            
            # Create a new bar every 15 minutes
            if (self.last_bar_time is None or 
                (current_time - self.last_bar_time).seconds >= 900):  # 15 minutes
                
                # Generate realistic OHLCV data
                base_price = self.current_price
                
                # Create realistic price movement within the bar
                open_price = base_price + self._generate_price_movement() * 0.5
                
                # Generate intrabar high/low
                volatility_range = base_price * self.volatility * 0.1
                high_price = open_price + abs(self._generate_price_movement()) + volatility_range * 0.3
                low_price = open_price - abs(self._generate_price_movement()) - volatility_range * 0.3
                
                # Close price with some movement from open
                close_price = open_price + self._generate_price_movement()
                
                # Ensure OHLC logic (High >= all, Low <= all)
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)
                
                bar_data = {
                    'timestamp': current_time,
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': 1000 + abs(self._generate_price_movement()) * 50000,
                    'symbol': self.symbol,
                    'timeframe': self.timeframe,
                    'simulated': True
                }
                
                # Update current price to close price
                self.current_price = close_price
                self.last_bar_time = current_time
                
                # Store bar data
                self.ohlcv_data.append(bar_data)
                
                # Keep only last 1000 bars
                if len(self.ohlcv_data) > 1000:
                    self.ohlcv_data = self.ohlcv_data[-1000:]
                
                return bar_data
            
            return None  # No new bar yet
            
        except Exception as e:
            logger.error(f"Error generating simulated bar: {e}")
            return None
    
    async def get_historical_data(self, 
                                symbol: str = "XAU/USDT", 
                                timeframe: str = "15m", 
                                limit: int = 500) -> List[Dict]:
        """
        Get historical OHLCV data
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            limit: Number of bars to fetch
            
        Returns:
            List of OHLCV bars
        """
        try:
            if not self.connected:
                return self._generate_historical_data(limit)
            
            # Fetch historical data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            bars = []
            for bar in ohlcv:
                bars.append({
                    'timestamp': datetime.fromtimestamp(bar[0] / 1000),
                    'open': bar[1],
                    'high': bar[2],
                    'low': bar[3],
                    'close': bar[4],
                    'volume': bar[5],
                    'symbol': symbol,
                    'timeframe': timeframe
                })
            
            return bars
            
        except Exception as e:
            logger.warning(f"Error fetching historical data: {e}. Using simulated data.")
            return self._generate_historical_data(limit)
    
    def _generate_historical_data(self, limit: int) -> List[Dict]:
        """Generate simulated historical data"""
        bars = []
        base_time = datetime.now() - timedelta(minutes=limit * 15)
        
        # Start with a base price
        price = 2025.50
        
        for i in range(limit):
            current_time = base_time + timedelta(minutes=i * 15)
            
            # Generate realistic price movement
            daily_trend = 0.0001 * (i - limit/2) / limit  # Slight trend
            volatility = 0.008 + (i % 96) * 0.0001  # Intraday volatility pattern
            
            # Price movement
            change = daily_trend + (price * volatility * (hash(str(i)) % 100 - 50) / 50)
            price += change
            
            # Generate OHLC
            open_price = price
            high_price = open_price + abs(change) * (1 + (hash(str(i+1)) % 100) / 100)
            low_price = open_price - abs(change) * (1 + (hash(str(i+2)) % 100) / 100)
            close_price = open_price + change + (hash(str(i+3)) % 100 - 50) / 500
            
            # Ensure OHLC logic
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            bar = {
                'timestamp': current_time,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': 800 + (hash(str(i+4)) % 400),
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'simulated': True
            }
            
            bars.append(bar)
            price = close_price
        
        return bars
    
    async def simulate_order(self, 
                           order_type: str, 
                           side: str, 
                           amount: float, 
                           price: float = None) -> Dict:
        """
        Simulate order execution
        
        Args:
            order_type: 'market' or 'limit'
            side: 'buy' or 'sell'
            amount: Order amount
            price: Order price (for limit orders)
            
        Returns:
            Dict with simulated execution details
        """
        try:
            # Simulate realistic execution
            execution_delay = 0.1 + (hash(str(datetime.now())) % 100) / 1000  # 0.1-0.2 seconds
            await asyncio.sleep(execution_delay)
            
            # Get current market price for market orders
            if order_type == 'market':
                current_price = self.current_price
                # Add slippage simulation
                if side == 'buy':
                    execution_price = current_price + self._simulate_slippage()
                else:
                    execution_price = current_price - self._simulate_slippage()
            else:
                execution_price = price
            
            # Calculate fees
            fee_rate = 0.001  # 0.1% Binance fee
            fee = amount * execution_price * fee_rate
            
            execution = {
                'id': f"PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                'symbol': self.symbol,
                'type': order_type,
                'side': side,
                'amount': amount,
                'price': execution_price,
                'cost': amount * execution_price,
                'fee': fee,
                'timestamp': datetime.now(),
                'status': 'filled',
                'simulated': True,
                'slippage_pips': self._calculate_slippage_pips(current_price, execution_price)
            }
            
            logger.info(f"Simulated order executed: {side} {amount} {self.symbol} @ {execution_price:.2f}")
            
            return execution
            
        except Exception as e:
            logger.error(f"Error simulating order: {e}")
            return {
                'error': str(e),
                'simulated': True
            }
    
    def _simulate_slippage(self) -> float:
        """Simulate slippage for market orders"""
        import random
        
        # Typical XAU slippage: 0.1-0.5 pips
        slippage = random.uniform(0.10, 0.50)
        return slippage
    
    def _calculate_slippage_pips(self, original_price: float, execution_price: float) -> float:
        """Calculate slippage in pips"""
        return abs(original_price - execution_price) / 0.01  # 1 pip = 0.01 for XAU
    
    async def get_account_balance(self) -> Dict:
        """
        Get simulated account balance
        
        Returns:
            Dict with balance information
        """
        return {
            'free': {
                'USDT': 10000.00,
                'XAU': 0.0
            },
            'used': {
                'USDT': 0.0,
                'XAU': 0.0
            },
            'total': {
                'USDT': 10000.00,
                'XAU': 0.0
            },
            'simulated': True
        }
    
    async def get_order_status(self, order_id: str) -> Dict:
        """
        Get simulated order status
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Dict with order status
        """
        return {
            'id': order_id,
            'symbol': self.symbol,
            'status': 'filled',
            'filled': True,
            'remaining': 0.0,
            'simulated': True
        }
    
    def update_price_simulation(self, new_price: float):
        """
        Update price simulation manually (for testing)
        
        Args:
            new_price: New price to simulate
        """
        self.current_price = new_price
        logger.info(f"Price simulation updated to ${new_price:.2f}")
    
    def get_connection_status(self) -> Dict:
        """Get connection status information"""
        return {
            'connected': self.connected,
            'exchange': 'binance',
            'sandbox': self.is_sandbox,
            'symbol': self.symbol,
            'current_price': self.current_price,
            'last_update': self.last_price_update.isoformat(),
            'last_error': self.last_error,
            'bars_stored': len(self.ohlcv_data)
        }


async def main():
    """Test the Binance connector"""
    connector = PaperBinanceConnector()
    
    print("Testing Paper Binance Connector...")
    
    # Test connection status
    status = connector.get_connection_status()
    print(f"Connection Status: {json.dumps(status, indent=2, default=str)}")
    
    # Test live prices
    print("\nTesting live prices...")
    prices = await connector.get_live_prices()
    print(f"Prices: {json.dumps(prices, indent=2, default=str)}")
    
    # Test OHLCV data
    print("\nTesting OHLCV data...")
    ohlcv = await connector.get_latest_15m_bar()
    if ohlcv:
        print(f"Latest Bar: {json.dumps(ohlcv, indent=2, default=str)}")
    
    # Test order simulation
    print("\nTesting order simulation...")
    order = await connector.simulate_order('market', 'buy', 0.1)
    print(f"Order: {json.dumps(order, indent=2, default=str)}")
    
    # Test account balance
    print("\nTesting account balance...")
    balance = await connector.get_account_balance()
    print(f"Balance: {json.dumps(balance, indent=2, default=str)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(main())