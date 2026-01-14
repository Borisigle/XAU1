"""
Binance Exchange Integration using CCXT
Handles data fetching and order management
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import ccxt
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BinanceConfig(BaseModel):
    """Binance exchange configuration"""
    
    api_key: Optional[str] = Field(default=None)
    api_secret: Optional[str] = Field(default=None)
    testnet: bool = Field(default=True)
    enable_rate_limit: bool = Field(default=True)
    timeout: int = Field(default=30000)


class BinanceClient:
    """Binance exchange client for XAUUSDT"""
    
    def __init__(self, config: Optional[BinanceConfig] = None):
        self.config = config or BinanceConfig()
        self.exchange = self._initialize_exchange()
        self.symbol = "XAUUSDT"
    
    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize CCXT exchange instance"""
        try:
            exchange = ccxt.binance({
                "enableRateLimit": self.config.enable_rate_limit,
                "timeout": self.config.timeout,
                "options": {
                    "defaultType": "spot",
                },
            })
            
            # Add API credentials if provided
            if self.config.api_key and self.config.api_secret:
                exchange.apiKey = self.config.api_key
                exchange.secret = self.config.api_secret
            
            # Load markets
            exchange.load_markets()
            logger.info("Binance exchange initialized successfully")
            
            return exchange
            
        except Exception as e:
            logger.error(f"Failed to initialize Binance exchange: {e}")
            raise
    
    def fetch_ohlcv(
        self,
        timeframe: str = "15m",
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
        to_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance
        
        Args:
            timeframe: Timeframe string (15m, 1h, 4h, 1d)
            since: Start date for data fetching
            limit: Maximum number of candles to fetch
            to_date: End date for data fetching
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert dates to milliseconds
            since_ms = None
            to_ms = None
            
            if since:
                since_ms = int(since.timestamp() * 1000)
            if to_date:
                to_ms = int(to_date.timestamp() * 1000)
            
            # Fetch data in chunks to avoid rate limits
            all_data = []
            current_since = since_ms
            
            while True:
                try:
                    # Fetch data
                    data = self.exchange.fetch_ohlcv(
                        self.symbol,
                        timeframe=timeframe,
                        since=current_since,
                        limit=min(limit or 1000, 1000),
                    )
                    
                    if not data:
                        break
                        
                    all_data.extend(data)
                    
                    # Check if we've reached the end date
                    last_timestamp = data[-1][0]
                    if to_ms and last_timestamp >= to_ms:
                        break
                        
                    # Update since for next chunk
                    current_since = last_timestamp + 1
                    
                    # Rate limit
                    self.exchange.sleep(100)
                    
                except ccxt.RateLimitError:
                    logger.warning("Rate limit hit, sleeping...")
                    self.exchange.sleep(1000)
                    continue
                    
            if not all_data:
                logger.warning(f"No data fetched for {self.symbol}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(
                all_data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            
            # Filter by date range if to_date specified
            if to_date:
                df = df[df.index <= to_date]
                
            logger.info(f"Fetched {len(df)} candles for {self.symbol} {timeframe}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            raise
    
    def fetch_recent_ohlcv(
        self,
        timeframe: str = "15m",
        days: int = 365,
    ) -> pd.DataFrame:
        """Fetch recent OHLCV data for specified number of days"""
        since = datetime.now() - timedelta(days=days)
        return self.fetch_ohlcv(timeframe=timeframe, since=since)
    
    def get_market_info(self) -> Dict:
        """Get market information for XAUUSDT"""
        try:
            market = self.exchange.market(self.symbol)
            return {
                "symbol": market["symbol"],
                "base": market["base"],
                "quote": market["quote"],
                "precision": market["precision"],
                "limits": market["limits"],
                "active": market.get("active", True),
            }
        except Exception as e:
            logger.error(f"Error getting market info: {e}")
            return {}
    
    def get_current_price(self) -> float:
        """Get current market price"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return float(ticker["last"])
        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
            raise
    
    def submit_order(
        self,
        side: str,
        amount: float,
        price: Optional[float] = None,
        order_type: str = "market",
        params: Optional[Dict] = None,
    ) -> Dict:
        """
        Submit an order to Binance
        
        Args:
            side: "buy" or "sell"
            amount: Order quantity
            price: Order price (for limit orders)
            order_type: "market" or "limit"
            params: Additional order parameters
            
        Returns:
            Order response from exchange
        """
        try:
            if order_type == "limit" and price is None:
                raise ValueError("Price required for limit orders")
                
            order_params = params or {}
            
            order = self.exchange.create_order(
                symbol=self.symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=order_params,
            )
            
            logger.info(f"Order submitted: {order['id']} - {side} {amount} @ {price or 'market'}")
            return order
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an existing order"""
        try:
            result = self.exchange.cancel_order(order_id, self.symbol)
            logger.info(f"Order canceled: {order_id}")
            return result
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            raise
    
    def get_balance(self) -> Dict:
        """Get account balance"""
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            raise