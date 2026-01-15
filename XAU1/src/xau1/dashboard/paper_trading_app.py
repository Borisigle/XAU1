"""
XAU1 Paper Trading Dashboard
Streamlit dashboard for real-time paper trading monitoring
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np

# Configure page
st.set_page_config(
    page_title="XAU1 Paper Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .position-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    
    .profit-positive {
        color: #28a745;
        font-weight: bold;
    }
    
    .profit-negative {
        color: #dc3545;
        font-weight: bold;
    }
    
    .signal-active {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    
    .signal-inactive {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

logger = logging.getLogger(__name__)


class PaperTradingDashboard:
    """Streamlit dashboard for paper trading monitoring"""
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        """Setup dashboard logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown("""
        <div class="main-header">
            <h1>🏆 XAU1 Paper Trading Dashboard</h1>
            <p>Real-time SMC + Order Flow Trading on XAU/USDT</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar_controls(self):
        """Render sidebar controls"""
        st.sidebar.header("📊 Controls")
        
        # Connection status
        st.sidebar.subheader("🔗 Connection Status")
        
        # Initialize session state
        if 'paper_trader' not in st.session_state:
            st.session_state.paper_trader = None
        if 'signal_generator' not in st.session_state:
            st.session_state.signal_generator = None
        if 'risk_manager' not in st.session_state:
            st.session_state.risk_manager = None
        if 'is_running' not in st.session_state:
            st.session_state.is_running = False
        
        # Control buttons
        if st.sidebar.button("🚀 Start Paper Trading", type="primary"):
            self.start_paper_trading()
        
        if st.sidebar.button("🛑 Stop Paper Trading", type="secondary"):
            self.stop_paper_trading()
        
        if st.sidebar.button("💾 Save Portfolio State"):
            self.save_portfolio_state()
        
        # Settings
        st.sidebar.subheader("⚙️ Settings")
        
        # Risk per trade
        risk_per_trade = st.sidebar.slider(
            "Risk per Trade (%)", 
            min_value=0.5, 
            max_value=3.0, 
            value=1.0, 
            step=0.1
        )
        
        # Stop trading conditions
        st.sidebar.subheader("🛡️ Risk Limits")
        
        max_daily_loss = st.sidebar.slider(
            "Max Daily Loss (%)",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.1
        )
        
        max_positions = st.sidebar.slider(
            "Max Positions",
            min_value=1,
            max_value=3,
            value=2
        )
        
        # Manual controls
        st.sidebar.subheader("🎮 Manual Controls")
        
        if st.sidebar.button("📊 Force Signal Check"):
            self.force_signal_check()
        
        if st.sidebar.button("📈 Update Prices"):
            self.update_prices()
        
        # Session info
        st.sidebar.subheader("ℹ️ Session Info")
        
        if st.session_state.is_running:
            st.sidebar.success("✅ Paper Trading Active")
        else:
            st.sidebar.warning("⏸️ Paper Trading Stopped")
        
        # Log viewer
        st.sidebar.subheader("📋 Recent Logs")
        
        return {
            'risk_per_trade': risk_per_trade,
            'max_daily_loss': max_daily_loss,
            'max_positions': max_positions
        }
    
    def render_portfolio_summary(self, portfolio_data: Dict):
        """Render portfolio summary metrics"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="💰 Total Equity",
                value=f"${portfolio_data.get('total_equity', 0):,.2f}",
                delta=f"{portfolio_data.get('total_return_pct', 0):.2f}%"
            )
        
        with col2:
            st.metric(
                label="📈 Available Capital",
                value=f"${portfolio_data.get('available_capital', 0):,.2f}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="🎯 Active Positions",
                value=portfolio_data.get('active_positions', 0),
                delta=None
            )
        
        with col4:
            win_rate = portfolio_data.get('win_rate', 0) * 100
            st.metric(
                label="🏆 Win Rate",
                value=f"{win_rate:.1f}%",
                delta=None
            )
        
        with col5:
            profit_factor = portfolio_data.get('profit_factor', 0)
            st.metric(
                label="💹 Profit Factor",
                value=f"{profit_factor:.2f}",
                delta=None
            )
    
    def render_equity_curve(self, equity_data: List[Dict]):
        """Render equity curve chart"""
        if not equity_data:
            st.info("No equity data available yet")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(equity_data)
        
        if df.empty:
            st.info("No equity curve data available")
            return
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['equity'],
            mode='lines',
            name='Portfolio Equity',
            line=dict(color='#667eea', width=2)
        ))
        
        # Add initial capital line
        if 'initial_capital' in df.columns:
            fig.add_hline(
                y=df['initial_capital'].iloc[0],
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Capital"
            )
        
        fig.update_layout(
            title="📈 Portfolio Equity Curve",
            xaxis_title="Time",
            yaxis_title="Equity ($)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_active_positions(self, positions: List[Dict]):
        """Render active positions"""
        st.subheader("📋 Active Positions")
        
        if not positions:
            st.info("No active positions")
            return
        
        for position in positions:
            with st.container():
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                # Position info
                direction = position.get('direction', 'unknown').upper()
                pnl = position.get('unrealized_pnl', 0)
                pnl_class = "profit-positive" if pnl > 0 else "profit-negative"
                
                with col1:
                    st.markdown(f"**{direction}**")
                
                with col2:
                    st.markdown(f"Entry: ${position.get('entry_price', 0):.2f}")
                
                with col3:
                    st.markdown(f"SL: ${position.get('stop_loss', 0):.2f}")
                
                with col4:
                    st.markdown(f"TP1: ${position.get('take_profit1', 0):.2f}")
                
                with col5:
                    st.markdown(f"TP2: ${position.get('take_profit2', 0):.2f}")
                
                with col6:
                    st.markdown(f'<span class="{pnl_class}">P&L: ${pnl:.2f}</span>', unsafe_allow_html=True)
                
                st.divider()
    
    def render_recent_trades(self, trades: List[Dict]):
        """Render recent trades table"""
        st.subheader("📋 Recent Trades")
        
        if not trades:
            st.info("No trades executed yet")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(trades)
        
        if df.empty:
            st.info("No trade data available")
            return
        
        # Format data for display
        df['pnl'] = df['pnl'].apply(lambda x: f"${x:.2f}" if x is not None else "N/A")
        df['exit_reason'] = df['exit_reason'].fillna('Unknown')
        df['direction'] = df['direction'].apply(lambda x: x.value if hasattr(x, 'value') else str(x))
        
        # Show table
        st.dataframe(
            df[['trade_id', 'direction', 'entry_price', 'exit_price', 'pnl', 'exit_reason', 'entry_time']],
            use_container_width=True
        )
    
    def render_market_data(self, market_data: Dict):
        """Render current market data"""
        st.subheader("📊 Current Market Data")
        
        if not market_data:
            st.info("No market data available")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💰 XAU/USDT Price",
                value=f"${market_data.get('price', 0):.2f}",
                delta=f"{market_data.get('change_24h', 0):.2f}%"
            )
        
        with col2:
            st.metric(
                label="📈 24h High",
                value=f"${market_data.get('high_24h', 0):.2f}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="📉 24h Low", 
                value=f"${market_data.get('low_24h', 0):.2f}",
                delta=None
            )
        
        with col4:
            volume = market_data.get('volume_24h', 0)
            st.metric(
                label="📊 24h Volume",
                value=f"{volume:,.0f}",
                delta=None
            )
    
    def render_signals_status(self, signal_status: Dict):
        """Render signals status"""
        st.subheader("🎯 Signals Status")
        
        if not signal_status:
            st.info("No signal data available")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if signal_status.get('is_running', False):
                st.markdown('<div class="signal-active">✅ Signal Generator Active</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="signal-inactive">⏸️ Signal Generator Stopped</div>', unsafe_allow_html=True)
        
        with col2:
            session = signal_status.get('current_session', 'None')
            st.markdown(f"**Session:** {session.title()}")
        
        with col3:
            signals_today = signal_status.get('signals_today', 0)
            st.markdown(f"**Signals Today:** {signals_today}")
        
        # Connection info
        if 'connection_status' in signal_status:
            conn = signal_status['connection_status']
            st.markdown(f"**Binance Connection:** {'🟢 Connected' if conn.get('connected', False) else '🔴 Disconnected'}")
    
    def render_risk_status(self, risk_data: Dict):
        """Render risk management status"""
        st.subheader("🛡️ Risk Management")
        
        if not risk_data:
            st.info("No risk data available")
            return
        
        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_dd = risk_data.get('capital', {}).get('current_drawdown_pct', 0)
            st.metric("Current DD", f"{current_dd:.2f}%")
        
        with col2:
            daily_pnl = risk_data.get('daily', {}).get('pnl', 0)
            st.metric("Daily P&L", f"${daily_pnl:.2f}")
        
        with col3:
            weekly_pnl = risk_data.get('weekly', {}).get('pnl', 0)
            st.metric("Weekly P&L", f"${weekly_pnl:.2f}")
        
        with col4:
            recent_wr = risk_data.get('trading', {}).get('recent_win_rate', 0) * 100
            st.metric("Recent WR", f"{recent_wr:.1f}%")
        
        # Risk flags
        risk_flags = risk_data.get('risk_flags', {})
        if any(risk_flags.values()):
            st.warning("⚠️ Active Risk Flags:")
            for flag, active in risk_flags.items():
                if active:
                    st.markdown(f"- {flag.replace('_', ' ').title()}")
        else:
            st.success("✅ All risk checks passed")
    
    def render_logs(self):
        """Render recent logs"""
        st.subheader("📋 Recent Activity")
        
        # In a real implementation, this would read from log files
        # For demo purposes, show static content
        logs = [
            f"{datetime.now().strftime('%H:%M:%S')} - System started",
            f"{datetime.now().strftime('%H:%M:%S')} - Connected to Binance",
            f"{datetime.now().strftime('%H:%M:%S')} - Signal generator active",
            f"{datetime.now().strftime('%H:%M:%S')} - Risk manager initialized"
        ]
        
        for log_entry in logs[-10:]:  # Show last 10 entries
            st.text(log_entry)
    
    def start_paper_trading(self):
        """Start paper trading session"""
        try:
            # Initialize paper trading components
            from xau1.paper_trading.paper_trader import PaperTrader
            from xau1.paper_trading.live_signals import LiveSignalGenerator
            from xau1.paper_trading.risk_manager import LiveRiskManager
            
            # Create components
            st.session_state.paper_trader = PaperTrader()
            st.session_state.risk_manager = LiveRiskManager()
            st.session_state.is_running = True
            
            # Load strategy config
            with open('src/xau1/config/strategy_params.yaml', 'r') as f:
                strategy_config = yaml.safe_load(f)
            
            st.session_state.signal_generator = LiveSignalGenerator(strategy_config)
            
            st.success("🚀 Paper trading started successfully!")
            
        except Exception as e:
            st.error(f"Error starting paper trading: {str(e)}")
            logger.error(f"Error starting paper trading: {e}")
    
    def stop_paper_trading(self):
        """Stop paper trading session"""
        try:
            if st.session_state.signal_generator:
                st.session_state.signal_generator.stop_live_signals()
            
            st.session_state.is_running = False
            st.success("🛑 Paper trading stopped")
            
        except Exception as e:
            st.error(f"Error stopping paper trading: {str(e)}")
            logger.error(f"Error stopping paper trading: {e}")
    
    def save_portfolio_state(self):
        """Save current portfolio state"""
        try:
            if st.session_state.paper_trader:
                st.session_state.paper_trader.save_portfolio_state()
                st.success("💾 Portfolio state saved!")
            else:
                st.warning("No active paper trading session to save")
                
        except Exception as e:
            st.error(f"Error saving portfolio state: {str(e)}")
    
    def force_signal_check(self):
        """Force a signal check"""
        try:
            st.info("🔄 Forced signal check triggered")
            # This would trigger a manual signal check in a real implementation
            
        except Exception as e:
            st.error(f"Error forcing signal check: {str(e)}")
    
    def update_prices(self):
        """Update market prices"""
        try:
            st.info("📈 Price update triggered")
            # This would trigger a price update in a real implementation
            
        except Exception as e:
            st.error(f"Error updating prices: {str(e)}")
    
    def run_dashboard(self):
        """Main dashboard application"""
        try:
            # Import yaml for config loading
            import yaml
            
            # Render header
            self.render_header()
            
            # Get sidebar settings
            settings = self.render_sidebar_controls()
            
            # Main dashboard content
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Portfolio summary
                portfolio_data = self.get_sample_portfolio_data()
                self.render_portfolio_summary(portfolio_data)
                
                # Equity curve
                equity_data = self.get_sample_equity_data()
                self.render_equity_curve(equity_data)
                
                # Active positions
                positions = self.get_sample_positions_data()
                self.render_active_positions(positions)
                
                # Recent trades
                trades = self.get_sample_trades_data()
                self.render_recent_trades(trades)
            
            with col2:
                # Market data
                market_data = self.get_sample_market_data()
                self.render_market_data(market_data)
                
                # Signals status
                signal_status = self.get_sample_signal_status()
                self.render_signals_status(signal_status)
                
                # Risk status
                risk_data = self.get_sample_risk_data()
                self.render_risk_status(risk_data)
                
                # Logs
                self.render_logs()
        
        except Exception as e:
            st.error(f"Error running dashboard: {str(e)}")
            logger.error(f"Dashboard error: {e}")
    
    # Sample data methods for demonstration
    def get_sample_portfolio_data(self) -> Dict:
        """Get sample portfolio data for demonstration"""
        return {
            'total_equity': 10543.21,
            'total_return_pct': 5.43,
            'available_capital': 8432.50,
            'active_positions': 1,
            'win_rate': 0.625,
            'profit_factor': 2.15,
            'total_trades': 8
        }
    
    def get_sample_equity_data(self) -> List[Dict]:
        """Get sample equity curve data for demonstration"""
        base_time = datetime.now() - timedelta(days=7)
        data = []
        
        for i in range(50):
            timestamp = base_time + timedelta(hours=i*3)
            equity = 10000 + i * 12.5 + np.random.normal(0, 25)
            data.append({
                'timestamp': timestamp,
                'equity': equity,
                'initial_capital': 10000
            })
        
        return data
    
    def get_sample_positions_data(self) -> List[Dict]:
        """Get sample positions data for demonstration"""
        return [
            {
                'trade_id': 'PAPER_001',
                'direction': 'long',
                'entry_price': 2425.50,
                'stop_loss': 2399.00,
                'take_profit1': 2475.50,
                'take_profit2': 2525.50,
                'unrealized_pnl': 145.30,
                'quantity': 0.5
            }
        ]
    
    def get_sample_trades_data(self) -> List[Dict]:
        """Get sample trades data for demonstration"""
        return [
            {
                'trade_id': 'PAPER_001',
                'direction': 'long',
                'entry_price': 2410.20,
                'exit_price': 2460.00,
                'pnl': 125.50,
                'exit_reason': 'TP1',
                'entry_time': datetime.now() - timedelta(hours=4)
            },
            {
                'trade_id': 'PAPER_002',
                'direction': 'short',
                'entry_price': 2418.50,
                'exit_price': 2391.00,
                'pnl': -67.25,
                'exit_reason': 'Stop Loss',
                'entry_time': datetime.now() - timedelta(hours=6)
            }
        ]
    
    def get_sample_market_data(self) -> Dict:
        """Get sample market data for demonstration"""
        return {
            'price': 2425.75,
            'change_24h': 1.25,
            'high_24h': 2435.20,
            'low_24h': 2415.30,
            'volume_24h': 15847.5
        }
    
    def get_sample_signal_status(self) -> Dict:
        """Get sample signal status for demonstration"""
        return {
            'is_running': True,
            'current_session': 'london',
            'signals_today': 2,
            'connection_status': {
                'connected': True,
                'last_update': datetime.now().isoformat()
            }
        }
    
    def get_sample_risk_data(self) -> Dict:
        """Get sample risk data for demonstration"""
        return {
            'capital': {
                'current_drawdown_pct': 2.15
            },
            'daily': {
                'pnl': 243.50
            },
            'weekly': {
                'pnl': 543.21
            },
            'trading': {
                'recent_win_rate': 0.60
            },
            'risk_flags': {
                'daily_loss_limit': False,
                'max_drawdown': False,
                'low_win_rate': False
            }
        }


def main():
    """Main dashboard application"""
    try:
        dashboard = PaperTradingDashboard()
        dashboard.run_dashboard()
        
    except Exception as e:
        st.error(f"Failed to start dashboard: {str(e)}")
        logger.error(f"Dashboard startup error: {e}")


if __name__ == "__main__":
    main()