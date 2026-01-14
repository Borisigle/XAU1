"""
Streamlit Dashboard for XAU1 Trading Bot
Interactive visualization of backtest results and live analysis
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Add src to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from xau1.backtest.backtester import XAU1Backtester
from xau1.dashboard.charts import ChartManager
from xau1.dashboard.metrics import MetricsDisplay
from xau1.engine.indicators import SMCIndicators
from xau1.utils.data import DataManager, load_config

# Configure page
st.set_page_config(
    page_title="XAU1 Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = logging.getLogger(__name__)


def load_backtest_data():
    """Load latest backtest results"""
    results_dir = Path("backtest_results")
    if not results_dir.exists():
        return None, None, None
    
    # Get latest files
    trades_files = sorted(results_dir.glob("trades_*.csv"), reverse=True)
    equity_files = sorted(results_dir.glob("equity_*.csv"), reverse=True)
    
    trades_df = None
    equity_df = None
    
    if trades_files:
        try:
            trades_df = pd.read_csv(trades_files[0], parse_dates=['entry_time', 'exit_time'])
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
    
    if equity_files:
        try:
            equity_df = pd.read_csv(equity_files[0], parse_dates=['timestamp'])
        except Exception as e:
            logger.error(f"Error loading equity: {e}")
    
    # Load results summary if available
    results_summary = None
    if trades_df is not None:
        # Recalculate summary if needed
        if len(trades_df) > 0:
            results_summary = {
                'total_trades': len(trades_df),
                'winning_trades': len(trades_df[trades_df['pnl'] > 0]),
                'losing_trades': len(trades_df[trades_df['pnl'] < 0]),
                'win_rate': len(trades_df[trades_df['pnl'] > 0]) / len(trades_df),
                'total_pnl': trades_df['pnl'].sum(),
                'profit_factor': trades_df[trades_df['pnl'] > 0]['pnl'].sum() / abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()),
                'max_drawdown_pct': 0,  # Would calculate from equity
                'sharpe_ratio': 0,
                'trades_per_week': 0,
                'avg_risk_reward': trades_df['risk_reward'].mean(),
                'total_return_pct': 0
            }
    
    return results_summary, trades_df, equity_df


def run_new_backtest(data_file: str, config_path: str):
    """Run new backtest with selected configuration"""
    
    st.info("Running backtest... This may take a few minutes.")
    
    try:
        # Load configuration
        strategy_config = load_config(config_path)
        backtest_config = load_config("src/xau1/config/backtest_params.yaml")
        
        # Load data
        data_manager = DataManager()
        df = data_manager.load_csv(data_file)
        
        if df is None or df.empty:
            st.error("No data available for backtest")
            return None, None, None
        
        # Run backtest
        backtester = XAU1Backtester(strategy_config, backtest_config)
        result = backtester.run_backtest(df)
        
        # Get results
        trades_df = backtester.get_trades_dataframe()
        equity_df = backtester.get_equity_curve()
        
        # Display success message
        st.success("Backtest completed successfully!")
        
        return result.model_dump(), trades_df, equity_df
        
    except Exception as e:
        st.error(f"Error running backtest: {e}")
        logger.error(f"Backtest error: {e}")
        return None, None, None


def main():
    """Main dashboard application"""
    
    st.title("📈 XAU1 Trading Bot Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        
        # Mode selection
        mode = st.radio("Mode", ["Backtest Results", "Live Analysis", "Run New Backtest"])
        
        # Configuration
        st.subheader("Configuration")
        
        # Data file selection
        data_manager = DataManager()
        data_files = list(Path("data").glob("*.csv")) if Path("data").exists() else []
        
        if data_files:
            data_file = st.selectbox(
                "Data File",
                options=data_files,
                format_func=lambda x: x.name
            )
        else:
            st.warning("No data files found. Run main.py --setup-first.")
            return
        
        # Strategy config
        config_files = list(Path("src/xau1/config").glob("*.yaml"))
        config_file = st.selectbox(
            "Strategy Config",
            options=config_files,
            format_func=lambda x: x.name
        )
        
        # Date range filter
        if mode in ["Backtest Results", "Live Analysis"]:
            df_sample = data_manager.load_csv(data_file.name)
            if df_sample is not None:
                min_date = df_sample.index.min().date()
                max_date = df_sample.index.max().date()
                
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
        
        # Run button for backtest
        if mode == "Run New Backtest":
            if st.button("Run Backtest", type="primary", use_container_width=True):
                with st.spinner("Running backtest..."):
                    results, trades_df, equity_df = run_new_backtest(data_file.name, str(config_file))
                    
                    if results:
                        st.session_state['results'] = results
                        st.session_state['trades_df'] = trades_df
                        st.session_state['equity_df'] = equity_df
                        st.rerun()
    
    # Main content
    if mode == "Backtest Results":
        st.header("Backtest Performance")
        
        # Load existing results
        results, trades_df, equity_df = load_backtest_data()
        
        if results:
            # KPI Metrics
            MetricsDisplay.display_kpi_metrics(results)
            
            st.markdown("---")
            
            # Charts tab
            tab1, tab2, tab3 = st.tabs(["Performance Charts", "Trade Analysis", "Strategy Details"])
            
            with tab1:
                chart_manager = ChartManager()
                
                if equity_df is not None:
                    # Equity curve
                    equity_chart = chart_manager.create_equity_curve_chart(equity_df)
                    st.plotly_chart(equity_chart, use_container_width=True, key="equity")
                
                # Performance metrics bar chart
                metrics_data = {
                    'Win Rate %': results['win_rate'] * 100,
                    'Profit Factor': results['profit_factor'],
                    'Sharpe Ratio': results['sharpe_ratio'],
                    'Trades/Week': results['trades_per_week'],
                    'Avg R:R': results['avg_risk_reward']
                }
                metrics_chart = chart_manager.create_metrics_bar_chart(metrics_data)
                st.plotly_chart(metrics_chart, use_container_width=True, key="metrics")
            
            with tab2:
                if trades_df is not None:
                    # Trade statistics
                    MetricsDisplay.display_trade_statistics(trades_df)
                    
                    # Trade table
                    st.subheader("Trade History (Last 50)")
                    trades_chart = ChartManager().create_trades_table(trades_df)
                    st.plotly_chart(trades_chart, use_container_width=True)
                    
                    # Session performance
                    MetricsDisplay.display_session_performance(trades_df)
            
            with tab3:
                # Strategy parameters
                strategy_config = load_config(str(config_file))
                MetricsDisplay.display_strategy_parameters(strategy_config)
                
                # Data quality
                df = data_manager.load_csv(data_file.name)
                MetricsDisplay.display_data_quality_info(df)
        
        else:
            st.info("No backtest results found. Run a backtest first.")
    
    elif mode == "Live Analysis":
        st.header("Live Market Analysis")
        
        # Load data
        df = data_manager.load_csv(data_file.name)
        
        if df is not None:
            # Date filtering
            if 'date_range' in locals():
                start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
            else:
                df_filtered = df
            
            # Calculate indicators
            with st.spinner("Calculating indicators..."):
                smc = SMCIndicators(df_filtered)
                df_indicators = smc.calculate_all_indicators()
            
            # Chart
            chart_manager = ChartManager()
            price_chart = chart_manager.create_price_chart(df_indicators)
            st.plotly_chart(price_chart, use_container_width=True)
            
            # Key levels summary
            st.subheader("Key SMC Levels")
            
            current_price = df_filtered['close'].iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Recent swing points
                recent_swings = []
                if 'swing_high' in df_indicators.columns:
                    recent_highs = df_indicators[df_indicators['swing_high'].notna()]['swing_high'].tail(3)
                    if not recent_highs.empty:
                        st.write("**Recent Swing Highs:**")
                        for price in recent_highs:
                            st.write(f"$ {price:.2f}")
                
                if 'swing_low' in df_indicators.columns:
                    recent_lows = df_indicators[df_indicators['swing_low'].notna()]['swing_low'].tail(3)
                    if not recent_lows.empty:
                        st.write("**Recent Swing Lows:**")
                        for price in recent_lows:
                            st.write(f"$ {price:.2f}")
            
            with col2:
                # FVG levels
                st.write("**Fair Value Gaps:**")
                if 'fvg_mid_bullish' in df_indicators.columns:
                    bullish_fvg = df_indicators[df_indicators['fvg_bullish'] == True]
                    if not bullish_fvg.empty:
                        recent_fvg = bullish_fvg['fvg_mid_bullish'].iloc[-1]
                        st.write(f"Bullish FVG: $ {recent_fvg:.2f}")
                
                if 'fvg_mid_bearish' in df_indicators.columns:
                    bearish_fvg = df_indicators[df_indicators['fvg_bearish'] == True]
                    if not bearish_fvg.empty:
                        recent_fvg = bearish_fvg['fvg_mid_bearish'].iloc[-1]
                        st.write(f"Bearish FVG: $ {recent_fvg:.2f}")
            
            with col3:
                # Order blocks
                st.write("**Order Blocks:**")
                if 'ob_level_bullish' in df_indicators.columns:
                    bullish_ob = df_indicators[df_indicators['ob_bullish'] == True]
                    if not bullish_ob.empty:
                        recent_ob = bullish_ob['ob_level_bullish'].iloc[-1]
                        st.write(f"Bullish OB: $ {recent_ob:.2f}")
                
                if 'ob_level_bearish' in df_indicators.columns:
                    bearish_ob = df_indicators[df_indicators['ob_bearish'] == True]
                    if not bearish_ob.empty:
                        recent_ob = bearish_ob['ob_level_bearish'].iloc[-1]
                        st.write(f"Bearish OB: $ {recent_ob:.2f}")
            
            # Current signal status
            st.subheader("Signal Status")
            
            current_candle = df_indicators.iloc[-1]
            
            signal_checks = []
            
            if current_candle.get('signal_long'):
                signal_checks.append(("✅ LONG Signal", "Long entry detected"))
            elif current_candle.get('signal_short'):
                signal_checks.append(("✅ SHORT Signal", "Short entry detected"))
            
            if current_candle.get('active_session'):
                signal_checks.append(("✅ Session Active", "London or NYC session"))
            else:
                signal_checks.append(("❌ No Session", "Outside trading hours"))
            
            if current_candle.get('is_friday_after_18'):
                signal_checks.append(("⚠️ Friday After 18:00", "Low liquidity period"))
            
            # Display checks
            for status, description in signal_checks:
                st.write(f"{status}: {description}")
    
    elif mode == "Run New Backtest":
        st.header("Run New Backtest")
        
        st.info("Configure your backtest in the sidebar and click 'Run Backtest' to start.")
        
        # Show cached results if available
        if 'results' in st.session_state:
            st.success("Backtest completed! Switch to 'Backtest Results' tab to view.")


if __name__ == "__main__":
    main()