"""
Metrics display components for Streamlit dashboard
"""

import logging
from typing import Dict, Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


class MetricsDisplay:
    """Display performance metrics in Streamlit"""
    
    @staticmethod
    def display_kpi_metrics(results: Dict):
        """Display KPI cards"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Return",
                value=f"{results.get('total_return_pct', 0):.1f}%",
                delta=None
            )
        
        with col2:
            win_rate = results.get('win_rate', 0) * 100
            target_win_rate = 55
            st.metric(
                label="Win Rate",
                value=f"{win_rate:.1f}%",
                delta=f"{'✅' if win_rate >= target_win_rate else '❌'} Target: {target_win_rate}%"
            )
        
        with col3:
            profit_factor = results.get('profit_factor', 0)
            target_pf = 1.8
            st.metric(
                label="Profit Factor",
                value=f"{profit_factor:.2f}",
                delta=f"{'✅' if profit_factor >= target_pf else '❌'} Target: {target_pf}"
            )
        
        with col4:
            max_dd = results.get('max_drawdown_pct', 0)
            target_dd = 12
            st.metric(
                label="Max Drawdown",
                value=f"{max_dd:.1f}%",
                delta=f"{'✅' if max_dd <= target_dd else '❌'} Target: <{target_dd}%"
            )
        
        # Second row of metrics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            trades_per_week = results.get('trades_per_week', 0)
            target_tpw = 3
            st.metric(
                label="Trades/Week",
                value=f"{trades_per_week:.1f}",
                delta=f"{'✅' if trades_per_week >= target_tpw else '❌'} Target: {target_tpw}"
            )
        
        with col6:
            sharpe = results.get('sharpe_ratio', 0)
            target_sharpe = 1.0
            st.metric(
                label="Sharpe Ratio",
                value=f"{sharpe:.2f}",
                delta=f"{'✅' if sharpe >= target_sharpe else '❌'} Target: {target_sharpe}"
            )
        
        with col7:
            recovery = results.get('recovery_factor', 0)
            target_recovery = 1.5
            st.metric(
                label="Recovery Factor",
                value=f"{recovery:.2f}",
                delta=f"{'✅' if recovery >= target_recovery else '❌'} Target: {target_recovery}"
            )
        
        with col8:
            avg_rr = results.get('avg_risk_reward', 0)
            st.metric(
                label="Avg R:R",
                value=f"{avg_rr:.2f}",
                delta=f"Min: 2.0"
            )
    
    @staticmethod
    def display_trade_statistics(trades_df: pd.DataFrame):
        """Display detailed trade statistics"""
        
        if trades_df.empty:
            st.warning("No trades to analyze")
            return
        
        st.subheader("Trade Statistics")
        
        # Win/Loss breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Winning Trades**")
            winning_trades = trades_df[trades_df['pnl'] > 0]
            if not winning_trades.empty:
                st.write(f"Count: {len(winning_trades)}")
                st.write(f"Avg PnL: ${winning_trades['pnl'].mean():.2f}")
                st.write(f"Best Trade: ${winning_trades['pnl'].max():.2f}")
                st.write(f"Avg R:R: {winning_trades['risk_reward'].mean():.2f}")
            else:
                st.write("No winning trades")
        
        with col2:
            st.write("**Losing Trades**")
            losing_trades = trades_df[trades_df['pnl'] < 0]
            if not losing_trades.empty:
                st.write(f"Count: {len(losing_trades)}")
                st.write(f"Avg Loss: ${losing_trades['pnl'].mean():.2f}")
                st.write(f"Worst Loss: ${losing_trades['pnl'].min():.2f}")
                st.write(f"Avg R:R: {losing_trades['risk_reward'].mean():.2f}")
            else:
                st.write("No losing trades")
        
        # Signal type performance
        st.subheader("Performance by Signal Type")
        if 'signal_type' in trades_df.columns:
            signal_perf = trades_df.groupby('signal_type').agg({
                'pnl': ['count', 'sum', 'mean'],
                'risk_reward': 'mean'
            }).round(2)
            
            signal_perf.columns = ['Trades', 'Total_PnL', 'Avg_PnL', 'Avg_RR']
            st.dataframe(signal_perf)
        
        # Confluence score analysis
        st.subheader("Confluence Score Analysis")
        if 'confluence_score' in trades_df.columns:
            confluence_perf = trades_df.groupby('confluence_score').agg({
                'pnl': ['count', 'sum', 'mean'],
                'risk_reward': 'mean'
            }).round(2)
            
            confluence_perf.columns = ['Trades', 'Total_PnL', 'Avg_PnL', 'Avg_RR']
            st.dataframe(confluence_perf)
    
    @staticmethod
    def display_session_performance(trades_df: pd.DataFrame):
        """Display performance by trading session"""
        
        if trades_df.empty or 'entry_time' not in trades_df.columns:
            return
        
        st.subheader("Session Performance")
        
        # Extract hour from entry time
        trades_df['hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
        
        # Define sessions
        def get_session(hour):
            if 13 <= hour < 20:  # London: 13:00-20:00 UTC
                return 'London'
            elif 13.5 <= hour < 21:  # NYC: 13:30-21:00 UTC
                return 'NYC'
            else:
                return 'Other'
        
        trades_df['session'] = trades_df['hour'].apply(get_session)
        
        # Calculate performance by session
        session_perf = trades_df.groupby('session').agg({
            'pnl': ['count', 'sum', 'mean'],
            'risk_reward': 'mean'
        }).round(2)
        
        if not session_perf.empty:
            session_perf.columns = ['Trades', 'Total_PnL', 'Avg_PnL', 'Avg_RR']
            st.dataframe(session_perf)
            
            # Visualize
            fig_session = session_perf['Avg_PnL'].plot(kind='bar', title='Avg PnL by Session')
            st.pyplot(fig_session.figure)
        
        # Hourly performance heatmap
        st.subheader("Hourly Performance Heatmap")
        hourly_perf = trades_df.groupby('hour')['pnl'].mean().reset_index()
        
        if not hourly_perf.empty:
            # Simple bar chart for hourly performance
            fig_hourly = hourly_perf.plot(
                x='hour', 
                y='pnl', 
                kind='bar', 
                title='Average PnL by Hour (UTC)',
                figsize=(12, 6)
            )
            plt = hourly_perf.plot
            st.pyplot(fig_hourly.figure)
    
    @staticmethod
    def display_strategy_parameters(config: Dict):
        """Display current strategy parameters"""
        
        st.subheader("Strategy Parameters")
        
        with st.expander("View Configuration"):
            # Risk management
            st.write("**Risk Management**")
            risk_config = config.get("risk_management", {})
            st.write(f"Risk per Trade: {risk_config.get('position_size_percentage', 1)}%")
            st.write(f"Stop Loss: {risk_config.get('stop_loss_pips', 30)} pips")
            st.write(f"Take Profit 1: {risk_config.get('take_profit1_pips', 50)} pips")
            st.write(f"Take Profit 2: {risk_config.get('take_profit2_pips', 100)} pips")
            st.write(f"Min R/R Ratio: {risk_config.get('min_risk_reward_ratio', 2.0)}")
            
            # Signal types
            st.write("**Signal Types**")
            entry_rules = config.get("entry_rules", {})
            for rule_name, rule_config in entry_rules.items():
                st.write(f"{rule_name}: {'✅ Enabled' if rule_config.get('enabled') else '❌ Disabled'}")
                st.write(f"  Min Confluence: {rule_config.get('min_confluence', 3)}")
    
    @staticmethod
    def display_data_quality_info(df: pd.DataFrame):
        """Display data quality information"""
        
        st.subheader("Data Quality")
        
        if df is None or df.empty:
            st.error("No data available")
            return
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Rows:** {len(df):,}")
        
        with col2:
            st.write(f"**Date Range:**")
            st.write(f"{df.index.min()} to {df.index.max()}")
        
        with col3:
            st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
        
        # Data completeness
        with st.expander("Data Statistics"):
            st.write(df.describe())