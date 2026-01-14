"""
Chart components for Streamlit dashboard
Interactive Plotly charts for XAUUSDT analysis
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class ChartManager:
    """Manage Plotly chart generation"""
    
    def __init__(self):
        self.colors = {
            'bullish': '#00ff00',
            'bearish': '#ff0000',
            'neutral': '#808080',
            'fvg_bullish': 'rgba(0, 255, 0, 0.2)',
            'fvg_bearish': 'rgba(255, 0, 0, 0.2)',
            'ob_bullish': 'rgba(0, 255, 150, 0.3)',
            'ob_bearish': 'rgba(255, 150, 0, 0.3)',
            'swing_high': '#ff6b6b',
            'swing_low': '#4ecdc4',
            'equity': '#4ecdc4',
            'drawdown': '#ff6b6b'
        }
    
    def create_price_chart(
        self,
        df: pd.DataFrame,
        trades: Optional[pd.DataFrame] = None,
        timeframe: str = "15m"
    ) -> go.Figure:
        """Create main price chart with SMC overlays"""
        
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price & SMC Levels', 'Volume', 'RSI'),
            row_width=[0.6, 0.2, 0.2]
        )
        
        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='XAUUSDT',
                increasing_line_color=self.colors['bullish'],
                decreasing_line_color=self.colors['bearish']
            ),
            row=1, col=1
        )
        
        # Swing points
        if 'swing_high' in df.columns:
            swing_highs = df[df['swing_high'].notna()]
            if not swing_highs.empty:
                fig.add_trace(
                    go.Scatter(
                        x=swing_highs.index,
                        y=swing_highs['swing_high'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=10,
                            color=self.colors['swing_high'],
                            line=dict(width=2, color='white')
                        ),
                        name='Swing Highs',
                        hovertemplate='Swing High<br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        if 'swing_low' in df.columns:
            swing_lows = df[df['swing_low'].notna()]
            if not swing_lows.empty:
                fig.add_trace(
                    go.Scatter(
                        x=swing_lows.index,
                        y=swing_lows['swing_low'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=10,
                            color=self.colors['swing_low'],
                            line=dict(width=2, color='white')
                        ),
                        name='Swing Lows',
                        hovertemplate='Swing Low<br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Fair Value Gaps
        self._add_fvg_overlays(fig, df)
        
        # Order Blocks
        self._add_ob_overlays(fig, df)
        
        # BOS markers
        if 'bos_bullish' in df.columns:
            bos_bullish = df[df['bos_bullish'] == True]
            if not bos_bullish.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bos_bullish.index,
                        y=bos_bullish['close'],
                        mode='markers+text',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color=self.colors['bullish'],
                            line=dict(width=2, color='white')
                        ),
                        text='BOS',
                        textposition='top center',
                        name='BOS Bullish',
                        hovertemplate='BOS Bullish<br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        if 'bos_bearish' in df.columns:
            bos_bearish = df[df['bos_bearish'] == True]
            if not bos_bearish.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bos_bearish.index,
                        y=bos_bearish['close'],
                        mode='markers+text',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color=self.colors['bearish'],
                            line=dict(width=2, color='white')
                        ),
                        text='BOS',
                        textposition='bottom center',
                        name='BOS Bearish',
                        hovertemplate='BOS Bearish<br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Trade entries
        if trades is not None and not trades.empty:
            long_trades = trades[trades['direction'] == 'LONG']
            short_trades = trades[trades['direction'] == 'SHORT']
            
            if not long_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=long_trades['entry_time'],
                        y=long_trades['entry_price'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=15,
                            color='green',
                            line=dict(width=2, color='white')
                        ),
                        name='Long Entry',
                        hovertemplate='LONG<br>Entry: $%{y:.2f}<br>%{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            if not short_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=short_trades['entry_time'],
                        y=short_trades['entry_price'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=15,
                            color='red',
                            line=dict(width=2, color='white')
                        ),
                        name='Short Entry',
                        hovertemplate='SHORT<br>Entry: $%{y:.2f}<br>%{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Volume
        colors = ['rgba(0,255,0,0.5)' if row['close'] >= row['open'] else 'rgba(255,0,0,0.5)' for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                marker_color=colors,
                name='Volume',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # High volume bars (potential order blocks)
        if 'high_volume' in df.columns:
            high_vol = df[df['high_volume'] == True]
            if not high_vol.empty:
                fig.add_trace(
                    go.Bar(
                        x=high_vol.index,
                        y=high_vol['volume'],
                        marker_color='cyan',
                        name='High Volume',
                        opacity=0.7
                    ),
                    row=2, col=1
                )
        
        # RSI
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['rsi'],
                    mode='lines',
                    line=dict(color='purple', width=1),
                    name='RSI',
                    showlegend=False
                ),
                row=3, col=1
            )
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
        
        # Session markers
        if 'active_session' in df.columns:
            session_bars = df[df['active_session'] == True]
            if not session_bars.empty:
                for _, bar in session_bars.iterrows():
                    fig.add_vrect(
                        x0=bar.name,
                        x1=bar.name + pd.Timedelta(minutes=15),
                        fillcolor="rgba(255,255,255,0.05)",
                        layer="below",
                        line_width=0,
                        row=1, col=1
                    )
        
        # Update layout
        fig.update_layout(
            title=f'XAUUSDT Price Analysis - {timeframe}',
            xaxis_title='Date/Time',
            yaxis_title='Price (USD)',
            template='plotly_dark',
            height=800,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        
        return fig
    
    def _add_fvg_overlays(self, fig: go.Figure, df: pd.DataFrame):
        """Add Fair Value Gap overlays to chart"""
        
        # Bullish FVG
        if 'fvg_bullish' in df.columns:
            fvg_bullish_bars = df[df['fvg_bullish'] == True]
            for _, bar in fvg_bullish_bars.iterrows():
                # Get the FVG range (current low to high.shift(2))
                fvg_high = bar['low']
                fvg_low = df.loc[:bar.name].iloc[-3]['high']  # 2 candles ago
                
                fig.add_hrect(
                    y0=fvg_low,
                    y1=fvg_high,
                    fillcolor=self.colors['fvg_bullish'],
                    layer="below",
                    line_width=0,
                    opacity=0.3
                )
        
        # Bearish FVG
        if 'fvg_bearish' in df.columns:
            fvg_bearish_bars = df[df['fvg_bearish'] == True]
            for _, bar in fvg_bearish_bars.iterrows():
                # Get the FVG range (current high to low.shift(2))
                fvg_low = bar['high']
                fvg_high = df.loc[:bar.name].iloc[-3]['low']  # 2 candles ago
                
                fig.add_hrect(
                    y0=fvg_low,
                    y1=fvg_high,
                    fillcolor=self.colors['fvg_bearish'],
                    layer="below",
                    line_width=0,
                    opacity=0.3
                )
    
    def _add_ob_overlays(self, fig: go.Figure, df: pd.DataFrame):
        """Add Order Block overlays to chart"""
        
        # Bullish Order Block
        if 'ob_bullish' in df.columns:
            ob_bullish_bars = df[df['ob_bullish'] == True]
            for _, bar in ob_bullish_bars.iterrows():
                # Highlight the candle body
                body_low = min(bar['open'], bar['close'])
                body_high = max(bar['open'], bar['close'])
                
                fig.add_hrect(
                    y0=body_low,
                    y1=body_high,
                    fillcolor=self.colors['ob_bullish'],
                    layer="below",
                    line_width=1,
                    line_color=self.colors['bullish'],
                    opacity=0.4
                )
        
        # Bearish Order Block
        if 'ob_bearish' in df.columns:
            ob_bearish_bars = df[df['ob_bearish'] == True]
            for _, bar in ob_bearish_bars.iterrows():
                # Highlight the candle body
                body_low = min(bar['open'], bar['close'])
                body_high = max(bar['open'], bar['close'])
                
                fig.add_hrect(
                    y0=body_low,
                    y1=body_high,
                    fillcolor=self.colors['ob_bearish'],
                    layer="below",
                    line_width=1,
                    line_color=self.colors['bearish'],
                    opacity=0.4
                )
    
    def create_equity_curve_chart(self, equity_df: pd.DataFrame) -> go.Figure:
        """Create equity curve with drawdown"""
        
        if equity_df.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Equity Curve', 'Drawdown'),
            row_heights=[0.7, 0.3]
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['equity'],
                mode='lines',
                line=dict(color=self.colors['equity'], width=2),
                name='Equity',
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # Calculate drawdown
        equity_df['drawdown'] = (equity_df['equity'] / equity_df['equity'].cummax() - 1) * 100
        
        fig.add_trace(
            go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['drawdown'],
                mode='lines',
                line=dict(color=self.colors['drawdown'], width=2),
                name='Drawdown',
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Portfolio Performance',
            template='plotly_dark',
            height=600,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        return fig
    
    def create_metrics_bar_chart(self, metrics: dict) -> go.Figure:
        """Create performance metrics bar chart"""
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color=[self.colors['bullish'] if v > 0 else self.colors['bearish'] for v in metrics.values()]
            )
        ])
        
        fig.update_layout(
            title='Performance Metrics',
            template='plotly_dark',
            height=400,
            xaxis_title='Metrics',
            yaxis_title='Value'
        )
        
        return fig
    
    def create_trades_table(self, trades_df: pd.DataFrame) -> go.Figure:
        """Create trades table visualization"""
        
        if trades_df.empty:
            return go.Figure()
        
        # Prepare trades data for table
        display_df = trades_df.head(50).copy()
        display_df['entry_time'] = display_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['exit_time'] = display_df['exit_time'].dt.strftime('%Y-%m-%d %H:%M')
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Entry Time', 'Exit Time', 'Dir', 'Entry', 'Exit', 'PnL', 'R:R'],
                fill_color='darkgrey',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[
                    display_df['entry_time'],
                    display_df['exit_time'],
                    display_df['direction'],
                    display_df['entry_price'].round(2),
                    display_df['exit_price'].round(2),
                    display_df['pnl'].round(2),
                    display_df['risk_reward'].round(2)
                ],
                fill_color=[
                    ['green' if pnl > 0 else 'red' for pnl in display_df['pnl']]
                ],
                font=dict(color='white', size=11)
            )
        )])
        
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        return fig