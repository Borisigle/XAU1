"""
HTML Report Generator for Backtest Results
Creates professional HTML reports with charts and tables
"""

import base64
import logging
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from xau1.dashboard.charts import ChartManager

logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """Generate professional HTML reports for backtest results"""
    
    def __init__(self):
        self.chart_manager = ChartManager()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_report(
        self,
        results: Dict,
        trades_df: pd.DataFrame,
        equity_df: pd.DataFrame,
        original_data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate complete HTML report
        
        Args:
            results: Backtest results dictionary
            trades_df: Trades DataFrame
            equity_df: Equity curve DataFrame
            original_data: Original OHLCV data
            output_path: Optional custom output path
            
        Returns:
            Path to generated HTML file
        """
        
        logger.info("Generating HTML report...")
        
        if output_path is None:
            output_dir = Path("backtest_results")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"report_{self.timestamp}.html"
        
        # Generate charts
        equity_chart = self._generate_equity_chart(equity_df, results)
        trades_dist_chart = self._generate_trades_distribution(trades_df)
        monthly_perf_chart = self._generate_monthly_performance(trades_df)
        
        # Create HTML content
        html_content = self._create_html_template(
            results=results,
            trades_df=trades_df,
            equity_chart=equity_chart,
            trades_dist_chart=trades_dist_chart,
            monthly_perf_chart=monthly_perf_chart
        )
        
        # Save HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_path}")
        return str(output_path)
    
    def _generate_equity_chart(self, equity_df: pd.DataFrame, results: Dict) -> str:
        """Generate equity curve chart"""
        
        if equity_df is None or equity_df.empty:
            return ""
        
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
                line=dict(color='#4ecdc4', width=2),
                name='Equity',
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # Drawdown
        equity_df['drawdown'] = (equity_df['equity'] / equity_df['equity'].cummax() - 1) * 100
        
        fig.add_trace(
            go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['drawdown'],
                mode='lines',
                line=dict(color='#ff6b6b', width=2),
                name='Drawdown',
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=500,
            title_text="Portfolio Performance",
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def _generate_trades_distribution(self, trades_df: pd.DataFrame) -> str:
        """Generate trades distribution chart"""
        
        if trades_df is None or trades_df.empty:
            return ""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('PnL Distribution', 'Trades by Signal Type')
        )
        
        # PnL histogram
        colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['pnl']]
        fig.add_trace(
            go.Histogram(
                x=trades_df['pnl'],
                nbinsx=20,
                marker_color=colors,
                opacity=0.7,
                name='PnL'
            ),
            row=1, col=1
        )
        
        # Trades by signal type
        if 'signal_type' in trades_df.columns:
            signal_counts = trades_df['signal_type'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=signal_counts.index,
                    y=signal_counts.values,
                    marker_color=['#4ecdc4', '#45b7d1', '#96ceb4'],
                    name='Signals'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        fig.update_xaxes(title_text="PnL ($)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Signal Type", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        
        return fig.to_html(include_plotlyjs=False)
    
    def _generate_monthly_performance(self, trades_df: pd.DataFrame) -> str:
        """Generate monthly performance chart"""
        
        if trades_df is None or trades_df.empty:
            return ""
        
        # Extract month from entry time
        trades_df['entry_month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
        
        # Calculate monthly stats
        monthly_stats = trades_df.groupby('entry_month').agg({
            'pnl': ['sum', 'count'],
            'risk_reward': 'mean'
        }).round(2)
        
        monthly_stats.columns = ['Total_PnL', 'Trade_Count', 'Avg_RR']
        monthly_stats['Win_Rate'] = trades_df.groupby('entry_month').apply(
            lambda x: (x['pnl'] > 0).sum() / len(x) * 100
        ).round(1)
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Monthly PnL', 'Monthly Trades & Win Rate')
        )
        
        # Monthly PnL bars
        colors = ['green' if pnl > 0 else 'red' for pnl in monthly_stats['Total_PnL']]
        fig.add_trace(
            go.Bar(
                x=monthly_stats.index.astype(str),
                y=monthly_stats['Total_PnL'],
                marker_color=colors,
                name='Monthly PnL'
            ),
            row=1, col=1
        )
        
        # Trades and win rate
        fig.add_trace(
            go.Bar(
                x=monthly_stats.index.astype(str),
                y=monthly_stats['Trade_Count'],
                marker_color='#45b7d1',
                name='Trade Count',
                yaxis='y3'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=monthly_stats.index.astype(str),
                y=monthly_stats['Win_Rate'],
                mode='lines+markers',
                line=dict(color='#ff6b6b', width=2),
                name='Win Rate %',
                yaxis='y4'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=500,
            template='plotly_white'
        )
        
        return fig.to_html(include_plotlyjs=False)
    
    def _create_html_template(
        self,
        results: Dict,
        trades_df: pd.DataFrame,
        equity_chart: str,
        trades_dist_chart: str,
        monthly_perf_chart: str
    ) -> str:
        """Create complete HTML template"""
        
        # Calculate trade statistics
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        total_trades = len(trades_df)
        
        best_trade = trades_df['pnl'].max() if not trades_df.empty else 0
        worst_trade = trades_df['pnl'].min() if not trades_df.empty else 0
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Performance indicators
        win_rate_target = 55
        pf_target = 1.8
        dd_target = 12
        
        win_rate_status = "✅ EXCELLENT" if results['win_rate'] * 100 >= win_rate_target else "❌ NEEDS IMPROVEMENT"
        pf_status = "✅ EXCELLENT" if results['profit_factor'] >= pf_target else "❌ NEEDS IMPROVEMENT"
        dd_status = "✅ EXCELLENT" if results['max_drawdown_pct'] <= dd_target else "⚠️ ACCEPTABLE"
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XAU1 Backtest Report - {self.timestamp}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        
        .metric-card h3 {{
            color: #667eea;
            font-size: 1.1em;
            margin-bottom: 10px;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .metric-value.green {{ color: #28a745; }}
        .metric-value.red {{ color: #dc3545; }}
        .metric-value.blue {{ color: #007bff; }}
        
        .metric-target {{
            font-size: 0.9em;
            color: #666;
        }}
        
        .section {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        .section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        .targets-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .target-item {{
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .target-item.excellent {{ background: #d4edda; border-left: 4px solid #28a745; }}
        .target-item.good {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .target-item.needs-work {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        
        .chart-container {{
            width: 100%;
            height: 500px;
            margin: 20px 0;
        }}
        
        .trades-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        .trades-table th,
        .trades-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        .trades-table th {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}
        
        .trades-table tr:hover {{
            background: #f5f5f5;
        }}
        
        .win {{ color: #28a745; font-weight: bold; }}
        .loss {{ color: #dc3545; font-weight: bold; }}
        
        .footer {{
            text-align: center;
            color: white;
            margin-top: 40px;
            padding: 20px;
        }}
        
        @media (max-width: 768px) {{
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1><i class="fas fa-chart-line"></i> XAU1 Trading Bot Report</h1>
            <div class="subtitle">
                Smart Money Concepts + Order Flow Strategy | Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </div>
        </div>
        
        <!-- Key Metrics -->
        <div class="section">
            <h2><i class="fas fa-tachometer-alt"></i> Key Performance Metrics</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Total Return</h3>
                    <div class="metric-value {'green' if results['total_return_pct'] > 0 else 'red'}">
                        {results['total_return_pct']:.1f}%
                    </div>
                </div>
                
                <div class="metric-card">
                    <h3>Win Rate</h3>
                    <div class="metric-value {'green' if results['win_rate']*100 > 50 else 'red'}">
                        {results['win_rate']*100:.1f}%
                    </div>
                    <div class="metric-target">Target: >55%</div>
                </div>
                
                <div class="metric-card">
                    <h3>Profit Factor</h3>
                    <div class="metric-value {'green' if results['profit_factor'] > 1.8 else 'red'}">
                        {results['profit_factor']:.2f}
                    </div>
                    <div class="metric-target">Target: >1.8</div>
                </div>
                
                <div class="metric-card">
                    <h3>Total PnL</h3>
                    <div class="metric-value {'green' if results['total_pnl'] > 0 else 'red'}">
                        ${results['total_pnl']:,.2f}
                    </div>
                </div>
                
                <div class="metric-card">
                    <h3>Max Drawdown</h3>
                    <div class="metric-value {'blue' if results['max_drawdown_pct'] <= 12 else 'red'}">
                        {results['max_drawdown_pct']:.1f}%
                    </div>
                    <div class="metric-target">Target: <12%</div>
                </div>
                
                <div class="metric-card">
                    <h3>Sharpe Ratio</h3>
                    <div class="metric-value {'green' if results['sharpe_ratio'] > 1 else 'red'}">
                        {results['sharpe_ratio']:.2f}
                    </div>
                    <div class="metric-target">Target: >1.0</div>
                </div>
                
                <div class="metric-card">
                    <h3>Trades per Week</h3>
                    <div class="metric-value {'green' if results['trades_per_week'] >= 3 else 'red'}">
                        {results['trades_per_week']:.1f}
                    </div>
                    <div class="metric-target">Target: 3-5</div>
                </div>
                
                <div class="metric-card">
                    <h3>Total Trades</h3>
                    <div class="metric-value blue">
                        {results['total_trades']}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Performance vs Targets -->
        <div class="section">
            <h2><i class="fas fa-bullseye"></i> Performance vs Targets</h2>
            
            <div class="targets-grid">
                <div class="target-item {'excellent' if 'EXCELLENT' in win_rate_status else 'needs-work'}">
                    <h4>Win Rate</h4>
                    <p>{win_rate_status}</p>
                </div>
                
                <div class="target-item {'excellent' if 'EXCELLENT' in pf_status else 'needs-work'}">
                    <h4>Profit Factor</h4>
                    <p>{pf_status}</p>
                </div>
                
                <div class="target-item {'excellent' if 'EXCELLENT' in dd_status else 'good'}">
                    <h4>Drawdown</h4>
                    <p>{dd_status}</p>
                </div>
            </div>
        </div>
        
        <!-- Equity Curve -->
        <div class="section">
            <h2><i class="fas fa-chart-area"></i> Equity Curve & Drawdown</h2>
            <div class="chart-container">
                {equity_chart}
            </div>
        </div>
        
        <!-- Trade Analysis -->
        <div class="section">
            <h2><i class="fas fa-analytics"></i> Trade Analysis</h2>
            
            <div class="chart-container">
                {trades_dist_chart}
            </div>
        </div>
        
        <!-- Monthly Performance -->
        <div class="section">
            <h2><i class="fas fa-calendar-alt"></i> Monthly Performance</h2>
            <div class="chart-container">
                {monthly_perf_chart}
            </div>
        </div>
        
        <!-- Trade Statistics -->
        <div class="section">
            <h2><i class="fas fa-list-alt"></i> Trade Statistics</h2>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px;">
                <div>
                    <h4>Winning Trades</h4>
                    <p><strong>{winning_trades}</strong> trades</p>
                    <p>Avg PnL: <span class="win">${avg_win:.2f}</span></p>
                    <p>Best: <span class="win">${best_trade:.2f}</span></p>
                </div>
                
                <div>
                    <h4>Losing Trades</h4>
                    <p><strong>{losing_trades}</strong> trades</p>
                    <p>Avg PnL: <span class="loss">${avg_loss:.2f}</span></p>
                    <p>Worst: <span class="loss">${worst_trade:.2f}</span></p>
                </div>
                
                <div>
                    <h4>Overall</h4>
                    <p>Win Rate: <strong>{results['win_rate']*100:.1f}%</strong></p>
                    <p>Avg R:R: <strong>{results['avg_risk_reward']:.2f}</strong></p>
                    <p>Avg PnL: <strong>${results['avg_trade_pnl']:.2f}</strong></p>
                </div>
            </div>
        </div>
        
        <!-- Recent Trades -->
        <div class="section">
            <h2><i class="fas fa-history"></i> Recent Trades (Last 20)</h2>
            
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Entry Time</th>
                        <th>Direction</th>
                        <th>Entry</th>
                        <th>Exit</th>
                        <th>PnL</th>
                        <th>R:R</th>
                        <th>Signal Type</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add recent trades rows
        recent_trades = trades_df.head(20)
        for _, trade in recent_trades.iterrows():
            pnl_class = 'win' if trade['pnl'] > 0 else 'loss'
            html_content += f"""
                    <tr>
                        <td>{trade['entry_time']}</td>
                        <td><strong>{trade['direction']}</strong></td>
                        <td>${trade['entry_price']:.2f}</td>
                        <td>${trade['exit_price']:.2f}</td>
                        <td class="{pnl_class}">${trade['pnl']:.2f}</td>
                        <td>{trade['risk_reward']:.2f}</td>
                        <td>{trade['signal_type']}</td>
                    </tr>
            """
        
        html_content += f"""
                </tbody>
            </table>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p><i class="fas fa-robot"></i> XAU1 Trading Bot | Generated with Python & Plotly</p>
            <p>This report contains simulated trading results. Past performance is not indicative of future results.</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_content