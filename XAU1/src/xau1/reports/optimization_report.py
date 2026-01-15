"""
XAU1 Optimization Report Generator
Creates comprehensive HTML report with visualizations
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

logger = logging.getLogger(__name__)


class OptimizationReportGenerator:
    """Generate comprehensive optimization report"""
    
    def __init__(self, results_path: str = "reports/"):
        self.results_path = results_path
        
    def generate_report(self, optimization_results: List[Dict], output_path: str = None) -> str:
        """
        Generate comprehensive optimization report
        
        Args:
            optimization_results: List of optimization results
            output_path: Output file path
            
        Returns:
            Path to generated report
        """
        logger.info("Generating optimization report...")
        
        if not optimization_results:
            logger.error("No optimization results provided")
            return ""
        
        if output_path is None:
            output_path = f"{self.results_path}/optimization_report.html"
        
        # Create output directory
        os.makedirs(self.results_path, exist_ok=True)
        
        # Generate report sections
        report_sections = self._generate_report_sections(optimization_results)
        
        # Generate HTML
        html_content = self._generate_html_report(report_sections, optimization_results)
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Optimization report saved to {output_path}")
        return output_path
    
    def _generate_report_sections(self, results: List[Dict]) -> Dict:
        """Generate all report sections"""
        sections = {}
        
        # Executive summary
        sections['executive_summary'] = self._generate_executive_summary(results)
        
        # Top configurations table
        sections['top_configurations'] = self._generate_top_configurations_table(results)
        
        # Visualization sections
        sections['visualizations'] = {
            'trades_vs_winrate': self._create_trades_vs_winrate_chart(results),
            'profit_factor_vs_drawdown': self._create_pf_vs_dd_chart(results),
            'parameter_heatmap': self._create_parameter_heatmap(results),
            'score_distribution': self._create_score_distribution_chart(results)
        }
        
        # Target analysis
        sections['target_analysis'] = self._analyze_target_achievement(results)
        
        # Sensitivity insights
        sections['sensitivity_insights'] = self._analyze_sensitivity(results)
        
        # Recommendations
        sections['recommendations'] = self._generate_recommendations(results)
        
        return sections
    
    def _generate_executive_summary(self, results: List[Dict]) -> Dict:
        """Generate executive summary"""
        total_configs = len(results)
        best_result = results[0] if results else None
        
        # Calculate statistics
        scores = [r['score'] for r in results]
        trades_per_week = [r['metrics']['trades_per_week'] for r in results]
        win_rates = [r['metrics']['win_rate'] for r in results]
        profit_factors = [r['metrics']['profit_factor'] for r in results]
        max_drawdowns = [r['metrics']['max_drawdown'] for r in results]
        sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in results]
        
        # Target achievement
        target_achievements = {
            'trades_per_week_3': sum(1 for t in trades_per_week if 2.5 <= t <= 3.5) / total_configs * 100,
            'win_rate_56': sum(1 for wr in win_rates if wr >= 0.56) / total_configs * 100,
            'profit_factor_22': sum(1 for pf in profit_factors if pf >= 2.2) / total_configs * 100,
            'max_drawdown_10': sum(1 for dd in max_drawdowns if dd <= 10.0) / total_configs * 100,
            'sharpe_14': sum(1 for sr in sharpe_ratios if sr >= 1.4) / total_configs * 100,
            'all_targets': sum(1 for r in results if r['meets_targets']) / total_configs * 100
        }
        
        return {
            'total_configurations_tested': total_configs,
            'best_score': best_result['score'] if best_result else 0,
            'best_config': best_result['params'] if best_result else {},
            'best_metrics': best_result['metrics'] if best_result else {},
            'statistics': {
                'score': {
                    'mean': sum(scores) / len(scores),
                    'max': max(scores),
                    'min': min(scores),
                    'std': (sum([(s - sum(scores)/len(scores))**2 for s in scores]) / len(scores))**0.5
                },
                'trades_per_week': {
                    'mean': sum(trades_per_week) / len(trades_per_week),
                    'closest_to_3': min(trades_per_week, key=lambda x: abs(x - 3.0))
                }
            },
            'target_achievements': target_achievements,
            'optimization_success': target_achievements['all_targets'] >= 10  # At least 10% of configs meet all targets
        }
    
    def _generate_top_configurations_table(self, results: List[Dict], top_n: int = 10) -> str:
        """Generate top configurations table HTML"""
        table_html = """
        <table class="config-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Score</th>
                    <th>Min Factors</th>
                    <th>RR Ratio</th>
                    <th>SL Pips</th>
                    <th>TP2 Pips</th>
                    <th>Trades/Wk</th>
                    <th>Win Rate</th>
                    <th>Profit Factor</th>
                    <th>Max DD</th>
                    <th>Sharpe</th>
                    <th>Targets</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for i, result in enumerate(results[:top_n], 1):
            params = result['params']
            metrics = result['metrics']
            
            # Color coding for target achievement
            target_class = "target-met" if result['meets_targets'] else "target-missed"
            target_icon = "✅" if result['meets_targets'] else "❌"
            
            table_html += f"""
                <tr class="{target_class}">
                    <td>{i}</td>
                    <td class="score">{result['score']:.2f}</td>
                    <td>{params['min_factors']}</td>
                    <td>{params['min_rr_ratio']:.1f}</td>
                    <td>{params['stop_loss_pips']}</td>
                    <td>{params['take_profit2_pips']}</td>
                    <td>{metrics['trades_per_week']:.1f}</td>
                    <td>{metrics['win_rate']*100:.1f}%</td>
                    <td>{metrics['profit_factor']:.2f}</td>
                    <td>{metrics['max_drawdown']:.1f}%</td>
                    <td>{metrics['sharpe_ratio']:.2f}</td>
                    <td>{target_icon}</td>
                </tr>
            """
        
        table_html += """
            </tbody>
        </table>
        """
        
        return table_html
    
    def _create_trades_vs_winrate_chart(self, results: List[Dict]) -> str:
        """Create trades per week vs win rate scatter plot"""
        # Prepare data
        x_data = [r['metrics']['trades_per_week'] for r in results]
        y_data = [r['metrics']['win_rate'] * 100 for r in results]  # Convert to percentage
        colors = [r['score'] for r in results]
        sizes = [max(5, min(20, r['score'] * 2)) for r in results]  # Size based on score
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                colorscale='Viridis',
                colorbar=dict(title="Score"),
                line=dict(width=1, color='black')
            ),
            text=[f"Score: {r['score']:.2f}<br>Trades/Wk: {r['metrics']['trades_per_week']:.1f}<br>WR: {r['metrics']['win_rate']*100:.1f}%<br>PF: {r['metrics']['profit_factor']:.2f}" for r in results],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add target zones
        fig.add_hrect(y0=56, y1=70, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Target WR ≥56%", annotation_position="top right")
        fig.add_vrect(x0=2.5, x1=3.5, fillcolor="blue", opacity=0.1, line_width=0, annotation_text="Target 3±0.5 trades/wk", annotation_position="top left")
        
        # Add best configuration
        best_result = results[0]
        fig.add_trace(go.Scatter(
            x=[best_result['metrics']['trades_per_week']],
            y=[best_result['metrics']['win_rate'] * 100],
            mode='markers',
            marker=dict(size=25, color='red', symbol='star', line=dict(width=2, color='darkred')),
            name='Best Configuration',
            showlegend=True
        ))
        
        fig.update_layout(
            title="Trades per Week vs Win Rate",
            xaxis_title="Trades per Week",
            yaxis_title="Win Rate (%)",
            template="plotly_white",
            width=800,
            height=500
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="trades-winrate-chart")
    
    def _create_pf_vs_dd_chart(self, results: List[Dict]) -> str:
        """Create profit factor vs max drawdown chart"""
        # Prepare data
        x_data = [r['metrics']['max_drawdown'] for r in results]
        y_data = [r['metrics']['profit_factor'] for r in results]
        colors = [r['score'] for r in results]
        sizes = [max(5, min(20, r['metrics']['trades_per_week'] * 3)) for r in results]
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                colorscale='Plasma',
                colorbar=dict(title="Score"),
                line=dict(width=1, color='black')
            ),
            text=[f"Score: {r['score']:.2f}<br>DD: {r['metrics']['max_drawdown']:.1f}%<br>PF: {r['metrics']['profit_factor']:.2f}<br>WR: {r['metrics']['win_rate']*100:.1f}%" for r in results],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add target zones
        fig.add_hrect(y0=2.2, y1=5.0, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Target PF ≥2.2", annotation_position="top right")
        fig.add_vrect(x0=0, x1=10, fillcolor="blue", opacity=0.1, line_width=0, annotation_text="Target DD ≤10%", annotation_position="top left")
        
        # Add best configuration
        best_result = results[0]
        fig.add_trace(go.Scatter(
            x=[best_result['metrics']['max_drawdown']],
            y=[best_result['metrics']['profit_factor']],
            mode='markers',
            marker=dict(size=25, color='red', symbol='star', line=dict(width=2, color='darkred')),
            name='Best Configuration',
            showlegend=True
        ))
        
        fig.update_layout(
            title="Profit Factor vs Max Drawdown",
            xaxis_title="Max Drawdown (%)",
            yaxis_title="Profit Factor",
            template="plotly_white",
            width=800,
            height=500
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="pf-dd-chart")
    
    def _create_parameter_heatmap(self, results: List[Dict]) -> str:
        """Create parameter correlation heatmap"""
        # Create correlation matrix
        param_cols = ['min_factors', 'min_rr_ratio', 'stop_loss_pips', 'take_profit2_pips']
        metric_cols = ['trades_per_week', 'win_rate', 'profit_factor', 'max_drawdown', 'score']
        
        # Create combined dataframe
        df_data = []
        for result in results:
            row = {**result['params'], **result['metrics']}
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Calculate correlations
        corr_matrix = df[param_cols + metric_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Parameter-Metric Correlation Heatmap",
            template="plotly_white",
            width=700,
            height=700
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="param-heatmap")
    
    def _create_score_distribution_chart(self, results: List[Dict]) -> str:
        """Create score distribution histogram"""
        scores = [r['score'] for r in results]
        
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=20,
            name="Score Distribution",
            marker_color='lightblue',
            marker_line=dict(color='black', width=1)
        ))
        
        # Add mean line
        mean_score = sum(scores) / len(scores)
        fig.add_vline(x=mean_score, line_dash="dash", line_color="red", annotation_text=f"Mean: {mean_score:.2f}")
        
        # Add best score line
        best_score = max(scores)
        fig.add_vline(x=best_score, line_dash="dot", line_color="green", annotation_text=f"Best: {best_score:.2f}")
        
        fig.update_layout(
            title="Optimization Score Distribution",
            xaxis_title="Score",
            yaxis_title="Frequency",
            template="plotly_white",
            width=600,
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="score-distribution")
    
    def _analyze_target_achievement(self, results: List[Dict]) -> Dict:
        """Analyze target achievement rates"""
        total_configs = len(results)
        
        # Individual target achievements
        target_achievements = {
            'trades_per_week_3': {
                'achieved': sum(1 for r in results if 2.5 <= r['metrics']['trades_per_week'] <= 3.5),
                'rate': sum(1 for r in results if 2.5 <= r['metrics']['trades_per_week'] <= 3.5) / total_configs * 100,
                'description': 'Trades per week in range 2.5-3.5'
            },
            'win_rate_56': {
                'achieved': sum(1 for r in results if r['metrics']['win_rate'] >= 0.56),
                'rate': sum(1 for r in results if r['metrics']['win_rate'] >= 0.56) / total_configs * 100,
                'description': 'Win rate ≥ 56%'
            },
            'profit_factor_22': {
                'achieved': sum(1 for r in results if r['metrics']['profit_factor'] >= 2.2),
                'rate': sum(1 for r in results if r['metrics']['profit_factor'] >= 2.2) / total_configs * 100,
                'description': 'Profit factor ≥ 2.2'
            },
            'max_drawdown_10': {
                'achieved': sum(1 for r in results if r['metrics']['max_drawdown'] <= 10.0),
                'rate': sum(1 for r in results if r['metrics']['max_drawdown'] <= 10.0) / total_configs * 100,
                'description': 'Max drawdown ≤ 10%'
            },
            'sharpe_14': {
                'achieved': sum(1 for r in results if r['metrics']['sharpe_ratio'] >= 1.4),
                'rate': sum(1 for r in results if r['metrics']['sharpe_ratio'] >= 1.4) / total_configs * 100,
                'description': 'Sharpe ratio ≥ 1.4'
            }
        }
        
        # Overall target achievement
        all_targets_achieved = sum(1 for r in results if r['meets_targets'])
        overall_rate = all_targets_achieved / total_configs * 100
        
        return {
            'individual_targets': target_achievements,
            'overall_achievement': {
                'configs_meeting_all': all_targets_achieved,
                'overall_rate': overall_rate,
                'success': overall_rate >= 10  # At least 10% success rate
            }
        }
    
    def _analyze_sensitivity(self, results: List[Dict]) -> Dict:
        """Analyze parameter sensitivity"""
        param_ranges = {
            'min_factors': {'min': min(r['params']['min_factors'] for r in results),
                          'max': max(r['params']['min_factors'] for r in results)},
            'min_rr_ratio': {'min': min(r['params']['min_rr_ratio'] for r in results),
                           'max': max(r['params']['min_rr_ratio'] for r in results)},
            'stop_loss_pips': {'min': min(r['params']['stop_loss_pips'] for r in results),
                              'max': max(r['params']['stop_loss_pips'] for r in results)},
            'take_profit2_pips': {'min': min(r['params']['tp2_pips'] for r in results),
                                 'max': max(r['params']['tp2_pips'] for r in results)}
        }
        
        # Calculate impact of each parameter
        parameter_insights = {}
        
        for param_name in ['min_factors', 'min_rr_ratio', 'stop_loss_pips', 'tp2_pips']:
            # Group results by parameter value
            param_groups = {}
            for result in results:
                param_value = result['params'][param_name]
                if param_value not in param_groups:
                    param_groups[param_value] = []
                param_groups[param_value].append(result)
            
            # Calculate average metrics for each parameter value
            param_metrics = {}
            for param_value, group_results in param_groups.items():
                avg_score = sum(r['score'] for r in group_results) / len(group_results)
                avg_trades = sum(r['metrics']['trades_per_week'] for r in group_results) / len(group_results)
                avg_wr = sum(r['metrics']['win_rate'] for r in group_results) / len(group_results)
                
                param_metrics[param_value] = {
                    'avg_score': avg_score,
                    'avg_trades_per_week': avg_trades,
                    'avg_win_rate': avg_wr,
                    'sample_size': len(group_results)
                }
            
            parameter_insights[param_name] = param_metrics
        
        return {
            'parameter_ranges': param_ranges,
            'parameter_insights': parameter_insights,
            'key_findings': self._extract_key_findings(parameter_insights, results)
        }
    
    def _extract_key_findings(self, parameter_insights: Dict, results: List[Dict]) -> List[str]:
        """Extract key findings from parameter analysis"""
        findings = []
        
        # Find best performing parameter values
        for param_name, metrics in parameter_insights.items():
            best_value = max(metrics.keys(), key=lambda k: metrics[k]['avg_score'])
            worst_value = min(metrics.keys(), key=lambda k: metrics[k]['avg_score'])
            
            best_score = metrics[best_value]['avg_score']
            worst_score = metrics[worst_value]['avg_score']
            score_difference = best_score - worst_score
            
            findings.append(f"{param_name}: Best value {best_value} (score {best_score:.2f}) vs worst {worst_value} (score {worst_score:.2f}), difference: {score_difference:.2f}")
        
        # Find configurations meeting targets
        target_configs = [r for r in results if r['meets_targets']]
        if target_configs:
            common_params = self._find_common_parameters(target_configs)
            findings.append(f"Common parameters in successful configs: {common_params}")
        
        return findings
    
    def _find_common_parameters(self, target_configs: List[Dict]) -> Dict:
        """Find common parameters in target-meeting configurations"""
        if not target_configs:
            return {}
        
        common_params = {}
        
        for param_name in ['min_factors', 'min_rr_ratio', 'stop_loss_pips', 'tp2_pips']:
            param_values = [r['params'][param_name] for r in target_configs]
            most_common = max(set(param_values), key=param_values.count)
            frequency = param_values.count(most_common)
            
            common_params[param_name] = {
                'most_common_value': most_common,
                'frequency': frequency,
                'percentage': frequency / len(target_configs) * 100
            }
        
        return common_params
    
    def _generate_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []
        
        best_result = results[0]
        target_configs = [r for r in results if r['meets_targets']]
        
        # Primary recommendation
        recommendations.append(
            f"🎯 PRIMARY RECOMMENDATION: Use configuration with min_factors={best_result['params']['min_factors']}, "
            f"RR_ratio={best_result['params']['min_rr_ratio']}, SL_pips={best_result['params']['stop_loss_pips']} "
            f"(Score: {best_result['score']:.2f}/10, meets all targets: {'Yes' if best_result['meets_targets'] else 'No'})"
        )
        
        # Parameter insights
        if best_result['params']['min_factors'] == 4:
            recommendations.append("📊 MIN_FACTORS=4 provides optimal balance between signal quality and frequency")
        elif best_result['params']['min_factors'] == 3:
            recommendations.append("📊 MIN_FACTORS=3 offers good compromise between selectivity and trade frequency")
        
        if 2.0 <= best_result['params']['min_rr_ratio'] <= 2.2:
            recommendations.append("⚖️ RR_RATIO 2.0-2.2 provides optimal risk-reward balance for XAU trading")
        
        if best_result['params']['stop_loss_pips'] <= 32:
            recommendations.append("🛡️ Stop loss 25-32 pips appropriate for XAU volatility patterns")
        
        # Alternative configurations
        if len(target_configs) >= 3:
            recommendations.append(f"🔄 ALTERNATIVE: {len(target_configs)} configurations meet all targets - consider portfolio approach")
        
        # Risk considerations
        if best_result['metrics']['max_drawdown'] > 8:
            recommendations.append("⚠️ Risk Note: Consider position sizing adjustments due to higher drawdown potential")
        
        if best_result['metrics']['sharpe_ratio'] > 1.6:
            recommendations.append("✅ Sharpe ratio >1.6 indicates excellent risk-adjusted returns")
        
        # Implementation notes
        recommendations.append("🔧 IMPLEMENTATION: Monitor live performance against backtested metrics")
        recommendations.append("📈 Consider walkforward validation before live deployment")
        
        return recommendations
    
    def _generate_html_report(self, sections: Dict, results: List[Dict]) -> str:
        """Generate complete HTML report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>XAU1 Strategy Optimization Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    border-bottom: 3px solid #2c3e50;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    color: #2c3e50;
                    margin: 0;
                    font-size: 2.5em;
                }}
                .timestamp {{
                    color: #7f8c8d;
                    font-style: italic;
                }}
                .section {{
                    margin: 30px 0;
                    padding: 20px;
                    border-left: 4px solid #3498db;
                    background-color: #f8f9fa;
                }}
                .section h2 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .executive-summary {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 25px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 1.8em;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .metric-label {{
                    color: #7f8c8d;
                    font-size: 0.9em;
                }}
                .config-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    background: white;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .config-table th {{
                    background-color: #34495e;
                    color: white;
                    padding: 12px;
                    text-align: left;
                    font-weight: bold;
                }}
                .config-table td {{
                    padding: 10px 12px;
                    border-bottom: 1px solid #ecf0f1;
                }}
                .config-table tr:hover {{
                    background-color: #f8f9fa;
                }}
                .target-met {{
                    background-color: #d5f4e6;
                }}
                .target-missed {{
                    background-color: #ffeaa7;
                }}
                .score {{
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .chart-container {{
                    margin: 20px 0;
                    padding: 15px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .recommendations {{
                    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                    padding: 25px;
                    border-radius: 10px;
                    margin: 30px 0;
                }}
                .recommendations h3 {{
                    color: #2c3e50;
                    margin-top: 0;
                }}
                .recommendation-item {{
                    margin: 10px 0;
                    padding: 10px;
                    background: rgba(255,255,255,0.7);
                    border-radius: 5px;
                }}
                .target-achievement {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .target-card {{
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #27ae60;
                }}
                .target-rate {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #27ae60;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 2px solid #ecf0f1;
                    color: #7f8c8d;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏆 XAU1 Strategy Optimization Report</h1>
                    <div class="timestamp">Generated: {timestamp}</div>
                </div>
                
                <div class="executive-summary">
                    <h2>📊 Executive Summary</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{sections['executive_summary']['total_configurations_tested']}</div>
                            <div class="metric-label">Configurations Tested</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{sections['executive_summary']['best_score']:.1f}/10</div>
                            <div class="metric-label">Best Score</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{sections['executive_summary']['target_achievements']['all_targets']:.1f}%</div>
                            <div class="metric-label">All Targets Achieved</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{sections['executive_summary']['best_metrics'].get('trades_per_week', 0):.1f}</div>
                            <div class="metric-label">Trades/Week (Best)</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎯 Top 10 Configurations</h2>
                    <p>Ranked by optimization score. Green rows meet all target criteria.</p>
                    {sections['top_configurations']}
                </div>
                
                <div class="section">
                    <h2>📈 Performance Analysis</h2>
                    <div class="chart-container">
                        <h3>Trades per Week vs Win Rate</h3>
                        {sections['visualizations']['trades_vs_winrate']}
                    </div>
                    <div class="chart-container">
                        <h3>Profit Factor vs Max Drawdown</h3>
                        {sections['visualizations']['profit_factor_vs_drawdown']}
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔍 Parameter Analysis</h2>
                    <div class="chart-container">
                        <h3>Parameter Correlation Heatmap</h3>
                        {sections['visualizations']['parameter_heatmap']}
                    </div>
                    <div class="chart-container">
                        <h3>Score Distribution</h3>
                        {sections['visualizations']['score_distribution']}
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎯 Target Achievement Analysis</h2>
                    <div class="target-achievement">
        """
        
        # Add target achievement cards
        for target_name, target_data in sections['target_analysis']['individual_targets'].items():
            html_template += f"""
                        <div class="target-card">
                            <h4>{target_data['description']}</h4>
                            <div class="target-rate">{target_data['achieved']}/{sections['executive_summary']['total_configurations_tested']}</div>
                            <div>Success Rate: {target_data['rate']:.1f}%</div>
                        </div>
            """
        
        html_template += f"""
                    </div>
                    <p><strong>Overall Success Rate:</strong> {sections['target_analysis']['overall_achievement']['overall_rate']:.1f}% of configurations meet all targets</p>
                </div>
                
                <div class="recommendations">
                    <h3>🎯 Strategic Recommendations</h3>
        """
        
        # Add recommendations
        for recommendation in sections['recommendations']:
            html_template += f"""
                    <div class="recommendation-item">{recommendation}</div>
            """
        
        html_template += f"""
                </div>
                
                <div class="footer">
                    <p>XAU1 Strategy Optimization Report | Generated by XAU1 Parameter Optimizer</p>
                    <p>Best Configuration: {sections['executive_summary']['best_config']}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template


def main():
    """Main function to generate optimization report"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load optimization results
    try:
        with open('reports/optimization_results.json', 'r') as f:
            optimization_results = json.load(f)
    except FileNotFoundError:
        logger.error("Optimization results not found. Run parameter_search.py first.")
        return
    
    # Generate report
    generator = OptimizationReportGenerator()
    report_path = generator.generate_report(optimization_results)
    
    print(f"✅ Optimization report generated: {report_path}")


if __name__ == "__main__":
    main()