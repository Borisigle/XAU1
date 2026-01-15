"""
XAU1 Parameter Optimization
Grid search for optimal parameters to achieve exactly 3 trades/week
"""

import itertools
import json
import logging
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from xau1.backtest.backtester import XAU1Backtester
from xau1.utils.data import load_market_data

logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """Parameter optimization engine"""
    
    def __init__(self, data_path: str = "data/"):
        self.data_path = data_path
        self.results = []
        
        # Optimization parameter ranges
        self.parameter_ranges = {
            'min_factors': [2, 3, 4, 5],
            'min_rr_ratio': [1.8, 1.9, 2.0, 2.1, 2.2],
            'stop_loss_pips': [25, 28, 30, 32, 35],
            'tp2_pips': [95, 100, 105, 110],
            'min_win_rate_filter': [0.48, 0.50, 0.52, 0.54]
        }
        
        # Target metrics
        self.target_trades_per_week = 3.0
        self.target_win_rate = 0.56
        self.target_profit_factor = 2.2
        self.target_max_drawdown = 10.0
        self.target_sharpe = 1.4
        
    def run_optimization(self, months: int = 12, save_results: bool = True) -> List[Dict]:
        """
        Run comprehensive parameter optimization
        
        Args:
            months: Number of months of historical data to use
            save_results: Whether to save results to CSV
            
        Returns:
            List of optimization results ranked by score
        """
        logger.info(f"Starting parameter optimization with {months} months of data")
        
        # Load historical data
        data_path = os.path.join(self.data_path, "xauusdt_15m.csv")
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            return []
        
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Filter data for optimization period
        end_date = df.index.max()
        start_date = end_date - timedelta(days=months*30)
        df = df[df.index >= start_date]
        
        logger.info(f"Using {len(df)} bars from {df.index.min()} to {df.index.max()}")
        
        # Generate all parameter combinations
        param_combinations = list(itertools.product(
            self.parameter_ranges['min_factors'],
            self.parameter_ranges['min_rr_ratio'],
            self.parameter_ranges['stop_loss_pips'],
            self.parameter_ranges['tp2_pips'],
            self.parameter_ranges['min_win_rate_filter']
        ))
        
        total_combinations = len(param_combinations)
        logger.info(f"Testing {total_combinations} parameter combinations")
        
        # Run optimization in parallel
        self.results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for i, (min_factors, min_rr, sl_pips, tp2_pips, min_wr) in enumerate(param_combinations):
                future = executor.submit(
                    self._evaluate_parameters,
                    df,
                    {
                        'min_factors': min_factors,
                        'min_rr_ratio': min_rr,
                        'stop_loss_pips': sl_pips,
                        'take_profit2_pips': tp2_pips,
                        'min_win_rate_filter': min_wr
                    }
                )
                futures.append((future, (min_factors, min_rr, sl_pips, tp2_pips, min_wr)))
            
            for i, (future, params) in enumerate(futures):
                try:
                    result = future.result()
                    if result:
                        self.results.append(result)
                    
                    if (i + 1) % 50 == 0:
                        logger.info(f"Completed {i + 1}/{total_combinations} combinations")
                        
                except Exception as e:
                    logger.error(f"Error evaluating params {params}: {e}")
        
        # Sort by score
        self.results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Optimization completed. Found {len(self.results)} valid configurations")
        
        # Save results
        if save_results and self.results:
            self._save_results()
            
        return self.results
    
    def _evaluate_parameters(self, df: pd.DataFrame, params: Dict) -> Optional[Dict]:
        """Evaluate a specific parameter combination"""
        try:
            # Create strategy config
            strategy_config = self._create_strategy_config(params)
            
            # Create backtest config
            backtest_config = {
                'simulation': {
                    'initial_capital': 10000,
                    'commission_rate': 0.0002,
                    'slippage_pips': 1.0,
                    'spread_pips': 0.5
                }
            }
            
            # Run backtest
            backtester = XAU1Backtester(strategy_config, backtest_config)
            result = backtester.run_backtest(df)
            
            # Calculate trades per week
            start_date = result.start_date
            end_date = result.end_date
            weeks = (end_date - start_date).days / 7
            trades_per_week = result.total_trades / weeks if weeks > 0 else 0
            
            # Calculate score
            score = self._calculate_score(result, trades_per_week)
            
            return {
                'params': params,
                'metrics': {
                    'trades_per_week': trades_per_week,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio,
                    'total_trades': result.total_trades,
                    'total_return_pct': result.total_return_pct
                },
                'score': score,
                'meets_targets': self._meets_targets(result, trades_per_week)
            }
            
        except Exception as e:
            logger.error(f"Error in parameter evaluation: {e}")
            return None
    
    def _create_strategy_config(self, params: Dict) -> Dict:
        """Create strategy configuration with given parameters"""
        return {
            'strategy': {
                'name': 'XAU1 SMC + Order Flow',
                'symbol': 'XAUUSDT',
                'exchange': 'binance'
            },
            'timeframes': {
                'bias': '4h',
                'confirm': '1h', 
                'entry': '15m'
            },
            'smc_indicators': {
                'swing_points': {'left_bars': 2, 'right_bars': 2},
                'fair_value_gaps': {'merge_consecutive': True},
                'order_blocks': {'lookback_period': 50, 'volume_threshold': 1.5},
                'market_structure': {'bos_lookback': 20, 'choch_lookback': 30},
                'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
                'atr': {'period': 14, 'multiplier': 1.5}
            },
            'trading_sessions': {
                'london': {'start': '13:00', 'end': '20:00', 'timezone': 'UTC'},
                'newyork': {'start': '13:30', 'end': '21:00', 'timezone': 'UTC'}
            },
            'entry_rules': {
                'type1_bos_fvg_rsi': {
                    'enabled': True,
                    'min_confluence': params['min_factors']
                },
                'type2_ob_liquidity': {
                    'enabled': True,
                    'min_confluence': max(params['min_factors'] - 1, 2)
                },
                'type3_rsi_divergence': {
                    'enabled': True,
                    'min_confluence': max(params['min_factors'] - 2, 2)
                }
            },
            'risk_management': {
                'position_size_percentage': 1.0,
                'stop_loss_pips': params['stop_loss_pips'],
                'take_profit1_pips': 50,
                'take_profit2_pips': params['tp2_pips'],
                'min_risk_reward_ratio': params['min_rr_ratio'],
                'max_positions': 2,
                'min_win_rate_filter': params['min_win_rate_filter']
            },
            'filters': {
                'max_trades_per_session': 3,
                'skip_friday_after': '18:00'
            }
        }
    
    def _calculate_score(self, result, trades_per_week: float) -> float:
        """Calculate optimization score (0-10)"""
        score = 0.0
        
        # Trades per week score (target: 3.0)
        if 2.5 <= trades_per_week <= 3.5:
            score += 2.5  # Perfect range
        elif 2.0 <= trades_per_week <= 4.0:
            score += 2.0  # Good range
        elif 1.5 <= trades_per_week <= 4.5:
            score += 1.5  # Acceptable range
        else:
            score += max(0, 2.5 - abs(trades_per_week - 3.0) * 0.5)
        
        # Win rate score (target: >= 56%)
        if result.win_rate >= 0.60:
            score += 2.5
        elif result.win_rate >= 0.58:
            score += 2.0
        elif result.win_rate >= 0.56:
            score += 1.5
        elif result.win_rate >= 0.54:
            score += 1.0
        else:
            score += max(0, result.win_rate * 1.5)
        
        # Profit factor score (target: >= 2.2)
        if result.profit_factor >= 2.5:
            score += 2.5
        elif result.profit_factor >= 2.2:
            score += 2.0
        elif result.profit_factor >= 2.0:
            score += 1.5
        elif result.profit_factor >= 1.8:
            score += 1.0
        else:
            score += max(0, result.profit_factor * 0.8)
        
        # Max drawdown score (target: <= 10%)
        if result.max_drawdown <= 8.0:
            score += 2.0
        elif result.max_drawdown <= 10.0:
            score += 1.5
        elif result.max_drawdown <= 12.0:
            score += 1.0
        else:
            score += max(0, 2.0 - (result.max_drawdown - 8.0) * 0.1)
        
        # Sharpe ratio score (target: >= 1.4)
        if result.sharpe_ratio >= 1.6:
            score += 1.0
        elif result.sharpe_ratio >= 1.4:
            score += 0.8
        elif result.sharpe_ratio >= 1.2:
            score += 0.6
        elif result.sharpe_ratio >= 1.0:
            score += 0.4
        else:
            score += max(0, result.sharpe_ratio * 0.3)
        
        return min(score, 10.0)  # Cap at 10.0
    
    def _meets_targets(self, result, trades_per_week: float) -> bool:
        """Check if results meet all target criteria"""
        return (
            2.5 <= trades_per_week <= 3.5 and
            result.win_rate >= self.target_win_rate and
            result.profit_factor >= self.target_profit_factor and
            result.max_drawdown <= self.target_max_drawdown and
            result.sharpe_ratio >= self.target_sharpe
        )
    
    def _save_results(self):
        """Save optimization results to files"""
        os.makedirs('reports', exist_ok=True)
        
        # Save CSV with detailed results
        results_data = []
        for result in self.results:
            row = {
                'min_factors': result['params']['min_factors'],
                'min_rr_ratio': result['params']['min_rr_ratio'],
                'stop_loss_pips': result['params']['stop_loss_pips'],
                'tp2_pips': result['params']['tp2_pips'],
                'min_win_rate_filter': result['params']['min_win_rate_filter'],
                'trades_per_week': result['metrics']['trades_per_week'],
                'win_rate': result['metrics']['win_rate'],
                'profit_factor': result['metrics']['profit_factor'],
                'max_drawdown': result['metrics']['max_drawdown'],
                'sharpe_ratio': result['metrics']['sharpe_ratio'],
                'total_trades': result['metrics']['total_trades'],
                'total_return_pct': result['metrics']['total_return_pct'],
                'score': result['score'],
                'meets_targets': result['meets_targets']
            }
            results_data.append(row)
        
        df_results = pd.DataFrame(results_data)
        df_results.to_csv('reports/optimization_results.csv', index=False)
        
        # Save JSON for detailed analysis
        with open('reports/optimization_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info("Results saved to reports/optimization_results.csv and .json")
    
    def get_top_configurations(self, top_n: int = 10) -> List[Dict]:
        """Get top N configurations"""
        return self.results[:top_n] if self.results else []
    
    def print_summary(self):
        """Print optimization summary"""
        if not self.results:
            logger.warning("No results to display")
            return
        
        print("\n" + "="*80)
        print("XAU1 PARAMETER OPTIMIZATION RESULTS")
        print("="*80)
        
        # Overall stats
        total_configs = len(self.results)
        configs_meeting_targets = sum(1 for r in self.results if r['meets_targets'])
        
        print(f"\nTotal configurations tested: {total_configs}")
        print(f"Configurations meeting targets: {configs_meeting_targets}")
        print(f"Success rate: {configs_meeting_targets/total_configs*100:.1f}%")
        
        # Top 5 configurations
        print("\nTOP 5 CONFIGURATIONS:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Score':<6} {'Factors':<8} {'RR':<5} {'SL':<5} {'TP2':<5} {'T/Wk':<6} {'WR%':<6} {'PF':<6} {'DD%':<6} {'Targets':<8}")
        print("-" * 80)
        
        for i, result in enumerate(self.results[:5], 1):
            params = result['params']
            metrics = result['metrics']
            
            target_icon = "✅" if result['meets_targets'] else "❌"
            
            print(f"{i:<4} {result['score']:<6.2f} {params['min_factors']:<8} "
                  f"{params['min_rr_ratio']:<5.1f} {params['stop_loss_pips']:<5} "
                  f"{params['tp2_pips']:<5} {metrics['trades_per_week']:<6.1f} "
                  f"{metrics['win_rate']*100:<6.1f} {metrics['profit_factor']:<6.2f} "
                  f"{metrics['max_drawdown']:<6.1f} {target_icon:<8}")
        
        # Target metrics summary
        print("\nTARGET METRICS ACHIEVED:")
        print("-" * 40)
        target_stats = {
            'Trades/Week (2.5-3.5)': sum(1 for r in self.results if 2.5 <= r['metrics']['trades_per_week'] <= 3.5),
            'Win Rate ≥56%': sum(1 for r in self.results if r['metrics']['win_rate'] >= 0.56),
            'Profit Factor ≥2.2': sum(1 for r in self.results if r['metrics']['profit_factor'] >= 2.2),
            'Max DD ≤10%': sum(1 for r in self.results if r['metrics']['max_drawdown'] <= 10.0),
            'Sharpe ≥1.4': sum(1 for r in self.results if r['metrics']['sharpe_ratio'] >= 1.4)
        }
        
        for metric, count in target_stats.items():
            print(f"{metric:<25}: {count:>3}/{total_configs} ({count/total_configs*100:>5.1f}%)")


def main():
    """Main optimization function"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    optimizer = ParameterOptimizer()
    
    # Run optimization
    results = optimizer.run_optimization(months=12, save_results=True)
    
    # Print summary
    optimizer.print_summary()
    
    # Save optimal config
    if results:
        best_config = results[0]
        with open('src/xau1/config/optimal_params.json', 'w') as f:
            json.dump(best_config, f, indent=2, default=str)
        
        print(f"\n✅ Best configuration saved to src/xau1/config/optimal_params.json")
        print(f"Score: {best_config['score']:.2f}/10.0")


if __name__ == "__main__":
    main()