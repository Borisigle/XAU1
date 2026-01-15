"""
XAU1 Robust Validation System
Walkforward validation and Monte Carlo simulation
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from xau1.backtest.backtester import XAU1Backtester

logger = logging.getLogger(__name__)


class RobustValidator:
    """Robust validation using walkforward and Monte Carlo methods"""
    
    def __init__(self, data_path: str = "data/"):
        self.data_path = data_path
        
    def walkforward_validation(self, 
                            strategy_config: Dict, 
                            total_months: int = 12, 
                            train_months: int = 9, 
                            test_months: int = 3,
                            min_train_trades: int = 30) -> Dict:
        """
        Perform walkforward validation
        
        Args:
            strategy_config: Strategy configuration to validate
            total_months: Total months of data to use
            train_months: Training period length
            test_months: Testing period length
            min_train_trades: Minimum trades required in training period
            
        Returns:
            Walkforward validation results
        """
        logger.info("Starting walkforward validation...")
        
        # Load data
        data_path = f"{self.data_path}/xauusdt_15m.csv"
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Setup periods
        end_date = df.index.max()
        start_date = end_date - timedelta(days=total_months*30)
        df = df[df.index >= start_date].copy()
        
        logger.info(f"Using data from {df.index.min()} to {df.index.max()}")
        
        # Run walkforward validation
        fold_results = []
        current_start = df.index.min()
        
        fold = 1
        while True:
            # Define train and test periods
            train_end = current_start + timedelta(days=train_months*30)
            test_start = train_end
            test_end = test_start + timedelta(days=test_months*30)
            
            # Check if we have enough data
            if test_end > df.index.max():
                break
            
            # Split data
            train_df = df[(df.index >= current_start) & (df.index < train_end)]
            test_df = df[(df.index >= test_start) & (df.index < test_end)]
            
            logger.info(f"Fold {fold}: Train {train_df.index.min()} to {train_df.index.max()} "
                       f"({len(train_df)} bars), Test {test_df.index.min()} to {test_df.index.max()} "
                       f"({len(test_df)} bars)")
            
            # Train model (backtest on training data)
            train_result = self._run_single_backtest(strategy_config, train_df)
            
            # Test model (backtest on test data)
            test_result = self._run_single_backtest(strategy_config, test_df)
            
            # Store results if we have enough trades
            if train_result and train_result.get('total_trades', 0) >= min_train_trades:
                fold_result = {
                    'fold': fold,
                    'train_start': train_df.index.min(),
                    'train_end': train_df.index.max(),
                    'test_start': test_df.index.min(),
                    'test_end': test_df.index.max(),
                    'train_result': train_result,
                    'test_result': test_result,
                    'overfitting_risk': self._calculate_overfitting_risk(train_result, test_result)
                }
                fold_results.append(fold_result)
                logger.info(f"Fold {fold} completed: Train WR={train_result['win_rate']:.3f}, "
                           f"Test WR={test_result['win_rate']:.3f}")
            else:
                logger.warning(f"Fold {fold} skipped: insufficient training trades "
                             f"({train_result.get('total_trades', 0)})")
            
            # Move to next period
            current_start = test_start
            fold += 1
            
            # Safety check to avoid infinite loops
            if fold > 10:
                break
        
        if not fold_results:
            logger.error("No valid fold results found")
            return {}
        
        # Aggregate results
        aggregated = self._aggregate_walkforward_results(fold_results)
        
        logger.info("Walkforward validation completed")
        return {
            'method': 'walkforward',
            'config': strategy_config,
            'fold_results': fold_results,
            'aggregated_results': aggregated,
            'validation_passed': self._validate_walkforward_results(aggregated)
        }
    
    def monte_carlo_simulation(self, 
                             trade_results: List[Dict], 
                             n_simulations: int = 1000,
                             random_seed: int = 42) -> Dict:
        """
        Perform Monte Carlo simulation on trade results
        
        Args:
            trade_results: List of trade dictionaries with 'pnl' values
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
            
        Returns:
            Monte Carlo simulation results
        """
        logger.info(f"Starting Monte Carlo simulation with {n_simulations} simulations...")
        
        # Extract PnL values
        pnls = [trade['pnl'] for trade in trade_results if trade.get('pnl') is not None]
        
        if len(pnls) < 10:
            logger.warning("Insufficient trade results for Monte Carlo simulation")
            return {}
        
        np.random.seed(random_seed)
        
        # Run simulations
        max_drawdowns = []
        final_returns = []
        win_rates = []
        sharpe_ratios = []
        
        n_trades = len(pnls)
        
        for i in range(n_simulations):
            # Randomly shuffle trades
            shuffled_pnls = np.random.choice(pnls, size=n_trades, replace=True)
            
            # Calculate cumulative returns
            cumulative_returns = np.cumsum(shuffled_pnls)
            initial_capital = 10000
            equity_curve = initial_capital + cumulative_returns
            
            # Calculate max drawdown
            running_max = np.maximum.accumulate(equity_curve)
            drawdowns = (equity_curve - running_max) / running_max
            max_drawdown = abs(np.min(drawdowns)) * 100
            
            # Calculate final return
            final_return = (equity_curve[-1] / initial_capital - 1) * 100
            
            # Calculate win rate
            win_rate = np.sum(np.array(shuffled_pnls) > 0) / n_trades
            
            # Calculate Sharpe ratio
            if np.std(shuffled_pnls) > 0:
                sharpe = np.mean(shuffled_pnls) / np.std(shuffled_pnls) * np.sqrt(252)
            else:
                sharpe = 0
            
            max_drawdowns.append(max_drawdown)
            final_returns.append(final_return)
            win_rates.append(win_rate)
            sharpe_ratios.append(sharpe)
        
        # Calculate statistics
        results = {
            'method': 'monte_carlo',
            'n_simulations': n_simulations,
            'n_trades_used': n_trades,
            'max_drawdown': {
                'mean': np.mean(max_drawdowns),
                'median': np.median(max_drawdowns),
                'std': np.std(max_drawdowns),
                'percentile_5': np.percentile(max_drawdowns, 5),
                'percentile_95': np.percentile(max_drawdowns, 95),
                'max': np.max(max_drawdowns),
                'min': np.min(max_drawdowns)
            },
            'final_return': {
                'mean': np.mean(final_returns),
                'median': np.median(final_returns),
                'std': np.std(final_returns),
                'percentile_5': np.percentile(final_returns, 5),
                'percentile_95': np.percentile(final_returns, 95),
                'max': np.max(final_returns),
                'min': np.min(final_returns)
            },
            'win_rate': {
                'mean': np.mean(win_rates),
                'median': np.median(win_rates),
                'std': np.std(win_rates),
                'percentile_5': np.percentile(win_rates, 5),
                'percentile_95': np.percentile(win_rates, 95),
                'max': np.max(win_rates),
                'min': np.min(win_rates)
            },
            'sharpe_ratio': {
                'mean': np.mean(sharpe_ratios),
                'median': np.median(sharpe_ratios),
                'std': np.std(sharpe_ratios),
                'percentile_5': np.percentile(sharpe_ratios, 5),
                'percentile_95': np.percentile(sharpe_ratios, 95),
                'max': np.max(sharpe_ratios),
                'min': np.min(sharpe_ratios)
            }
        }
        
        # Calculate probability of meeting targets
        target_drawdown = 10.0
        target_return = 20.0
        target_win_rate = 0.56
        target_sharpe = 1.4
        
        results['probabilities'] = {
            'max_drawdown_under_10pct': np.mean(np.array(max_drawdowns) <= target_drawdown),
            'final_return_over_20pct': np.mean(np.array(final_returns) >= target_return),
            'win_rate_over_56pct': np.mean(np.array(win_rates) >= target_win_rate),
            'sharpe_over_1.4': np.mean(np.array(sharpe_ratios) >= target_sharpe),
            'all_targets_met': np.mean(
                (np.array(max_drawdowns) <= target_drawdown) &
                (np.array(final_returns) >= target_return) &
                (np.array(win_rates) >= target_win_rate) &
                (np.array(sharpe_ratios) >= target_sharpe)
            )
        }
        
        logger.info("Monte Carlo simulation completed")
        return results
    
    def sensitivity_analysis(self, 
                          strategy_config: Dict, 
                          base_result: Dict,
                          df: pd.DataFrame) -> Dict:
        """
        Perform sensitivity analysis on key parameters
        
        Args:
            strategy_config: Base strategy configuration
            base_result: Base backtest result
            df: Historical data
            
        Returns:
            Sensitivity analysis results
        """
        logger.info("Starting sensitivity analysis...")
        
        # Define sensitivity ranges
        sensitivity_params = {
            'slippage_pips': [-1, -0.5, 0, 0.5, 1, 1.5, 2],  # Slippage impact
            'commission_rate': [-0.01, -0.005, 0, 0.005, 0.01, 0.015, 0.02],  # Commission impact
            'win_rate_adjustment': [-0.05, -0.03, -0.01, 0, 0.01, 0.03, 0.05],  # Win rate variance
            'spread_pips': [-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5]  # Spread impact
        }
        
        base_slippage = base_result.get('slippage_pips', 1.0)
        base_commission = base_result.get('commission_rate', 0.0002)
        base_win_rate = base_result.get('win_rate', 0.55)
        base_spread = base_result.get('spread_pips', 0.5)
        
        sensitivity_results = {}
        
        for param_name, adjustments in sensitivity_params.items():
            logger.info(f"Testing {param_name} sensitivity...")
            
            param_results = []
            
            for adjustment in adjustments:
                try:
                    # Create modified config
                    modified_config = strategy_config.copy()
                    
                    if param_name == 'slippage_pips':
                        new_slippage = base_slippage + adjustment
                        modified_config['backtest']['simulation']['slippage_pips'] = new_slippage
                        test_value = new_slippage
                    elif param_name == 'commission_rate':
                        new_commission = max(0, base_commission + adjustment)
                        modified_config['backtest']['simulation']['commission_rate'] = new_commission
                        test_value = new_commission
                    elif param_name == 'win_rate_adjustment':
                        # Simulate win rate impact by adjusting confluence requirements
                        new_wr_filter = max(0.4, min(0.6, base_win_rate + adjustment))
                        modified_config['risk_management']['min_win_rate_filter'] = new_wr_filter
                        test_value = new_wr_filter
                    elif param_name == 'spread_pips':
                        new_spread = max(0, base_spread + adjustment)
                        modified_config['backtest']['simulation']['spread_pips'] = new_spread
                        test_value = new_spread
                    
                    # Run backtest
                    result = self._run_single_backtest(modified_config, df)
                    
                    if result:
                        param_results.append({
                            'adjustment': adjustment,
                            'test_value': test_value,
                            'total_return_pct': result.get('total_return_pct', 0),
                            'max_drawdown': result.get('max_drawdown', 0),
                            'win_rate': result.get('win_rate', 0),
                            'profit_factor': result.get('profit_factor', 0),
                            'total_trades': result.get('total_trades', 0)
                        })
                    
                except Exception as e:
                    logger.error(f"Error in sensitivity test for {param_name}={adjustment}: {e}")
            
            sensitivity_results[param_name] = param_results
        
        # Calculate sensitivity metrics
        sensitivity_summary = {}
        for param_name, results in sensitivity_results.items():
            if not results:
                continue
                
            returns = [r['total_return_pct'] for r in results]
            drawdowns = [r['max_drawdown'] for r in results]
            
            sensitivity_summary[param_name] = {
                'return_sensitivity': np.std(returns),  # How much returns vary
                'drawdown_sensitivity': np.std(drawdowns),  # How much drawdown varies
                'robustness_score': 1 / (1 + np.std(returns) + np.std(drawdowns)),  # Lower is better
                'worst_case_return': min(returns),
                'best_case_return': max(returns),
                'parameter_stability': np.std(returns) / abs(np.mean(returns)) if np.mean(returns) != 0 else float('inf')
            }
        
        logger.info("Sensitivity analysis completed")
        return {
            'method': 'sensitivity',
            'base_result': base_result,
            'parameter_results': sensitivity_results,
            'sensitivity_summary': sensitivity_summary,
            'robustness_assessment': self._assess_robustness(sensitivity_summary)
        }
    
    def _run_single_backtest(self, strategy_config: Dict, df: pd.DataFrame) -> Optional[Dict]:
        """Run a single backtest and return results"""
        try:
            # Create backtest config
            backtest_config = strategy_config.get('backtest', {})
            if not backtest_config:
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
            
            # Convert to dict
            return {
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'total_return_pct': result.total_return_pct,
                'initial_capital': result.initial_capital,
                'final_capital': result.final_capital,
                'start_date': result.start_date,
                'end_date': result.end_date,
                'slippage_pips': backtest_config['simulation']['slippage_pips'],
                'commission_rate': backtest_config['simulation']['commission_rate'],
                'spread_pips': backtest_config['simulation']['spread_pips']
            }
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return None
    
    def _calculate_overfitting_risk(self, train_result: Dict, test_result: Dict) -> float:
        """Calculate overfitting risk score (0-1, higher is worse)"""
        if not train_result or not test_result:
            return 1.0
        
        # Compare key metrics
        train_wr = train_result.get('win_rate', 0)
        test_wr = test_result.get('win_rate', 0)
        
        train_pf = train_result.get('profit_factor', 0)
        test_pf = test_result.get('profit_factor', 0)
        
        train_dd = train_result.get('max_drawdown', 0)
        test_dd = test_result.get('max_drawdown', 0)
        
        # Calculate performance degradation
        wr_degradation = max(0, train_wr - test_wr)
        pf_degradation = max(0, train_pf - test_pf) / max(train_pf, 1)
        dd_degradation = max(0, test_dd - train_dd) / max(train_dd, 1)
        
        # Overfitting risk score
        overfitting_risk = (wr_degradation * 0.4 + pf_degradation * 0.3 + dd_degradation * 0.3)
        
        return min(overfitting_risk, 1.0)
    
    def _aggregate_walkforward_results(self, fold_results: List[Dict]) -> Dict:
        """Aggregate walkforward validation results"""
        if not fold_results:
            return {}
        
        # Extract metrics
        train_wrs = [r['train_result']['win_rate'] for r in fold_results]
        test_wrs = [r['test_result']['win_rate'] for r in fold_results]
        train_pfs = [r['train_result']['profit_factor'] for r in fold_results]
        test_pfs = [r['test_result']['profit_factor'] for r in fold_results]
        train_dds = [r['train_result']['max_drawdown'] for r in fold_results]
        test_dds = [r['test_result']['max_drawdown'] for r in fold_results]
        overfitting_risks = [r['overfitting_risk'] for r in fold_results]
        
        # Calculate averages and consistency
        aggregated = {
            'n_folds': len(fold_results),
            'train_win_rate': {
                'mean': np.mean(train_wrs),
                'std': np.std(train_wrs),
                'min': np.min(train_wrs),
                'max': np.max(train_wrs)
            },
            'test_win_rate': {
                'mean': np.mean(test_wrs),
                'std': np.std(test_wrs),
                'min': np.min(test_wrs),
                'max': np.max(test_wrs)
            },
            'train_profit_factor': {
                'mean': np.mean(train_pfs),
                'std': np.std(train_pfs),
                'min': np.min(train_pfs),
                'max': np.max(train_pfs)
            },
            'test_profit_factor': {
                'mean': np.mean(test_pfs),
                'std': np.std(test_pfs),
                'min': np.min(test_pfs),
                'max': np.max(test_pfs)
            },
            'train_max_drawdown': {
                'mean': np.mean(train_dds),
                'std': np.std(train_dds),
                'min': np.min(train_dds),
                'max': np.max(train_dds)
            },
            'test_max_drawdown': {
                'mean': np.mean(test_dds),
                'std': np.std(test_dds),
                'min': np.min(test_dds),
                'max': np.max(test_dds)
            },
            'overfitting_risk': {
                'mean': np.mean(overfitting_risks),
                'max': np.max(overfitting_risks),
                'consistency_score': 1 - np.std(overfitting_risks)
            }
        }
        
        return aggregated
    
    def _validate_walkforward_results(self, aggregated: Dict) -> bool:
        """Validate if walkforward results are acceptable"""
        if not aggregated:
            return False
        
        # Check consistency
        test_wr_std = aggregated['test_win_rate']['std']
        test_pf_std = aggregated['test_profit_factor']['std']
        overfitting_mean = aggregated['overfitting_risk']['mean']
        
        # Validation criteria
        criteria = {
            'test_wr_std_below_0.15': test_wr_std < 0.15,
            'test_pf_std_below_0.5': test_pf_std < 0.5,
            'overfitting_risk_below_0.3': overfitting_mean < 0.3,
            'avg_test_wr_above_0.50': aggregated['test_win_rate']['mean'] > 0.50,
            'avg_test_pf_above_1.8': aggregated['test_profit_factor']['mean'] > 1.8,
            'avg_test_dd_below_15': aggregated['test_max_drawdown']['mean'] < 15.0
        }
        
        passed_criteria = sum(criteria.values())
        total_criteria = len(criteria)
        
        logger.info(f"Walkforward validation: {passed_criteria}/{total_criteria} criteria passed")
        for criterion, passed in criteria.items():
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {criterion}")
        
        return passed_criteria >= total_criteria * 0.7  # 70% of criteria must pass
    
    def _assess_robustness(self, sensitivity_summary: Dict) -> Dict:
        """Assess overall robustness based on sensitivity analysis"""
        robustness_scores = {}
        
        for param_name, metrics in sensitivity_summary.items():
            # Lower sensitivity = higher robustness
            return_stability = 1 / (1 + metrics['return_sensitivity'])
            drawdown_stability = 1 / (1 + metrics['drawdown_sensitivity'])
            overall_robustness = (return_stability + drawdown_stability) / 2
            
            robustness_scores[param_name] = {
                'return_stability': return_stability,
                'drawdown_stability': drawdown_stability,
                'overall_robustness': overall_robustness,
                'is_robust': overall_robustness > 0.7
            }
        
        # Overall robustness
        overall_robustness = np.mean([s['overall_robustness'] for s in robustness_scores.values()])
        
        return {
            'parameter_robustness': robustness_scores,
            'overall_robustness': overall_robustness,
            'is_robust_overall': overall_robustness > 0.7,
            'recommendation': self._get_robustness_recommendation(overall_robustness, robustness_scores)
        }
    
    def _get_robustness_recommendation(self, overall_robustness: float, param_scores: Dict) -> str:
        """Get recommendation based on robustness assessment"""
        if overall_robustness >= 0.8:
            return "HIGHLY ROBUST - Strategy is very stable across different market conditions"
        elif overall_robustness >= 0.7:
            return "ROBUST - Strategy shows good stability with minor sensitivity to some parameters"
        elif overall_robustness >= 0.6:
            return "MODERATELY ROBUST - Strategy has some sensitivity but acceptable overall"
        elif overall_robustness >= 0.5:
            return "SENSITIVE - Strategy shows notable sensitivity to parameter changes"
        else:
            return "HIGHLY SENSITIVE - Strategy may not be suitable for live trading"


def main():
    """Main validation function"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load optimal configuration
    import json
    with open('src/xau1/config/optimal_params.json', 'r') as f:
        optimal_config = json.load(f)
    
    validator = RobustValidator()
    
    # Run walkforward validation
    wf_results = validator.walkforward_validation(optimal_config['params'])
    
    # Load trade results for Monte Carlo
    trades_df = pd.read_csv('reports/backtest_results.csv')  # Assuming this exists
    trade_results = trades_df.to_dict('records')
    
    mc_results = validator.monte_carlo_simulation(trade_results)
    
    # Run sensitivity analysis
    import pandas as pd
    df = pd.read_csv('data/xauusdt_15m.csv', index_col=0, parse_dates=True)
    sens_results = validator.sensitivity_analysis(optimal_config['params'], optimal_config['metrics'], df)
    
    # Save validation report
    validation_report = {
        'walkforward': wf_results,
        'monte_carlo': mc_results,
        'sensitivity': sens_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('reports/validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    
    print("\n✅ Robust validation completed!")
    print("Results saved to reports/validation_report.json")


if __name__ == "__main__":
    main()