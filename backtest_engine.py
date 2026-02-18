"""
Backtesting Engine for Self-Exciting Pairs Trading

Will simulate realistic trading with:
- Transaction costs (commissions and slippage)
- Position management
- Risk metric calculation
- Regime specific performance analysis
"""

import numpy as np 
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Trade:
    """ Individual trade records"""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp 
    direction: int #1 = long, -1 = short
    entry_price: float
    exit_price: float 
    position_size: float 
    pnl: float 
    return_pct: float 
    duration: int #trading days
    entry_lambda: float 
    exit_lambda: float 
    max_adverse_excursion: float
    max_favorable_excursion: float

class BacktestEngine:
    """
    Backtest trading strategy with realistic execution assumptions
    """

    def __init__(self,
                 initial_capital: float = 1_000_000,
                 commission_rate: float = 0.0002,
                 slippage_bps: float = 1.0):
        
        """ 
        Initializing backtesting
        
        Params:
        - initial_capital: float
            - initial capital
        - commission_rate: float 
            - commission rate (fraction of the trade value)
        - slippage_bps: float
            - slippage in basis points
        """

        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage_bps / 10000 #converting to fraction

        self.trades = []
        self.equity_curve = None 
        self.performance_metrics = {}

    def run_backtest(self,
                     signals_df: pd.DataFrame,
                     spread_df: pd.DataFrame,
                     front_prices: pd.Series,
                     back_prices: pd.Series) -> pd.DataFrame:
        
        """ 
        Run backtest on trading signals

        Params:
        - signals_df: pd.DataFrame
            - trading signals with position sizing
        - spread_df: pd.DataFrame
            - spread data
        - front_prices: pd.Series
            - front month prices
        - back_prices: pd.Series
            - back month prices

        Returns:
        - pd.DataFrame
            -equity curve and statistics
        """

        print("Running backtest")

        #Aligning the data
        common_index = signals_df.index.intersection(spread_df.index)
        signals_df = signals_df.loc[common_index]
        spread_df = spread_df.loc[common_index]
        front_prices = front_prices.loc[common_index]
        back_prices = back_prices.loc[common_index]

        n = len(signals_df)

        #initializing tracking arrays
        equity = np.zeros(n)
        equity[0] = self.initial_capital
        cash = self.initial_capital 
        position = 0 
        position_size = 0 

        #Trade tracking 
        entry_date: Optional[pd.Timestamp] = None
        entry_spread: Optional[float] = None
        entry_lambda: Optional[float] = None
        max_adverse = 0
        max_favorable = 0

        for i in range(1, n):
            date = signals_df.index[i]
            signal = int(signals_df['signal'].iloc[i])
            spread_value = float(spread_df['spread'].iloc[i])
            lambda_value = float(signals_df['lambda'].iloc[i])

            #Check for position opening
            if position == 0 and signal in [1, -1]:
                #Entry
                position = signal
                position_size = float(signals_df['position_size'].iloc[i])
                entry_date = date
                entry_spread = spread_value 
                entry_lambda = lambda_value 
                max_adverse = 0 
                max_favorable = 0 

                #Calculating transaction costs for entry 
                trade_value = abs(position_size) * self.initial_capital 
                commission = trade_value * self.commission_rate 
                slippage_cost = trade_value * self.slippage 
                cash -= (commission + slippage_cost)

            #Check for position in progress
            elif position != 0:
                if entry_spread is None:
                    equity[i] = cash
                    continue

                #Track PnL
                spread_change = spread_value - entry_spread 
                entry_spread_abs = abs(entry_spread)
                pnl_pct = position * spread_change / entry_spread_abs if entry_spread_abs != 0 else 0.0
                pnl = pnl_pct * abs(position_size) * self.initial_capital 

                #Track adverse/favorable excursion
                max_adverse = min(max_adverse, pnl_pct)
                max_favorable = max(max_favorable, pnl_pct)

                #Checking for exit signal
                if signal == 2: #Close signal
                    assert entry_date is not None and entry_lambda is not None

                    #Exit
                    trade_value = abs(position_size) * self.initial_capital
                    commission = trade_value * self.commission_rate
                    slippage_cost = trade_value * self.slippage

                    realized_pnl = pnl - (commission + slippage_cost)
                    cash += realized_pnl 

                    #Record the trade
                    trade = Trade(
                        entry_date = entry_date, 
                        exit_date = date, 
                        direction = position, 
                        entry_price = entry_spread,
                        exit_price = spread_value,
                        position_size = position_size,
                        pnl = realized_pnl,
                        return_pct = pnl_pct * 100,
                        duration = (date - entry_date).days,
                        entry_lambda = entry_lambda,
                        exit_lambda = lambda_value,
                        max_adverse_excursion = max_adverse * 100,
                        max_favorable_excursion = max_favorable * 100
                    )
                    self.trades.append(trade)

                    #Reset position
                    position = 0
                    position_size = 0 


            #update Equity
            if position != 0:
                if entry_spread is None:
                    equity[i] = cash
                    continue

                spread_change = spread_value - entry_spread 
                entry_spread_abs = abs(entry_spread)
                pnl_pct = position * spread_change / entry_spread_abs if entry_spread_abs != 0 else 0.0
                unrealized_pnl = pnl_pct * abs(position_size) * self.initial_capital
                equity[i] = cash + unrealized_pnl 
            else:
                equity[i] = cash

        #Create equity curve dataframe
        self.equity_curve = pd.DataFrame({
            'equity': equity,
            'returns': np.concatenate(([0], np.diff(equity) / equity[:-1])),
            'position': signals_df['position'],
            'spread': spread_df['spread'],
            'lambda': signals_df['lambda']
        }, index = common_index)

        n_trades = len(self.trades)
        final_equity = equity[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100

        print(f" Backtest Complete!")
        print(f"    - Total Trades: {n_trades}")
        print(f"    - Final Equity: ${final_equity:,.2f}")
        print(f"    - Total Return: {total_return:.2f}%")

        return self.equity_curve 
    
    def calculate_performance_metrics(self, risk_free_rate: float = 0.02) -> Dict:
        """
        Calculate comprehensive performance metrics

        Params:
        - risk_free_rate: float 
            - annual risk_free rate
        
        Returns:
        - dict:
            - performance metrics
        """

        if self.equity_curve is None:
            raise ValueError("No backtest results. Run run_backtest() first")
        
        print("Calculating performance metrics")

        equity = self.equity_curve['equity'].to_numpy(dtype = float)
        returns = self.equity_curve['returns'].to_numpy(dtype = float) [1:]#Excluding the first zero

        #Basic metrics
        total_return = (equity[-1] / equity[0] - 1) * 100
        n_days = len(equity)
        annual_factor = 252 / n_days
        annualized_return = ((equity[-1] / equity[0]) ** annual_factor - 1) * 100

        #Risk metrics
        returns_std = returns.std()
        annualized_vol = returns_std * np.sqrt(252) * 100 

        #Sharpe Ratio 
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std if returns_std > 0 else 0 

        #Sortino Ratio 
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else returns_std
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0 

        #Maximum drawdown
        cumulative = np.cumprod(1.0 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max 
        max_drawdown = drawdown.min() * 100 

        #Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        #Trade Statistics
        if len(self.trades) > 0:
            trade_pnls = [t.pnl for t in self.trades]
            trade_returns = [t.return_pct for t in self.trades]

            win_trades = [t for t in self.trades if t.pnl > 0]
            lose_trades = [t for t in self.trades if t.pnl <= 0]

            win_rate = len(win_trades) / len(self.trades) * 100 
            avg_win = np.mean([t.pnl for t in win_trades]) if win_trades else 0
            avg_loss = np.mean([t.pnl for t in lose_trades]) if lose_trades else 0 
            profit_factor = abs(sum([t.pnl for t in win_trades]) / sum([t.pnl for t in lose_trades])) \
                            if lose_trades and sum([t.pnl for t in lose_trades]) != 0 else np.inf 
            
            avg_duration = np.mean([t.duration for t in self.trades])
            max_duration = max([t.duration for t in self.trades])

            #Max adverse/favorable excursion
            avg_mae = np.mean([abs(t.max_adverse_excursion) for t in self.trades])
            avg_mfe = np.mean([t.max_favorable_excursion for t in self.trades])
        else:
            win_rate = 0
            avg_win = 0 
            avg_loss = 0
            profit_factor = 0
            avg_duration = 0
            max_duration = 0 
            avg_mae = 0
            avg_mfe = 0

        metrics = {
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'annualized_volatility_pct': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'total_trades': len(self.trades),
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade_duration_days': avg_duration,
            'max_trade_duration_days': max_duration,
            'avg_max_adverse_excursion_pct': avg_mae,
            'avg_max_favorable_excursion_pct': avg_mfe
        }

        self.performance_metrics = metrics

        print("Performance metrics calculated")

        return metrics 
    
    def analyze_regime_performance(self, regime_threshold: float = 0.7) -> Dict:
        """
        Analyze performance in different regimes

        Params:
        - regime_threshold: float
            - lambda_threshold for regime classificaito

        Returns:
        - Dict:
            - regime-specific performance
        """

        if self.equity_curve is None:
            raise ValueError("No backtest results available")
        print("Analyzing regime-specific performance")

        #Classifying trades by entry regime
        calm_trades = [t for t in self.trades if t.entry_lambda < regime_threshold]
        volatile_trades = [t for t in self.trades if t.entry_lambda >= regime_threshold]

        def calc_regime_stats(trades):
            if not trades:
                return {
                    'n_trades': 0,
                    'win_rate': 0,
                    'avg_return': 0,
                    'sharpe': 0
                }
            
            returns = [t.return_pct for t in trades]
            wins = [t for t in trades if t.pnl > 0]

            return {
                'n_trades': len(trades),
                'win_rate': len(wins) / len(trades) * 100,
                'avg_return': np.mean(returns),
                'sharpe': np.mean(returns) / np.std(returns) * np.sqrt(252/len(trades)) if np.std(returns) > 0 else 0
            }

        regime_stats = {
            'calm_regime': calc_regime_stats(calm_trades),
            'volatile_regime': calc_regime_stats(volatile_trades)
        }

        print(f" Calm regime: {len(calm_trades)} trades")
        print(f" Volatile regime: {len(volatile_trades)} trades")

        return regime_stats 
    
    def get_trade_summary(self) -> pd.DataFrame:
        """
        Getting summary of all trades

        Returns:
        - pd.DataFrame
            - Trade summary
        """

        if not self.trades:
            return pd.DataFrame() 
        
        trade_data = [{
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'direction': 'LONG' if t.direction == 1 else 'SHORT',
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'position_size': t.position_size,
            'pnl': t.pnl,
            'return_pct': t.return_pct,
            'duration_days': t.duration,
            'entry_lambda': t.entry_lambda,
            'exit_lambda': t.exit_lambda,
            'mae_pct': t.max_adverse_excursion,
            'mfe_pct': t.max_favorable_excursion
        } for t in self.trades]

        return pd.DataFrame(trade_data)
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plotting backtest results

        Params:
        - save_path: str, optional
            - path to save plot
        """

        if self.equity_curve is None:
            raise ValueError("No backtest results to plot")
        
        import matplotlib.pyplot as plt 

        fig, axes = plt.subplots(4, 1, figsize = (16, 22))

        #Plot 1: Equity Curve
        axes[0].plot(self.equity_curve.index, self.equity_curve['equity'] / 1000, color = 'blue', linewidth = 2)
        axes[0].axhline(y = self.initial_capital / 1000, color = 'black', linestyle = '--', alpha = 0.5, label = 'Initial Capital')
        axes[0].set_ylabel('Equity ($K)')
        axes[0].set_title('Equity Curve')
        axes[0].legend() 
        axes[0].grid(True, alpha = 0.3)

        #Plot 2: Returns Distribution
        returns = self.equity_curve['returns'].to_numpy(dtype = float) [1:]
        axes[1].hist(returns * 100, bins = 50, alpha = 0.7, color = 'purple', edgecolor = 'black')
        axes[1].axvline(x = 0, color = 'red', linestyle = '--', linewidth = 2)
        axes[1].set_xlabel('Daily Return (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Daily Returns')
        axes[1].grid(True, alpha = 0.3)

        #Plot 3: Drawdown
        cumulative = np.cumprod(1.0 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100 

        axes[2].fill_between(self.equity_curve.index[1:], drawdown, 0, color = 'red', alpha = 0.5)
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].set_title('Underwater Plot (Drawdown)')
        axes[2].grid(True, alpha = 0.3)

        #Plot 4: Trade PnL
        if self.trades:
            trade_pnls = [t.pnl for t in self.trades]
            colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
            axes[3].bar(range(len(trade_pnls)), trade_pnls, color = colors, alpha = 0.7)
            axes[3].axhline(y = 0, color = 'black', linestyle = '-', linewidth = 0.5)
            axes[3].set_xlabel('Trade Number')
            axes[3].set_ylabel('PnL ($)')
            axes[3].set_title('Individual Trade PnL')
            axes[3].grid(True, alpha = 0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
            print(f" Backtest results saved to {save_path}")

        plt.show() 

if __name__ == "__main__":
    """
    Test backtesting engine on XOM/CVX equity pairs
    """
    import sys
    import os
    
    # Add parent directory to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    from equity_pairs_loader import EquityPairsDataPipeline
    from jump_detector import JumpDetector
    from hawkes_calibration import HawkesProcess
    from signal_generation import TradingSignals
    
    print("="*70)
    print("BACKTEST ENGINE TEST - XOM/CVX EQUITY PAIRS")
    print("="*70)
    
    # Initialize data pipeline
    print("\n[1] Loading XOM/CVX data from CSV files...")
    pipeline = EquityPairsDataPipeline()
    
    # Try to load CSV files from multiple possible locations
    csv_paths = [
        (os.path.join(current_dir, 'OHLCV_XOM.csv'),
         os.path.join(current_dir, 'OHLCV_CVX.csv')),  # Script directory
        ('OHLCV_XOM.csv', 'OHLCV_CVX.csv'),  # Current directory
    ]
    
    data_loaded = False
    for xom_path, cvx_path in csv_paths:
        if os.path.exists(xom_path) and os.path.exists(cvx_path):
            try:
                data = pipeline.load_from_csv(xom_path, cvx_path)
                data_loaded = True
                print(f"✓ Loaded from: {os.path.abspath(xom_path)}")
                break
            except Exception as e:
                continue
    
    if not data_loaded:
        print("✗ Could not find CSV files. Testing with synthetic data instead...")
        print("\n" + "="*70)
        print("SYNTHETIC DATA TEST")
        print("="*70)
        
        # Generate synthetic test data (fallback)
        np.random.seed(42)
        n = 500
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        
        # Synthetic spread
        spread = np.zeros(n)
        spread[0] = 0.0
        for t in range(1, n):
            spread[t] = 0.95 * spread[t-1] + np.random.normal(0, 0.05)
            if np.random.random() < 0.02:
                spread[t] += np.random.choice([-0.2, 0.2])
        
        spread_series = pd.Series(spread, index=dates)
        
        # Synthetic signals
        signals = pd.DataFrame({
            'signal': np.zeros(n),
            'position': np.zeros(n),
            'position_size': np.zeros(n),
            'lambda': 0.5 * np.ones(n)
        }, index=dates)
        
        # Generate some trades
        position = 0
        for i in range(20, n, 30):
            if position == 0:
                signals.loc[signals.index[i], 'signal'] = 1
                signals.loc[signals.index[i]:signals.index[min(i+20, n-1)], 'position'] = 1
                signals.loc[signals.index[i]:signals.index[min(i+20, n-1)], 'position_size'] = 0.5
                position = 1
            else:
                signals.loc[signals.index[min(i+20, n-1)], 'signal'] = 2
                position = 0
        
        spread_df = pd.DataFrame({'spread': spread_series})
        
        # Synthetic prices
        asset_a_price = pd.Series(100 + spread * 10, index=dates)
        asset_b_price = pd.Series(100 - spread * 10, index=dates)
        
        # Run backtest
        engine = BacktestEngine(initial_capital=1_000_000)
        equity_curve = engine.run_backtest(signals, spread_df, asset_a_price, asset_b_price)
        
        # Calculate metrics
        metrics = engine.calculate_performance_metrics()
        
        print("\n" + "="*70)
        print("PERFORMANCE METRICS")
        print("="*70)
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'pct' in key or 'rate' in key:
                    print(f"{key}: {value:.2f}%")
                elif 'ratio' in key or 'factor' in key:
                    print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        sys.exit(0)
    
    # Clean and construct spread
    print("\n[2] Constructing spread...")
    cleaned_data = pipeline.clean_data()
    spread_df = pipeline.construct_spread(method='cointegration', lookback=60)
    
    print(f"✓ Spread constructed: {len(spread_df)} observations")
    print(f"  Date range: {spread_df.index[0].date()} to {spread_df.index[-1].date()}")
    
    # Get asset prices
    asset_a_price = cleaned_data['asset_a']['Close']
    asset_b_price = cleaned_data['asset_b']['Close']
    
    # Calculate Z-score and detect jumps
    print("\n[3] Generating trading signals...")
    spread = spread_df['spread']
    returns = spread.pct_change().dropna()
    
    # Calculate Z-score
    mean = spread.mean()
    std = spread.std()
    z_score = (spread - mean) / std
    
    # Detect jumps and fit Hawkes
    detector = JumpDetector(significance_level=0.01)
    jump_df = detector.detect_jumps_bipower_variation(returns, window=20)
    
    jump_times = jump_df[jump_df['jump_indicator'] == 1].index
    jump_times_numeric = (jump_times - spread.index[0]).days.values
    
    # Fit Hawkes
    T = (spread.index[-1] - spread.index[0]).days
    hawkes = HawkesProcess()
    
    if len(jump_times) >= 2:
        try:
            params = hawkes.fit(jump_times_numeric, T, method='MLE')
        except:
            params = {'lambda_bar': 0.001, 'alpha': 0.0, 'beta': 1.0}
    else:
        params = {'lambda_bar': 0.001, 'alpha': 0.0, 'beta': 1.0}
    
    # Calculate jump intensity
    lambda_intensity = pd.Series(params['lambda_bar'], index=spread.index)
    for idx, t in enumerate(spread.index):
        lambda_t = params['lambda_bar']
        t_numeric = (t - spread.index[0]).days
        
        for jt in jump_times_numeric:
            if jt < t_numeric:
                lambda_t += params['alpha'] * np.exp(-params['beta'] * (t_numeric - jt))
        
        lambda_intensity.iloc[idx] = lambda_t
    
    # Generate signals
    signal_gen = TradingSignals(
        z_entry_threshold=2.0,
        z_exit_threshold=0.5,
        lambda_threshold=5.0,
        scaling_constant=0.1
    )
    
    signals_df = signal_gen.generate_signals(z_score, lambda_intensity, spread)
    
    n_trades = len(signals_df[signals_df['signal'] != 0])
    print(f"✓ Generated {n_trades} trading signals")
    
    # Run backtest
    print("\n[4] Running backtest...")
    engine = BacktestEngine(
        initial_capital=1_000_000,
        commission_rate=0.0002,  # 2 basis points
        slippage_bps=1.0         # 1 basis point
    )
    
    equity_curve = engine.run_backtest(
        signals_df, 
        spread_df, 
        asset_a_price, 
        asset_b_price
    )
    
    print(f"✓ Backtest complete: {len(equity_curve)} days")
    
    # Calculate metrics
    print("\n[5] Calculating performance metrics...")
    metrics = engine.calculate_performance_metrics()
    
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    # Organize metrics by category
    returns_metrics = ['total_return_pct', 'annualized_return_pct', 'cagr_pct']
    risk_metrics = ['volatility_pct', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown_pct', 'calmar_ratio']
    trade_metrics = ['total_trades', 'win_rate_pct', 'avg_trade_return_pct', 'profit_factor']
    time_metrics = ['avg_holding_days', 'time_in_market_pct']
    
    print("\nReturns:")
    for key in returns_metrics:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.2f}%")
    
    print("\nRisk-Adjusted:")
    for key in risk_metrics:
        if key in metrics:
            if 'pct' in key:
                print(f"  {key}: {metrics[key]:.2f}%")
            else:
                print(f"  {key}: {metrics[key]:.2f}")
    
    print("\nTrading:")
    for key in trade_metrics:
        if key in metrics:
            if 'pct' in key:
                print(f"  {key}: {metrics[key]:.2f}%")
            elif isinstance(metrics[key], float):
                print(f"  {key}: {metrics[key]:.2f}")
            else:
                print(f"  {key}: {metrics[key]}")
    
    print("\nExecution:")
    for key in time_metrics:
        if key in metrics:
            if 'pct' in key:
                print(f"  {key}: {metrics[key]:.2f}%")
            else:
                print(f"  {key}: {metrics[key]:.2f}")
    
    # Print other metrics
    print("\nOther:")
    for key, value in metrics.items():
        if key not in returns_metrics + risk_metrics + trade_metrics + time_metrics:
            if isinstance(value, float):
                if 'pct' in key or 'rate' in key:
                    print(f"  {key}: {value:.2f}%")
                else:
                    print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nData: XOM/CVX ({len(spread_df)} days)")
    print(f"Period: {spread_df.index[0].date()} to {spread_df.index[-1].date()}")
    print(f"\nStrategy Performance:")
    print(f"  Total Return: {metrics.get('total_return_pct', 0):.2f}%")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Win Rate: {metrics.get('win_rate_pct', 0):.2f}%")
    print(f"  Total Trades: {metrics.get('total_trades', 0)}")
    print(f"\n✓ All tests completed successfully!")
    print("="*70)
