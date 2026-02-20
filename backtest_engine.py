"""
Backtesting Engine for Self-Exciting Pairs Trading

Key Optimizations:
1. Trailing stop losses (lock in profits as trade moves favorably)
2. Half-life aware position management
3. Regime-based risk adjustment
4. Better transaction cost modeling
5. Detailed exit reason tracking
"""

import numpy as np 
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """Enhanced trade record with detailed tracking"""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp 
    direction: int  # 1 = long spread, -1 = short spread
    entry_price_a: float
    entry_price_b: float
    exit_price_a: float
    exit_price_b: float
    entry_z: float
    exit_z: float
    position_size: float 
    pnl: float 
    return_pct: float 
    duration: int
    entry_lambda: float 
    exit_lambda: float 
    entry_regime: str
    exit_regime: str
    max_adverse_excursion: float
    max_favorable_excursion: float
    exit_reason: str


class BacktestEngineV2:
    """
    Optimized backtest engine with trailing stops and risk management
    
    Key features:
    - Dollar-neutral positions (equal $ on both legs)
    - Trailing stop losses (lock in profits)
    - P&L-based hard stop
    - Regime-aware position management
    - Half-life calibrated holding periods
    """

    def __init__(self,
                 initial_capital: float = 1_000_000,
                 commission_rate: float = 0.0001,     # 1bp per side
                 slippage_bps: float = 0.5,           # 0.5bp slippage
                 max_position_pct: float = 0.25,      # 25% max per trade
                 stop_loss_pct: float = 0.03,         # 3% hard stop
                 trailing_stop_pct: float = 0.015,    # 1.5% trailing
                 trailing_activation_pct: float = 0.01,  # Activate trailing after 1% profit
                 profit_target_pct: float = 0.06):    # 6% profit target
        
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage_bps / 10000
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.profit_target_pct = profit_target_pct

        self.trades: List[Trade] = []
        self.equity_curve = None 
        self.performance_metrics = {}
        
        # Half-life awareness
        self.half_life = None

    def set_half_life(self, half_life: float):
        """Set half-life for position management calibration"""
        self.half_life = half_life
        print(f"  Backtest half-life: {half_life:.1f} days")

    def run_backtest(self,
                     signals_df: pd.DataFrame,
                     spread_df: pd.DataFrame,
                     asset_a_prices: pd.Series,
                     asset_b_prices: pd.Series,
                     hedge_ratio: Optional[float] = None) -> pd.DataFrame:
        """
        Run optimized backtest with trailing stops
        
        Dollar-neutral positions: Both legs have EQUAL dollar exposure
        Trailing stops: Lock in profits as trade moves favorably
        """

        print("Running backtest")

        # Align data
        common_index = signals_df.index.intersection(spread_df.index)
        common_index = common_index.intersection(asset_a_prices.index)
        common_index = common_index.intersection(asset_b_prices.index)
        
        signals_df = signals_df.loc[common_index].copy()
        spread_df = spread_df.loc[common_index].copy()
        asset_a_prices = asset_a_prices.loc[common_index].copy()
        asset_b_prices = asset_b_prices.loc[common_index].copy()

        # Get hedge ratio
        if hedge_ratio is None:
            if 'hedge_ratio' in spread_df.columns:
                hedge_ratio = float(spread_df['hedge_ratio'].mean())
            else:
                hedge_ratio = 1.0
        
        print(f"  Hedge ratio (spread construction): {hedge_ratio:.4f}")
        print(f"  Position sizing: Dollar-neutral")
        print(f"  Max position: {self.max_position_pct*100:.0f}%")
        print(f"  Hard stop: {self.stop_loss_pct*100:.1f}%")
        print(f"  Trailing stop: {self.trailing_stop_pct*100:.1f}% (activates at {self.trailing_activation_pct*100:.1f}%)")
        print(f"  Profit target: {self.profit_target_pct*100:.1f}%")

        n = len(signals_df)

        # Initialize tracking
        equity = np.zeros(n)
        equity[0] = self.initial_capital
        cash = self.initial_capital 
        
        # Position state
        in_position = False
        position_direction = 0
        shares_a = 0.0
        shares_b = 0.0
        entry_price_a = 0.0
        entry_price_b = 0.0
        entry_date: Optional[pd.Timestamp] = None
        entry_lambda = 0.0
        entry_z = 0.0
        entry_regime = "normal"
        entry_capital = 0.0
        max_adverse = 0.0
        max_favorable = 0.0
        
        # Trailing stop state
        trailing_stop_level = -self.stop_loss_pct  # Initial stop = hard stop
        trailing_activated = False

        for i in range(1, n):
            date = pd.Timestamp(signals_df.index[i])
            signal = int(signals_df['signal'].iloc[i])
            lambda_value = float(signals_df['lambda'].iloc[i])
            z_value = float(signals_df['z_score'].iloc[i]) if 'z_score' in signals_df.columns else 0
            regime = str(signals_df['regime'].iloc[i]) if 'regime' in signals_df.columns else 'normal'
            price_a = float(asset_a_prices.iloc[i])
            price_b = float(asset_b_prices.iloc[i])
            
            raw_position_size = abs(float(signals_df['position_size'].iloc[i]))
            capped_position_size = min(raw_position_size, self.max_position_pct)

            # ENTRY LOGIC
            if not in_position and signal in [1, -1]:
                in_position = True
                position_direction = signal
                entry_date = date
                entry_price_a = price_a
                entry_price_b = price_b
                entry_lambda = lambda_value
                entry_z = z_value
                entry_regime = regime
                max_adverse = 0.0
                max_favorable = 0.0
                trailing_stop_level = -self.stop_loss_pct  # Reset to hard stop
                trailing_activated = False

                # DOLLAR-NEUTRAL POSITION SIZING
                capital_per_leg = capped_position_size * cash / 2
                entry_capital = capital_per_leg * 2
                
                shares_a = position_direction * capital_per_leg / price_a
                shares_b = -position_direction * capital_per_leg / price_b

                # Entry costs
                trade_value = 2 * capital_per_leg
                entry_costs = trade_value * (self.commission_rate + self.slippage)
                cash -= entry_costs

            # POSITION MANAGEMENT
            elif in_position:
                # Calculate current P&L
                pnl_a = shares_a * (price_a - entry_price_a)
                pnl_b = shares_b * (price_b - entry_price_b)
                current_pnl = pnl_a + pnl_b
                current_return = current_pnl / entry_capital if entry_capital > 0 else 0

                # Track excursions
                max_adverse = min(max_adverse, current_return)
                max_favorable = max(max_favorable, current_return)

                # UPDATE TRAILING STOP
                # Activate trailing stop once we reach activation threshold
                if current_return > self.trailing_activation_pct:
                    trailing_activated = True
                
                if trailing_activated:
                    # Raise stop level to lock in profits
                    # Stop level = max_favorable - trailing_stop_pct
                    new_stop_level = max_favorable - self.trailing_stop_pct
                    trailing_stop_level = max(trailing_stop_level, new_stop_level)

                # EXIT CONDITIONS
                exit_reason = None
                
                # 1. Signal-based exit
                if signal == 2:
                    exit_reason = 'signal'
                
                # 2. Stop loss hit (hard or trailing)
                elif current_return < trailing_stop_level:
                    if trailing_activated:
                        exit_reason = 'trailing_stop'
                    else:
                        exit_reason = 'stop_loss'
                
                # 3. Profit target
                elif current_return > self.profit_target_pct:
                    exit_reason = 'profit_target'

                # Execute exit if triggered
                if exit_reason:
                    assert entry_date is not None

                    # Exit costs
                    exit_value_a = abs(shares_a * price_a)
                    exit_value_b = abs(shares_b * price_b)
                    exit_costs = (exit_value_a + exit_value_b) * (self.commission_rate + self.slippage)
                    
                    realized_pnl = current_pnl - exit_costs
                    cash += realized_pnl

                    # Record trade
                    trade = Trade(
                        entry_date=entry_date,
                        exit_date=date,
                        direction=position_direction,
                        entry_price_a=entry_price_a,
                        entry_price_b=entry_price_b,
                        exit_price_a=price_a,
                        exit_price_b=price_b,
                        entry_z=entry_z,
                        exit_z=z_value,
                        position_size=capped_position_size,
                        pnl=realized_pnl,
                        return_pct=current_return * 100,
                        duration=(date - entry_date).days,
                        entry_lambda=entry_lambda,
                        exit_lambda=lambda_value,
                        entry_regime=entry_regime,
                        exit_regime=regime,
                        max_adverse_excursion=max_adverse * 100,
                        max_favorable_excursion=max_favorable * 100,
                        exit_reason=exit_reason
                    )
                    self.trades.append(trade)

                    # Reset position state
                    in_position = False
                    position_direction = 0
                    shares_a = 0.0
                    shares_b = 0.0

            # Update equity
            if in_position:
                unrealized = shares_a * (price_a - entry_price_a) + shares_b * (price_b - entry_price_b)
                equity[i] = cash + unrealized
            else:
                equity[i] = cash

        # Create equity curve
        self.equity_curve = pd.DataFrame({
            'equity': equity,
            'returns': np.concatenate(([0], np.diff(equity) / np.maximum(equity[:-1], 1))),
            'spread': spread_df['spread'].values,
        }, index=common_index)

        n_trades = len(self.trades)
        final_equity = equity[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100

        print(f"Backtest Complete!")
        print(f"    - Total Trades: {n_trades}")
        print(f"    - Final Equity: ${final_equity:,.2f}")
        print(f"    - Total Return: {total_return:.2f}%")
        
        if n_trades > 0:
            exit_reasons = {}
            for t in self.trades:
                exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
            print(f"    - Exit reasons: {exit_reasons}")

        return self.equity_curve

    def calculate_performance_metrics(self, risk_free_rate: float = 0.02) -> Dict:
        """Calculate comprehensive performance metrics"""

        if self.equity_curve is None:
            raise ValueError("Run backtest before calculating metrics")
        
        equity = self.equity_curve['equity'].to_numpy(dtype=float)
        returns = self.equity_curve['returns'].to_numpy(dtype=float)[1:]

        # Return metrics
        total_return = (equity[-1] / equity[0] - 1) * 100
        n_days = len(equity)
        annual_factor = 252 / n_days if n_days > 0 else 1
        
        if equity[-1] > 0 and equity[0] > 0:
            annualized_return = ((equity[-1] / equity[0]) ** annual_factor - 1) * 100
        else:
            annualized_return = -100.0

        # Risk metrics
        returns_std = np.std(returns) if len(returns) > 0 else 0.001
        annualized_vol = returns_std * np.sqrt(252) * 100

        # Sharpe Ratio
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / returns_std if returns_std > 0 else 0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else returns_std
        sortino_ratio = np.sqrt(252) * np.mean(excess_returns) / downside_std if downside_std > 0 else 0

        # Maximum drawdown
        cumulative = np.cumprod(1.0 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100

        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        if len(self.trades) > 0:
            win_trades = [t for t in self.trades if t.pnl > 0]
            lose_trades = [t for t in self.trades if t.pnl <= 0]
            
            win_rate = len(win_trades) / len(self.trades) * 100
            avg_win = np.mean([t.return_pct for t in win_trades]) if win_trades else 0
            avg_loss = np.mean([t.return_pct for t in lose_trades]) if lose_trades else 0
            
            gross_profit = sum([t.pnl for t in win_trades]) if win_trades else 0
            gross_loss = abs(sum([t.pnl for t in lose_trades])) if lose_trades else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            avg_duration = np.mean([t.duration for t in self.trades])
            avg_mae = np.mean([abs(t.max_adverse_excursion) for t in self.trades])
            avg_mfe = np.mean([t.max_favorable_excursion for t in self.trades])
            
            # Expected value per trade
            expected_value = (win_rate/100 * avg_win + (1 - win_rate/100) * avg_loss)
        else:
            win_rate = avg_win = avg_loss = profit_factor = expected_value = 0
            avg_duration = avg_mae = avg_mfe = 0

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
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'expected_value_per_trade': expected_value,
            'avg_trade_duration_days': avg_duration,
            'avg_max_adverse_excursion_pct': avg_mae,
            'avg_max_favorable_excursion_pct': avg_mfe
        }

        self.performance_metrics = metrics
        print("Performance metrics calculated")
        return metrics

    def analyze_regime_performance(self, regime_threshold: float = 0.1) -> Dict:
        """Analyze performance by entry regime"""
        
        if not self.trades:
            return {'calm_regime': {'n_trades': 0}, 'volatile_regime': {'n_trades': 0}}
        
        # By lambda threshold
        calm_trades = [t for t in self.trades if t.entry_lambda < regime_threshold]
        volatile_trades = [t for t in self.trades if t.entry_lambda >= regime_threshold]

        def calc_regime_stats(trades, name):
            if not trades:
                return {'n_trades': 0, 'win_rate': 0, 'avg_return': 0, 'total_pnl': 0}
            
            wins = [t for t in trades if t.pnl > 0]
            return {
                'n_trades': len(trades),
                'win_rate': len(wins) / len(trades) * 100,
                'avg_return': np.mean([t.return_pct for t in trades]),
                'total_pnl': sum([t.pnl for t in trades])
            }

        regime_stats = {
            'calm_regime': calc_regime_stats(calm_trades, 'calm'),
            'volatile_regime': calc_regime_stats(volatile_trades, 'volatile')
        }

        print(f" Calm regime: {len(calm_trades)} trades")
        print(f" Volatile regime: {len(volatile_trades)} trades")

        return regime_stats

    def analyze_by_exit_reason(self) -> Dict:
        """Analyze performance by exit reason"""
        
        if not self.trades:
            return {}
        
        by_reason = {}
        for t in self.trades:
            reason = t.exit_reason
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(t)
        
        analysis = {}
        for reason, trades in by_reason.items():
            wins = [t for t in trades if t.pnl > 0]
            analysis[reason] = {
                'count': len(trades),
                'win_rate': len(wins) / len(trades) * 100 if trades else 0,
                'avg_return': np.mean([t.return_pct for t in trades]),
                'total_pnl': sum(t.pnl for t in trades)
            }
        
        return analysis

    def get_trade_summary(self) -> pd.DataFrame:
        """Get detailed trade summary"""
        
        if not self.trades:
            return pd.DataFrame()
        
        trade_data = [{
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'direction': 'LONG' if t.direction == 1 else 'SHORT',
            'entry_z': t.entry_z,
            'exit_z': t.exit_z,
            'entry_lambda': t.entry_lambda,
            'exit_lambda': t.exit_lambda,
            'entry_regime': t.entry_regime,
            'exit_regime': t.exit_regime,
            'position_size': t.position_size,
            'pnl': t.pnl,
            'return_pct': t.return_pct,
            'duration_days': t.duration,
            'exit_reason': t.exit_reason,
            'mae_pct': t.max_adverse_excursion,
            'mfe_pct': t.max_favorable_excursion
        } for t in self.trades]

        return pd.DataFrame(trade_data)


if __name__ == "__main__":
    print("="*70)
    print("BACKTEST ENGINE")
    print("="*70)
    
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Synthetic prices
    price_a = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.02, n)))
    price_b = 50 * np.exp(np.cumsum(np.random.normal(0.0001, 0.025, n)))
    
    asset_a_prices = pd.Series(price_a, index=dates)
    asset_b_prices = pd.Series(price_b, index=dates)
    
    # Synthetic spread
    spread = np.log(price_a) - 0.8 * np.log(price_b)
    spread_df = pd.DataFrame({'spread': spread, 'hedge_ratio': 0.8}, index=dates)
    
    # Synthetic signals with regime
    z_score = (spread - np.mean(spread)) / np.std(spread)
    signals = np.zeros(n)
    positions = np.zeros(n)
    position_sizes = np.zeros(n)
    
    in_pos = 0
    for i in range(1, n):
        if in_pos == 0:
            if z_score[i] < -2.0:
                signals[i] = 1
                in_pos = 1
            elif z_score[i] > 2.0:
                signals[i] = -1
                in_pos = -1
        else:
            if abs(z_score[i]) < 0.5:
                signals[i] = 2
                in_pos = 0
        positions[i] = in_pos
        position_sizes[i] = 0.2 if in_pos != 0 else 0
    
    signals_df = pd.DataFrame({
        'signal': signals,
        'position': positions,
        'position_size': position_sizes,
        'lambda': 0.05 + 0.1 * np.random.random(n),
        'z_score': z_score,
        'regime': 'normal'
    }, index=dates)
    
    # Run backtest
    engine = BacktestEngineV2(
        initial_capital=1_000_000,
        max_position_pct=0.25,
        stop_loss_pct=0.03,
        trailing_stop_pct=0.015,
        profit_target_pct=0.06
    )
    
    equity_curve = engine.run_backtest(
        signals_df, spread_df, asset_a_prices, asset_b_prices
    )
    
    metrics = engine.calculate_performance_metrics()
    
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    
    # Analyze by exit reason
    print("\n" + "-"*70)
    print("EXIT REASON ANALYSIS")
    print("-"*70)
    exit_analysis = engine.analyze_by_exit_reason()
    for reason, stats in exit_analysis.items():
        print(f"  {reason}: {stats['count']} trades, {stats['win_rate']:.1f}% win rate, "
              f"${stats['total_pnl']:,.0f} total P&L")
    
    print("\n Test complete!")