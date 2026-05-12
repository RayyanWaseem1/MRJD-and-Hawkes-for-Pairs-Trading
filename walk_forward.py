"""
Walk-Forward Backtesting Engine for Self-Exciting Pairs Trading.

At each quarter-end, refits all models on training data up to that date,
then trades the next quarter with frozen params. Concatenates quarterly
equity curves into one continuous OOS curve.
"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(project_root, ".matplotlib_cache"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import ConfigV2
from main import SelfExcitingPairsTradingV4, ModelBundle, run_with_output_capture
from backtest_engine import BacktestEngineV2, compute_alpha_tstat


class WalkForwardEngine:
    """Quarterly walk-forward backtest."""

    def __init__(self, config: ConfigV2, min_train_days: int = 504):
        self.config         = config
        self.min_train_days = min_train_days
        self.quarterly_results: List[Dict] = []
        self.oos_equity_curve: Optional[pd.DataFrame] = None
        self.oos_metrics: Dict = {}

    def _get_quarter_end_dates(self, index: pd.DatetimeIndex) -> List[pd.Timestamp]:
        if len(index) <= self.min_train_days:
            return []

        if hasattr(index, 'tz') and index.tz is not None:
            naive_index = index.tz_convert(None)
        else:
            naive_index = index

        first_eligible = naive_index[self.min_train_days]
        last           = naive_index[-1]

        # 'QE' for pandas >= 2.2, fall back to 'Q' for older versions
        try:
            cal_q_ends = pd.date_range(start=first_eligible, end=last, freq='QE')
        except ValueError:
            cal_q_ends = pd.date_range(start=first_eligible, end=last, freq='Q')

        result = []
        for q in cal_q_ends:
            mask = naive_index <= q
            if not mask.any():
                continue
            actual = index[mask][-1]
            if not result or actual != result[-1]:
                result.append(actual)

        return result

    def run(self) -> Dict:
        cfg = self.config

        print("\n" + "=" * 70)
        print("WALK-FORWARD BACKTESTING ENGINE")
        print(f"  Min training: {self.min_train_days} days "
              f"(~{self.min_train_days // 252} years)")
        print(f"  Rebalance frequency: quarterly")
        print("=" * 70)

        system = SelfExcitingPairsTradingV4(cfg)
        system._acquire_data()
        full_spread  = system.spread_df
        full_cleaned = system.cleaned_data

        if full_spread is None or full_cleaned is None:
            raise RuntimeError("Failed to load full dataset")
        if not isinstance(full_spread.index, pd.DatetimeIndex):
            raise TypeError("Walk-forward backtesting requires full_spread to use a DatetimeIndex")
        full_index = full_spread.index

        print(f"\nFull dataset: {full_index[0].date()} → "
              f"{full_index[-1].date()}  ({len(full_spread)} obs)")

        quarter_ends = self._get_quarter_end_dates(full_index)
        if not quarter_ends:
            raise RuntimeError("Not enough data for the requested min_train_days")

        print(f"\nRebalancing dates: {len(quarter_ends)} quarters")
        print(f"  First rebalance: {quarter_ends[0].date()}")
        print(f"  Last  rebalance: {quarter_ends[-1].date()}")

        all_equity_curves = []
        for i, q_end in enumerate(quarter_ends):
            after_q = full_index[full_index > q_end]
            if len(after_q) == 0:
                continue
            eval_start = after_q[0]
            eval_end   = quarter_ends[i + 1] if i + 1 < len(quarter_ends) else full_index[-1]

            if eval_start >= eval_end:
                continue

            n_train = (full_index <= q_end).sum()
            n_eval  = ((full_index >= eval_start) &
                       (full_index <= eval_end)).sum()

            print(f"\n--- Quarter {i + 1}/{len(quarter_ends)} ---")
            print(f"  Train: ≤ {q_end.date()}  ({n_train} obs)")
            print(f"  Eval : {eval_start.date()} → {eval_end.date()}  ({n_eval} obs)")

            train_spread  = full_spread.loc[:q_end]
            train_cleaned = {
                'asset_a': full_cleaned['asset_a'].loc[:q_end],
                'asset_b': full_cleaned['asset_b'].loc[:q_end],
            }

            try:
                bundle = system._fit_models(train_spread, train_cleaned)
            except Exception as e:
                print(f"  Skipping quarter — model fitting failed: {e}")
                continue

            span_spread = full_spread.loc[:eval_end]
            span_cleaned = {
                'asset_a': full_cleaned['asset_a'].loc[:eval_end],
                'asset_b': full_cleaned['asset_b'].loc[:eval_end],
            }

            try:
                artifacts = system._compute_full_sample_artifacts(span_spread, bundle)
            except Exception as e:
                print(f"  Skipping quarter — artifact computation failed: {e}")
                continue

            try:
                result = system._evaluate_period(
                    span_spread, span_cleaned, artifacts, bundle,
                    eval_start.strftime('%Y-%m-%d'),
                    eval_end.strftime('%Y-%m-%d'),
                    label=f"Q{i + 1} OOS ({eval_start.date()} → {eval_end.date()})",
                )
            except Exception as e:
                print(f"  Skipping quarter — evaluation failed: {e}")
                continue

            if not result or 'equity_curve' not in result:
                continue

            self.quarterly_results.append({
                'quarter':    i + 1,
                'train_end':  q_end,
                'eval_start': eval_start,
                'eval_end':   eval_end,
                'bundle':     bundle,
                'metrics':    result['metrics'],
                'equity_curve': result['equity_curve'],
                'signals_df': result.get('signals_df'),
                'engine': result.get('engine'),
            })
            all_equity_curves.append(result['equity_curve'])

        if not all_equity_curves:
            print("\nNo quarterly results were collected.")
            return {}

        self.oos_equity_curve = self._stitch_equity_curves(
            all_equity_curves, base_capital=cfg.backtest.initial_capital
        )

        temp_engine = BacktestEngineV2(initial_capital=cfg.backtest.initial_capital)
        temp_engine.equity_curve = self.oos_equity_curve
        oos_metrics = temp_engine.calculate_performance_metrics(
            risk_free_rate=cfg.backtest.risk_free_rate
        )

        oos_metrics['total_trades'] = sum(
            qr['metrics'].get('total_trades', 0) for qr in self.quarterly_results
        )
        total_winning = sum(
            qr['metrics'].get('total_trades', 0) *
            qr['metrics'].get('win_rate_pct', 0) / 100.0
            for qr in self.quarterly_results
        )
        if oos_metrics['total_trades'] > 0:
            oos_metrics['win_rate_pct'] = 100.0 * total_winning / oos_metrics['total_trades']
        else:
            oos_metrics['win_rate_pct'] = 0.0

        oos_alpha = compute_alpha_tstat(
            self.oos_equity_curve,
            cfg.backtest.benchmark_csv,
            cfg.backtest.risk_free_rate,
        )
        oos_metrics.update(oos_alpha)
        self.oos_metrics = oos_metrics

        self._print_summary()

        return {
            'oos_metrics':        oos_metrics,
            'quarterly_results':  self.quarterly_results,
            'oos_equity_curve':   self.oos_equity_curve,
        }

    @staticmethod
    def _stitch_equity_curves(curves: List[pd.DataFrame],
                              base_capital: float) -> pd.DataFrame:
        """Chain quarterly equity curves into one continuous compound curve."""
        rescaled = []
        running_equity = base_capital

        for ec in curves:
            if ec is None or ec.empty:
                continue
            start_eq = float(ec['equity'].iloc[0])
            if start_eq <= 0:
                continue
            scale = running_equity / start_eq
            scaled_eq = ec['equity'] * scale
            rescaled.append(scaled_eq)
            running_equity = float(scaled_eq.iloc[-1])

        if not rescaled:
            return pd.DataFrame(columns=['equity', 'returns'])

        oos_equity  = pd.concat(rescaled).sort_index()
        oos_equity  = oos_equity[~oos_equity.index.duplicated(keep='last')]
        oos_returns = oos_equity.pct_change().fillna(0.0)

        return pd.DataFrame({'equity': oos_equity, 'returns': oos_returns})

    def _print_summary(self):
        m = self.oos_metrics

        print("\n" + "=" * 76)
        print("WALK-FORWARD OOS PERFORMANCE SUMMARY")
        print(f"  Quarters evaluated: {len(self.quarterly_results)}")
        if self.oos_equity_curve is not None and not self.oos_equity_curve.empty:
            print(f"  OOS span: {self.oos_equity_curve.index[0].date()} → "
                  f"{self.oos_equity_curve.index[-1].date()}")
        print("=" * 76)

        print(f"\n  Returns")
        print(f"    Total Return:      {m.get('total_return_pct', 0):.2f}%")
        print(f"    Annualized Return: {m.get('annualized_return_pct', 0):.2f}%")
        print(f"    Annualized Vol:    {m.get('annualized_volatility_pct', 0):.2f}%")

        print(f"\n  Risk-Adjusted")
        print(f"    Sharpe Ratio:      {m.get('sharpe_ratio', 0):.3f}")
        print(f"    Sortino Ratio:     {m.get('sortino_ratio', 0):.3f}")
        print(f"    Max Drawdown:      {m.get('max_drawdown_pct', 0):.2f}%")
        print(f"    Calmar Ratio:      {m.get('calmar_ratio', 0):.3f}")

        print(f"\n  Alpha vs SPY (HEADLINE)")
        print(f"    Alpha (annualized): {m.get('alpha_annualized', 0) * 100:.4f}%")
        print(f"    Alpha T-Stat:       {m.get('alpha_tstat', 0):.3f}")
        print(f"    Alpha P-Value:      {m.get('alpha_pvalue', 1):.4f}")
        print(f"    Beta:               {m.get('beta', 0):.3f}")
        print(f"    R²:                 {m.get('r_squared', 0):.4f}")
        print(f"    Information Ratio:  {m.get('information_ratio', 0):.3f}")

        print(f"\n  Trading Activity")
        print(f"    Total Trades: {m.get('total_trades', 0)}")
        print(f"    Win Rate:     {m.get('win_rate_pct', 0):.2f}%")

        print(f"\n  Per-Quarter Performance:")
        print(f"  {'Q':<4} {'Train ≤':<12} {'Eval window':<28} "
              f"{'Ann.Ret%':>10} {'Sharpe':>8} {'Trades':>8}")
        print(f"  {'-' * 76}")
        for qr in self.quarterly_results:
            qm    = qr['metrics']
            ann   = qm.get('annualized_return_pct', 0)
            shrp  = qm.get('sharpe_ratio', 0)
            trd   = qm.get('total_trades', 0)
            ev    = f"{qr['eval_start'].date()} → {qr['eval_end'].date()}"
            print(f"  Q{qr['quarter']:<3} {str(qr['train_end'].date()):<12} "
                  f"{ev:<28} {ann:>10.2f} {shrp:>8.3f} {trd:>8}")
        print("=" * 76)

    def _default_output_dir(self) -> str:
        cfg = self.config.data
        pair_slug = f"{cfg.asset_a_symbol}_{cfg.asset_b_symbol}".replace("/", "_")
        return os.path.join(project_root, "outputs", pair_slug, "walk_forward")

    def _save_visualizations(self, output_dir: str) -> None:
        """Save walk-forward equity and quarterly summary plots."""
        if self.oos_equity_curve is None or self.oos_equity_curve.empty:
            return

        import matplotlib.pyplot as plt

        vis_cfg = self.config.visualization
        fig, axes = plt.subplots(2, 1, figsize=(vis_cfg.figure_size[0], vis_cfg.figure_size[1] + 2), sharex=True)

        equity = self.oos_equity_curve['equity']
        axes[0].plot(equity.index, equity.values, color='blue', linewidth=1.2)
        axes[0].set_title('Walk-Forward OOS Equity Curve')
        axes[0].set_ylabel('Equity')
        axes[0].grid(True, alpha=0.3)

        drawdown = (equity / equity.cummax() - 1.0) * 100
        axes[1].plot(drawdown.index, drawdown.values, color='red', linewidth=1.0)
        axes[1].set_title('Walk-Forward OOS Drawdown')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        output_path = os.path.join(output_dir, f'walk_forward_equity.{vis_cfg.plot_format}')
        fig.savefig(output_path, dpi=vis_cfg.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved walk-forward equity visualization: {output_path}")

        if self.quarterly_results:
            quarters = [f"Q{qr['quarter']}" for qr in self.quarterly_results]
            returns = [qr['metrics'].get('annualized_return_pct', 0.0) for qr in self.quarterly_results]
            sharpes = [qr['metrics'].get('sharpe_ratio', 0.0) for qr in self.quarterly_results]

            fig, axes = plt.subplots(2, 1, figsize=(vis_cfg.figure_size[0], vis_cfg.figure_size[1] + 2))
            axes[0].bar(quarters, returns, color='steelblue')
            axes[0].axhline(0, color='black', linewidth=0.8)
            axes[0].set_title('Quarterly Annualized Return')
            axes[0].set_ylabel('Return (%)')
            axes[0].tick_params(axis='x', rotation=90)
            axes[0].grid(True, axis='y', alpha=0.3)

            axes[1].bar(quarters, sharpes, color='darkorange')
            axes[1].axhline(0, color='black', linewidth=0.8)
            axes[1].set_title('Quarterly Sharpe Ratio')
            axes[1].set_ylabel('Sharpe')
            axes[1].tick_params(axis='x', rotation=90)
            axes[1].grid(True, axis='y', alpha=0.3)

            fig.tight_layout()
            output_path = os.path.join(output_dir, f'quarterly_performance.{vis_cfg.plot_format}')
            fig.savefig(output_path, dpi=vis_cfg.dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved quarterly performance visualization: {output_path}")

    def save_results(self, output_dir: Optional[str] = None) -> None:
        """Save CSV and PNG artifacts from the completed walk-forward run."""
        if output_dir is None:
            output_dir = self._default_output_dir()

        print(f"\nSaving walk-forward results to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        if self.oos_metrics:
            pd.DataFrame([self.oos_metrics]).to_csv(
                os.path.join(output_dir, 'walk_forward_metrics.csv'),
                index=False,
            )

        quarterly_rows = []
        for qr in self.quarterly_results:
            row = {
                'quarter': qr['quarter'],
                'train_end': qr['train_end'],
                'eval_start': qr['eval_start'],
                'eval_end': qr['eval_end'],
            }
            row.update(qr.get('metrics', {}))
            quarterly_rows.append(row)
        if quarterly_rows:
            pd.DataFrame(quarterly_rows).to_csv(
                os.path.join(output_dir, 'quarterly_metrics.csv'),
                index=False,
            )

        if self.oos_equity_curve is not None and not self.oos_equity_curve.empty:
            self.oos_equity_curve.to_csv(
                os.path.join(output_dir, 'walk_forward_equity_curve.csv'),
                index_label='date',
            )

        quarterly_equity = []
        quarterly_signals = []
        quarterly_trades = []
        bundle_rows = []
        for qr in self.quarterly_results:
            quarter = qr['quarter']
            equity_curve = qr.get('equity_curve')
            signals_df = qr.get('signals_df')
            engine = qr.get('engine')
            bundle = qr.get('bundle')

            if isinstance(equity_curve, pd.DataFrame) and not equity_curve.empty:
                ec = equity_curve.copy()
                ec.insert(0, 'quarter', quarter)
                quarterly_equity.append(ec)

            if isinstance(signals_df, pd.DataFrame) and not signals_df.empty:
                sig = signals_df.copy()
                sig.insert(0, 'quarter', quarter)
                quarterly_signals.append(sig)

            if engine is not None:
                trades = engine.get_trade_summary()
                if not trades.empty:
                    trades.insert(0, 'quarter', quarter)
                    quarterly_trades.append(trades)

            if isinstance(bundle, ModelBundle):
                bundle_row = {
                    'quarter': quarter,
                    'train_end': qr['train_end'],
                    'eval_start': qr['eval_start'],
                    'eval_end': qr['eval_end'],
                    'half_life': bundle.half_life,
                    'use_hawkes_regimes': bundle.use_hawkes_regimes,
                }
                bundle_row.update({f"hawkes_{k}": v for k, v in bundle.hawkes_params.items()})
                bundle_row.update({f"mrjd_{k}": v for k, v in bundle.mrjd_params.items()})
                bundle_row.update({f"lambda_{k}": v for k, v in bundle.lambda_percentiles.items()})
                bundle_rows.append(bundle_row)

        if quarterly_equity:
            pd.concat(quarterly_equity).to_csv(
                os.path.join(output_dir, 'quarterly_equity_curves.csv'),
                index_label='date',
            )
        if quarterly_signals:
            pd.concat(quarterly_signals).to_csv(
                os.path.join(output_dir, 'quarterly_trading_signals.csv'),
                index_label='date',
            )
        if quarterly_trades:
            pd.concat(quarterly_trades, ignore_index=True).to_csv(
                os.path.join(output_dir, 'quarterly_trade_summary.csv'),
                index=False,
            )
        if bundle_rows:
            pd.DataFrame(bundle_rows).to_csv(
                os.path.join(output_dir, 'quarterly_model_bundles.csv'),
                index=False,
            )

        self._save_visualizations(output_dir)
        print("Walk-forward results saved successfully")


def main():
    """Run a walk-forward backtest with the default configured pair."""
    config = ConfigV2()

    config.trading.z_entry_threshold = 2.0
    config.trading.z_exit_threshold  = 0.5
    config.trading.lambda_threshold  = 0.15

    engine  = WalkForwardEngine(config, min_train_days=504)
    results = engine.run()
    engine.save_results()

    print("\n" + "=" * 70)
    print("WALK-FORWARD EXECUTION COMPLETE")
    print("=" * 70)

    return engine, results


if __name__ == "__main__":
    try:
        engine, results = run_with_output_capture("walk_forward_output.txt", main)
    except Exception:
        sys.exit(1)
