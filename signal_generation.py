"""
Trading signal generation

Changes from the audited version:
1. THE EXIT CHAIN. Exits were an `elif` chain:

    if abs(z) < z _exit and held >= min_hold:   # 1 mean reversion
    elif held >= min_hold:                      #2 profit target
        ...inner if/elif may leave exit_signal False...
    elif held >= max_hold:                      #3 time stop 
    elif regime == CRISIS and entry != CRISIS:  #4 regime exit 
    elif position == 1 and z < entry_z - 2.5    #5 emergency stop 

Branch 2 was entered whenever `held >= min_hold` regardless of whether its
inner conditions fired, and min_hold (0.5 x HL) is always reached before
max_hold (1.5 x HL). So condition 3 could NEVER fire, condition 4 only in 
[min_hold/2, min_hold), and condition 5 only before min_hold. All ten
committed trade logs across all five pairs confirm it: zero `max_hold`, 
zero `regime_crisis`, zero `emergency_stop` exits, ever. 

Exits are now INDEPENDENT checks collected into a list, with a documented
priority order applied afterwards. 

2. JUMP ENTRIES GO THROUGH THE SAME GATES. The jump-assisted entry was an 
`elif` on the main z-threshold test, so it sat OUTSIDE the CRISIS block and 
the lambda-decay filter -- entering at a 35% looser threshold precisely on 
jump days, i.e. mid cascade when lambda is spiking. That is the exact 
opposite of the project's stated rule ("wait for the cascade to subside").
Jump entries now pass through both filters. 

3. REGIMES ARE EXCITATION-BASED. lambda(t) = lambda_bar + excitation >= 
lambda_bar ALWAYS, so percentile bucketing of a spike train is degenerate: 
p25 equalled lambda_bar EXACTLY for GS/MS and CVX/XOM (CALM unreachable),
and the [p25, p75) band labelled "NORMAL" spanned a 10x intensity range.
Regimes are now cut on relative excess intensity (lambda_t - lambda_bar) / lambda_bar,
which is interpretable and cannot degenerate.

4. `HawkesRegime` is an Enum, consistent with `SignalType` (it was a 
dataclass of string class attributes).
"""

from __future__ import annotations 

from enum import Enum 
from typing import Dict, List, Optional, Tuple 

import numpy as np 
import pandas as pd 

__all__ = ["SignalType", "HawkesRegime", "TradingSignals", "EXIT_PRIORITY"]

class SignalType(Enum):
    NO_SIGNAL = 0
    LONG = 1
    SHORT = -1
    CLOSE = 2

class HawkesRegime(str, Enum):
    """ Regime by excess Hawkes intensity over baseline"""
    CALM = "calm"
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS = "crisis"

# Exit priority, most urgent first. Applied after ALL conditions are 
# evaluated independently, so no condition can mask another
EXIT_PRIORITY: Tuple[str, ...] = (
    "emergency_stop",
    "regime_crisis",
    "max_hold",
    "profit_target",
    "mean_reversion",
)

class TradingSignals:
    """ Generate entry/exit signals from a spread, a z-score, and (optionally) Hawkes intensity"""

    def __init__(
        self,
        z_entry_threshold: float = 2.0,
        z_exit_threshold: float = 0.5,
        lambda_decay_lookback: int = 5,
        min_lambda_decay_pct: float = 0.15,
        max_position_size: float = 0.25,
        min_position_size: float = 0.10,
        min_hold_fraction: float = 0.5,
        target_hold_fraction: float = 0.8,
        max_hold_fraction: float = 0.8,
        max_holding_period_cap: int = 120,
        use_jump_entries: bool = True,
        use_hawkes_regimes: bool = True,
        z_lookback: int = 60,
        emergency_z_move: float = 2.5,
        regime_excess_calm: float = 0.05,
        regime_excess_elevated: float = 1.0,
        regime_excess_crisis: float = 5.0,
        verbose: bool = True,
    ):
        self.z_entry = z_entry_threshold
        self.z_exit = z_exit_threshold 

        self.lambda_decay_lookback = lambda_decay_lookback
        self.min_lambda_decay_cpt = min_lambda_decay_pct

        self.max_position = max_position_size 
        self.min_position = min_position_size

        self.min_hold_fraction = min_hold_fraction
        self.target_hold_fraction = target_hold_fraction
        self.max_hold_fraction = max_hold_fraction 
        self.max_holding_period_cap = max_holding_period_cap

        self.use_jump_entries = use_jump_entries
        self.use_hawkes_regimes = use_hawkes_regimes
        self.z_lookback = z_lookback
        self.emergency_z_move = emergency_z_move 

        self.regime_excess_calm = regime_excess_calm 
        self.regime_excess_elevated = regime_excess_elevated 
        self.regime_excess_crisis = regime_excess_crisis

        self.verbose = verbose 

        self.half_life: Optional[float] = None 
        self.min_hold = 0
        self.target_hold = 0
        self.max_holding_period = 30 
        self.lambda_baseline: Optional[float] = None 

        # Diagnostics -- reported, not silently accumulated 
        self.entries_blocked_by_regime = 0
        self.entries_blocked_by_decay = 0
        self.jump_entries_taken = 0
        self.exit_reason_counts: Dict[str, int] = {}

    def _log(self, *args) -> None:
        if self.verbose:
            print(*args)

    ##########

    def set_half_life(self, half_life: float) -> None:
        """ Calibrate holding periods to the spread's half-life (trading days)"""
        if not np.isfinite(half_life) or half_life <= 0:
            half_life = 30.0

        self.half_life = float(half_life)
        self.min_hold = max(int(half_life * self.min_hold_fraction), 1)
        self.target_hold = max(int(half_life * self.target_hold_fraction), 1)
        self.max_holding_period = min(
            max(int(half_life * self.max_hold_fraction), 2), self.max_holding_period_cap
        )

        self._log(
            f" Half life {half_life:.1f}d -> hold [{self.min_hold}, "
            f"{self.max_holding_period}] trading days (target {self.target_hold})"
        )

    def set_lambda_baseline(self, lambda_bar: float) -> None:
        """
        Anchor regime classification to the FITTED baseline intensity 

        Regimes are cut on excess over lambda_bar, so this must come from the 
        training bundle -- not from the evaluation period's own distribution
        """
        self.lambda_baseline = float(lambda_bar)


    #####

    def _get_regime(self, lambda_t: float) -> HawkesRegime:
        """
        Classify by relative excess intensity over baseline.

            excess = (lambda_t - lambda_bar) / lambda_bar

        Percentile bucketing cannot work here: lambda(t) >= lambda_bar by 
        construction, so the lower quantiles pile up on an atom at the 
        baseline 
        """

        if not self.use_hawkes_regimes or self.lambda_baseline is None:
            return HawkesRegime.NORMAL

        base = self.lambda_baseline 
        if base <= 0:
            return HawkesRegime.NORMAL

        excess = (lambda_t - base) / base 

        if excess < self.regime_excess_calm:
            return HawkesRegime.CALM
        if excess < self.regime_excess_elevated:
            return HawkesRegime.NORMAL 
        if excess < self.regime_excess_crisis:
            return HawkesRegime.ELEVATED
        return HawkesRegime.CRISIS 

    def _get_regime_thresholds(self, regime: HawkesRegime) -> Tuple[float, float, int]:
        """ Regime-adjusted (entry threshold, exit threshold, max hold)"""
        if regime == HawkesRegime.CALM:
            return self.z_entry * 0.85, self.z_exit * 0.85, int(self.max_holding_period * 1.2)
        if regime == HawkesRegime.NORMAL:
            return self.z_entry, self.z_exit, self.max_holding_period 
        if regime == HawkesRegime.ELEVATED:
            return self.z_entry * 1.25, self.z_exit * 1.2, int(self.max_holding_period * 0.75)
        return self.z_entry * 1.5, self.z_exit * 1.3, int(self.max_holding_period * 0.5)

    def _is_lambda_decaying(self, lambda_series: pd.Series, i: int) -> bool:
        """
        True when it is safe to enter: either no cascade is in progress, or one 
        is and it has decayed far enoguh from its recent peak. 

        The "wait for the cascade to subside" rule only has meaning when there 
        IS a cascade. When alpha ~ 0 -- which is what the corrected jump 
        detection produces on most of these pairs -- lambda(t) sits flat at 
        lambda_bar, no decay can ever be observed, and requiring a 15% drop 
        blocks EVERY entry for the entire sample. Gate on elevation first
        """

        if not self.use_hawkes_regimes:
            return True 
        if i < self.lambda_decay_lookback:
            return True 

        current = float(lambda_series.iloc[i])

        #No cascade in progress -> nothing to wait for 
        if self.lambda_baseline and self.lambda_baseline > 0:
            excess = (current - self.lambda_baseline) / self.lambda_baseline 
            if excess < self.regime_excess_calm:
                return True 

        window = lambda_series.iloc[i - self.lambda_decay_lookback : i + 1]
        peak = float(window.max())
        if peak <= 0:
            return True 
        return ((peak - current) / peak) >= self.min_lambda_decay_cpt

    def _position_size(self, z: float, lambda_t: float, regime: HawkesRegime) -> float:
        """ Size on signal strength, intensity, and regime"""
        z_factor = min(abs(z) / 3.0, 1.5)

        if self.use_hawkes_regimes and self.lambda_baseline and self.lambda_baseline > 0:
            excess = max((lambda_t - self.lambda_baseline) / self.lambda_baseline, 0.0)
            lambda_factor = float(np.clip(1.5 - excess / max(self.regime_excess_elevated, 1e-9), 0.5, 1.5))
        else:
            lambda_factor = 1.0 

        regime_factor = {
            HawkesRegime.CALM: 1.2, 
            HawkesRegime.NORMAL: 1.0,
            HawkesRegime.ELEVATED: 0.7,
            HawkesRegime.CRISIS: 0.5,
        }[regime]

        size = self.max_position * z_factor * lambda_factor * regime_factor 
        return float(max(self.min_position, min(size, self.max_position)))

    def calculate_empirical_zscore(
        self, spread: pd.Series, lookback: Optional[int] = None
    ) -> pd.Series:
        """ Rolling z-score. Uses a strictly preceding window (no same-bar leakage)"""
        lookback = lookback or self.z_lookback
        prior = spread.shift(1)
        mean = prior.rolling(window = lookback, min_periods = 20).mean()
        sd = prior.rolling(window = lookback, min_periods = 20).std()
        with np.errstate(divide = "ignore", invalid = "ignore"):
            z = (spread - mean) / sd 
        return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ######

    def generate_signals(
        self,
        spread: pd.Series,
        lambda_intensity: Optional[pd.Series] = None,
        jump_indicator: Optional[pd.Series] = None,
        z_score: Optional[pd.Series] = None,
        half_life: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Produce entry/exit signals

        Entry (ALL must hold, for both normal and jump-assisted entries):
            1. |z| above the regime-adjusted threshold (0.65x for jump entries)
            2. regime is not CRISIS
            3. lambda is in a decay phase 

        Exit (evaluated INDEPENDENTLY, resolved by EXIT_PRIORITY):
            - emergency_stop: z moved `emergency_z_move` against the entry 
            - regime_crisis: regeime escalated to CRISIS after a non-CRISIS entry
            - max_hold: holding period exceeded
            - profit_target: z crossed through zero past the target
            - mean_reversion: |z| inside the exit band, min hold satisfied
        """
        self._log("Generating trading signals...")

        if half_life is not None:
            self.set_half_life(half_life)
        elif self.half_life is None:
            self.set_half_life(30.)

        index = spread.index 
        if lambda_intensity is not None:
            index = index.intersection(lambda_intensity.index)
        if jump_indicator is not None:
            index = index.intersection(jump_indicator.index)

        spread = spread.loc[index]
        if lambda_intensity is not None:
            lambda_intensity = lambda_intensity.loc[index]
        else:
            lambda_intensity = pd.Series(
                self.lambda_baseline if self.lambda_baseline else 0.0, index = index
            )
        if jump_indicator is not None:
            jump_indicator = jump_indicator.loc[index]

        if z_score is None:
            z_score = self.calculate_empirical_zscore(spread)
        else:
            z_score = z_score.loc[index]

        if self.lambda_baseline is None and self.use_hawkes_regimes:
            self.lambda_baseline = float(lambda_intensity.min())

        n = len(spread)
        signals = np.zeros(n)
        positions = np.zeros(n)
        sizes = np.zeros(n)
        regimes: List[str] = [HawkesRegime.NORMAL.value] * n
        exit_reasons: List[str] = [""] * n 

        self.entries_blocked_by_regime = 0
        self.entries_blocked_by_decay = 0
        self.jump_entries_taken = 0 
        self.exit_reason_counts = {}

        position = 0
        entry_idx = 0
        entry_z = 0.0
        entry_regime = HawkesRegime.NORMAL
        current_size = 0.0 

        for i in range(1, n):
            z_t = float(z_score.iloc[i])
            lambda_t = float(lambda_intensity.iloc[i])
            regime = self._get_regime(lambda_t)
            regimes[i] = regime.value 

            z_entry_adj, z_exit_adj, max_hold_adj = self._get_regime_thresholds(regime)
            is_jump = bool(jump_indicator.iloc[i] == 1) if jump_indicator is not None else False 

            # Entry 
            if position == 0:
                normal_long = z_t < -z_entry_adj 
                normal_short = z_t > z_entry_adj 

                jump_long = jump_short = False
                if self.use_jump_entries and is_jump: 
                    jump_threshold = z_entry_adj * 0.65
                    jump_long = z_t < -jump_threshold 
                    jump_short = z_t > jump_threshold 

                want_long = normal_long or jump_long 
                want_short = normal_short or jump_short 

                if want_long or want_short:
                    # Both entry paths pass through BOTH gates. 
                    if regime == HawkesRegime.CRISIS:
                        self.entries_blocked_by_regime += 1
                    elif not self._is_lambda_decaying(lambda_intensity, i):
                        self.entries_blocked_by_decay += 1
                    else:
                        position = 1 if want_long else -1 
                        signals[i] = (
                            SignalType.LONG.value if want_long else SignalType.SHORT.value
                        )
                        entry_idx, entry_z, entry_regime = i, z_t, regime 
                        current_size = self._position_size(z_t, lambda_t, regime)

                        # jump assisted entries are sized down 
                        if (jump_long or jump_short) and not (normal_long or normal_short):
                            current_size *= 0.8
                            self.jump_entries_taken += 1

            # Exit
            elif position != 0:
                held = i - entry_idx 
                _, z_exit_current, max_hold_current = self._get_regime_thresholds(regime)

                triggered: List[str] = []

                # Every condition is evaluated; none can mask another
                if abs(z_t) < z_exit_current and held >= self.min_hold:
                    triggered.append("mean_reversion")

                if held >= self.min_hold:
                    if position == 1 and z_t > self.z_exit:
                        triggered.append("profit_target")
                    elif position == -1 and z_t < -self.z_exit:
                        triggered.append("profit_target")

                if held >= max_hold_current:
                    triggered.append("max_hold")

                if regime == HawkesRegime.CRISIS and entry_regime != HawkesRegime.CRISIS:
                    if held >= max(self.min_hold // 2, 1):
                        triggered.append("regime_crisis")

                if position == 1 and z_t < entry_z - self.emergency_z_move:
                    triggered.append("emergency_stop")
                elif position == -1 and z_t > entry_z + self.emergency_z_move:
                    triggered.append("emergency_stop")

                if triggered:
                    reason = next(r for r in EXIT_PRIORITY if r in triggered)
                    signals[i] = SignalType.CLOSE.value
                    exit_reasons[i] = reason 
                    self.exit_reason_counts[reason] = (
                        self.exit_reason_counts.get(reason, 0) + 1
                    )
                    position = 0
                    current_size = 0.0

            positions[i] = position
            sizes[i] = current_size if position != 0 else 0.0 

        signals_df = pd.DataFrame(
            {
                "signal": signals,
                "position": positions,
                "position_size": sizes,
                "z_score": z_score,
                "lambda": lambda_intensity,
                "spread": spread, 
                "regime": regimes, 
                "signal_exit_reason": exit_reasons,
            },
            index = index,
        )
        if jump_indicator is not None:
            signals_df["jump"] = jump_indicator

        n_long = int((signals == SignalType.LONG.value).sum())
        n_short = int((signals == SignalType.SHORT.value).sum())
        n_close = int((signals == SignalType.CLOSE.value).sum())

        self._log(
            f" {n_long} long, {n_short} short, {n_close} close "
            f"({self.jump_entries_taken} jump-assisted)"
        )
        self._log(
            f" blocked: {self.entries_blocked_by_regime} by CRISIS regime, "
            f"{self.entries_blocked_by_decay} by lambda decay"
        )
        if self.exit_reason_counts:
            self._log(f" signal exit reasons: {self.exit_reason_counts}")

        regime_counts = pd.Series(regimes).value_counts().to_dict()
        self._log(f" regime days: {regime_counts}")

        return signals_df 

    def calculate_signal_quality(self, signals_df: pd.DataFrame) -> Dict:
        """ Descriptive statistics about how often the strategy is active"""
        positions = signals_df["position"]
        entries = int((signals_df["signal"].abs() == 1).sum())
        n = len(signals_df)

        return {
            "time_in_market": float((positions != 0).mean()),
            "n_entries": entries,
            "entry_discipline": float(entries / n) if n else 0.0,
            "avg_position_size": float(
                signals_df.loc[positions != 0, "position_size"].mean()
            )
            if (positions != 0).any()
            else 0.0,
            "entries_blocked_by_regime": self.entries_blocked_by_regime,
            "entries_blocked_by_decay": self.entries_blocked_by_decay,
            "jump_entries_taken": self.jump_entries_taken,
            "exit_reason_counts": dict(self.exit_reason_counts),
        }
    
