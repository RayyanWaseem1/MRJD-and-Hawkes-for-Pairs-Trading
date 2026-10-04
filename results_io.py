"""
Artifact writing -- the code that produces everything in `outputs/`.

Why this module exists
----------------------
In the audited repository, `grep` across every `.py` file for
`train_trading_signals`, `model_bundle`, `causal_artifacts`,
`walk_forward_metrics`, `quarterly_metrics`, `train_val_equity`, `spread.csv`,
`quarterly_model_bundles` or `z_score.png` returned ZERO hits. All 15 CSVs and
3 of the 5 PNGs per pair per split were produced by scripts that were never
committed. `main.py`'s `save_results()` wrote a different, smaller file set and
was never called by `run_train_val_pipeline()`.

The README meanwhile stated that "the saved result CSVs in `outputs/` are the
source of truth for the tables below" -- so the source of truth for every
number in the README was unreproducible from the repository. That is the
fastest thing an interviewer can check.

Every artifact is now written from here, and both pipelines call it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

__all__ = [
    "ResultsWriter",
    "ensure_dir",
    "flatten_for_csv",
]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def flatten_for_csv(payload: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dict into scalar columns suitable for a one-row CSV."""
    flat: Dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_for_csv(value, prefix=f"{name}_"))
        elif isinstance(value, (list, tuple)):
            if len(value) <= 4 and all(
                isinstance(v, (int, float, np.integer, np.floating)) for v in value
            ):
                for i, v in enumerate(value):
                    flat[f"{name}_{i}"] = float(v)
            else:
                flat[name] = json.dumps(value, default=str)
        elif isinstance(value, (np.integer, np.floating)):
            flat[name] = float(value)
        elif isinstance(value, (str, int, float, bool, type(None))):
            flat[name] = value
        else:
            flat[name] = str(value)
    return flat


class ResultsWriter:
    """Write every artifact for one pair / one split."""

    def __init__(self, output_dir: str | Path, verbose: bool = True):
        self.dir = ensure_dir(output_dir)
        self.verbose = verbose
        self.written: List[str] = []

    def _log(self, *args) -> None:
        if self.verbose:
            print(*args)

    def _record(self, name: str) -> Path:
        self.written.append(name)
        return self.dir / name

    # ------------------------------------------------------------------ #
    # tabular artifacts
    # ------------------------------------------------------------------ #

    def write_frame(self, frame: pd.DataFrame, name: str, index: bool = True) -> None:
        """
        Write a table. A table with columns but no rows (e.g. a run with zero
        trades) is written header-only, so it can never be mistaken for the
        previous run's file. A frame with no columns at all is skipped, and
        any stale file of the same name is removed.
        """
        if frame is None or (len(frame) == 0 and len(frame.columns) == 0):
            stale = self.dir / name
            if stale.exists():
                stale.unlink()
            self._log(f"    (skipped empty {name})")
            return
        frame.to_csv(self._record(name), index=index)

    def write_row(self, payload: Mapping[str, Any], name: str) -> None:
        """One-row CSV from a (possibly nested) dict."""
        pd.DataFrame([flatten_for_csv(payload)]).to_csv(self._record(name), index=False)

    def write_json(self, payload: Dict, name: str) -> None:
        with open(self._record(name), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

    # ------------------------------------------------------------------ #
    # composed artifacts
    # ------------------------------------------------------------------ #

    def write_spread(self, spread_df: pd.DataFrame) -> None:
        self.write_frame(spread_df, "spread.csv")

    def write_causal_artifacts(
        self,
        jump_df: pd.DataFrame,
        intensity: pd.Series,
        z_score: pd.Series,
    ) -> None:
        """The frozen-parameter artifacts used by every evaluation window."""
        frame = pd.DataFrame(
            {
                "jump_indicator": jump_df["jump_indicator"],
                "hawkes_intensity": intensity,
                "z_score": z_score,
            }
        )
        for extra in ("L_statistic", "p_value", "z_statistic", "returns"):
            if extra in jump_df.columns:
                frame[extra] = jump_df[extra]
        self.write_frame(frame, "causal_artifacts.csv")

    def write_model_bundle(self, bundle) -> None:
        """Flatten a ModelBundle into a one-row CSV."""
        payload = {
            "half_life": bundle.half_life,
            "hedge_ratio": bundle.hedge_ratio,
            "use_hawkes_regimes": bundle.use_hawkes_regimes,
            "hawkes_active": bundle.hawkes_active,
            "hawkes_inactive_reason": bundle.hawkes_inactive_reason,
            "detection_basis": bundle.detection_basis,
            "n_jumps_fdr": bundle.n_jumps_fdr,
            "n_jumps_nominal": bundle.n_jumps_nominal,
        }
        for key, value in (bundle.hawkes_params or {}).items():
            payload[f"hawkes_{key}"] = value
        for key, value in (bundle.mrjd_params or {}).items():
            payload[f"mrjd_{key}"] = value
        for key, value in (bundle.hedge_diagnostics or {}).items():
            payload[f"hedge_{key}"] = value
        for key, value in (bundle.spread_stats or {}).items():
            payload[f"spread_{key}"] = value
        self.write_row(payload, "model_bundle.csv")

    def write_hawkes_inference(self, inference: Dict) -> None:
        """LR test, standard errors, goodness of fit, intensity calibration."""
        self.write_row(inference, "hawkes_inference.csv")
        self.write_json(inference, "hawkes_inference.json")

    def write_jump_comparison(self, comparison: pd.DataFrame) -> None:
        self.write_frame(comparison, "jump_detector_comparison.csv", index=False)

    def write_evaluation(
        self,
        label: str,
        equity_curve: pd.DataFrame,
        signals_df: pd.DataFrame,
        trade_summary: pd.DataFrame,
        metrics: Dict,
    ) -> None:
        """Write one evaluation window's equity curve, signals, trades, metrics."""
        prefix = label.lower().replace(" ", "_").replace("/", "_")
        self.write_frame(equity_curve, f"{prefix}_equity_curve.csv")
        self.write_frame(signals_df, f"{prefix}_trading_signals.csv")
        self.write_frame(trade_summary, f"{prefix}_trade_summary.csv", index=False)
        self.write_row(metrics, f"{prefix}_performance_metrics.csv")

    def write_comparison(self, rows: List[Dict], name: str) -> None:
        if not rows:
            return
        pd.DataFrame([flatten_for_csv(r) for r in rows]).to_csv(
            self._record(name), index=False
        )

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #

    def _figure_path(self, name: str) -> str:
        """Return a Matplotlib-compatible output path while recording the artifact."""
        return str(self._record(name))

    def plot_spread_and_jumps(
        self, spread_df: pd.DataFrame, jump_df: pd.DataFrame, title: str, dpi: int = 150
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(spread_df.index, spread_df["spread"], lw=0.9, color="#1f5f63")
        jumps = jump_df.index[jump_df["jump_indicator"] == 1]
        overlap = spread_df.index.intersection(jumps)
        if len(overlap):
            axes[0].scatter(
                overlap, spread_df.loc[overlap, "spread"],
                color="#a63232", s=42, marker="x", zorder=5,
                label=f"jumps (n={len(overlap)})",
            )
            axes[0].legend(loc="upper left")
        axes[0].set_ylabel("log spread")
        axes[0].set_title(title, fontweight="bold")
        axes[0].grid(alpha=0.25)

        stat_col = "L_statistic" if "L_statistic" in jump_df.columns else "z_statistic"
        if stat_col in jump_df.columns:
            axes[1].plot(jump_df.index, jump_df[stat_col], lw=0.7, color="#b0741c")
            if "threshold" in jump_df.columns:
                axes[1].axhline(
                    float(pd.Series(jump_df["threshold"]).iloc[0]),
                    color="#a63232", ls="--", lw=1, label="critical value",
                )
                axes[1].legend(loc="upper left")
            axes[1].set_ylabel(stat_col)
        axes[1].set_xlabel("date")
        axes[1].grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(self._figure_path("jump_detection.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_intensity(
        self, intensity: pd.Series, lambda_bar: float, title: str, dpi: int = 150
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(intensity.index, intensity.to_numpy(), lw=0.9, color="#1f5f63")
        ax.axhline(lambda_bar, color="#a63232", ls="--", lw=1,
                   label=f"baseline lambda_bar = {lambda_bar:.5f}")
        ax.set_ylabel("lambda(t)")
        ax.set_xlabel("date")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(self._figure_path("hawkes_intensity.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_zscore(self, z_score: pd.Series, entry: float, exit_: float,
                    title: str, dpi: int = 150) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(z_score.index, z_score.to_numpy(), lw=0.8, color="#1f5f63")
        for level, style, colour in (
            (entry, "--", "#a63232"), (-entry, "--", "#a63232"),
            (exit_, ":", "#b0741c"), (-exit_, ":", "#b0741c"),
        ):
            ax.axhline(level, ls=style, lw=1, color=colour)
        ax.axhline(0, lw=0.8, color="#5a6472")
        ax.set_ylabel("z-score")
        ax.set_xlabel("date")
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(self._figure_path("z_score.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_equity(self, curves: Dict[str, pd.DataFrame], title: str,
                    filename: str = "equity.png", dpi: int = 150) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 6))
        palette = ["#1f5f63", "#a63232", "#b0741c", "#6b5c7a"]
        for i, (label, curve) in enumerate(curves.items()):
            if curve is None or curve.empty:
                continue
            normalised = curve["equity"] / float(curve["equity"].iloc[0])
            ax.plot(curve.index, normalised, lw=1.2,
                    color=palette[i % len(palette)], label=label)
        ax.axhline(1.0, color="#5a6472", lw=0.8, ls="--")
        ax.set_ylabel("equity (normalised)")
        ax.set_xlabel("date")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(self._figure_path(filename), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_qq_residuals(self, gof: Dict, title: str, dpi: int = 150) -> None:
        """QQ plot of compensator residuals against Exp(1)."""
        if "theoretical_quantiles" not in gof:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        theo = np.asarray(gof["theoretical_quantiles"], dtype=float)
        emp = np.asarray(gof["empirical_quantiles"], dtype=float)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(theo, emp, s=22, alpha=0.7, color="#1f5f63")
        lim = [float(min(theo.min(), emp.min())), float(max(theo.max(), emp.max()))]
        ax.plot(lim, lim, ls="--", lw=1.2, color="#a63232", label="perfect fit")
        ax.set_xlabel("theoretical quantiles  Exp(1)")
        ax.set_ylabel("empirical quantiles")
        ax.set_title(
            f"{title}\nKS p = {gof.get('ks_pvalue', float('nan')):.4f}",
            fontweight="bold",
        )
        ax.legend(loc="upper left")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(self._figure_path("hawkes_qq_residuals.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------ #

    def manifest(self) -> None:
        """Record exactly which files this run produced."""
        # Record the manifest itself too: a manifest that omits its own filename
        # is needlessly awkward for an artifact-integrity check.
        if "MANIFEST.json" not in self.written:
            self.written.append("MANIFEST.json")
        payload = {"files": sorted(self.written), "n_files": len(self.written)}
        with open(self.dir / "MANIFEST.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        self._log(f"    wrote {len(self.written)} artifacts to {self.dir}")
