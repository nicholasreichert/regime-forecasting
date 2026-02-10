from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_hvfut_rmse_by_regime(
    per_regime_compare_csv: Path,
    out_path: Path,
    title: str | None = None,
) -> None:
    df = pd.read_csv(per_regime_compare_csv)

    required = {"regime", "regime_label", "ridge_rmse_hv_fut", "hmm_rmse_hv_fut"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {per_regime_compare_csv}: {sorted(missing)}")

    df = df.sort_values("regime").reset_index(drop=True)

    labels = df["regime_label"].astype(str).tolist()
    ridge = df["ridge_rmse_hv_fut"].to_numpy(dtype=float)
    hmm = df["hmm_rmse_hv_fut"].to_numpy(dtype=float)

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    ax.bar(x - width / 2, ridge, width, label="Ridge (baseline)")
    ax.bar(x + width / 2, hmm, width, label="HMM-conditioned Ridge")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Regime (ordered by stress proxy)")
    ax.set_ylabel("RMSE on hv_fut subset")

    if title:
        ax.set_title(title)

    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="Experiment run_id (e.g. results_2026-02-05T14-14-06Z)")
    ap.add_argument("--target", required=True, help="Target name (e.g. y_absret_h1)")
    ap.add_argument(
        "--use",
        choices=["best_overall_model", "best_hv_fut_model"],
        default="best_overall_model",
        help="Which winner column to use from best_models.csv",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output figure path (e.g. artifacts/figures/hvfut_rmse_by_regime_h1.png)",
    )
    args = ap.parse_args()

    best_csv = Path("artifacts") / "results" / f"{args.run_id}_best_models.csv"
    if not best_csv.exists():
        raise FileNotFoundError(f"Could not find {best_csv}")

    best_df = pd.read_csv(best_csv)
    row = best_df.loc[best_df["target"] == args.target]

    if row.empty:
        raise ValueError(f"Target {args.target} not found in {best_csv}")

    model_name = row.iloc[0][args.use]
    if not isinstance(model_name, str) or not model_name.startswith("hmm"):
        raise ValueError(
            f"Selected winner '{model_name}' is not an HMM regime model; "
            f"per-regime comparison requires an HMM winner."
        )

    regime_dir = (
        Path("artifacts")
        / "regimes"
        / args.run_id
        / args.target
        / model_name
    )

    per_regime_csv = regime_dir / "per_regime_compare_ridge_vs_hmm.csv"
    if not per_regime_csv.exists():
        raise FileNotFoundError(f"Missing per-regime comparison CSV: {per_regime_csv}")

    title = f"hv_fut RMSE by regime ({args.target})" 

    plot_hvfut_rmse_by_regime(
        per_regime_compare_csv=per_regime_csv,
        out_path=Path(args.out),
        title=title,
    )

    print(f"[ok] wrote figure to {args.out}")
    print(f"[info] model = {model_name}")
    print(f"[info] source = {per_regime_csv}")


if __name__ == "__main__":
    main()
