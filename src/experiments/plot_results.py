from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_rmse_vs_horizon(best_df: pd.DataFrame, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    fig_path = out_dir / "rmse_vs_horizon.png"

    d = best_df.copy()
    d["horizon"] = d["horizon"].astype(int)
    d = d.sort_values(["target", "horizon"], kind="mergesort")

    for target, g in d.groupby("target", sort=False):
        plt.plot(g["horizon"], g["best_overall_rmse"], marker="o", label=target)

    plt.xlabel("Horizon (days)")
    plt.ylabel("RMSE (overall)")
    plt.title("Best Overall RMSE vs Horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path


def plot_rmse_hv_fut_vs_horizon(best_df: pd.DataFrame, out_dir: Path) -> Path | None:
    if "best_hv_fut_rmse" not in best_df.columns:
        return None

    ensure_dir(out_dir)
    fig_path = out_dir / "rmse_hv_fut_vs_horizon.png"

    d = best_df.copy()
    d["horizon"] = d["horizon"].astype(int)
    d = d.sort_values(["target", "horizon"], kind="mergesort")

    d = d.dropna(subset=["best_hv_fut_rmse"])
    if d.empty:
        return None

    for target, g in d.groupby("target", sort=False):
        plt.plot(g["horizon"], g["best_hv_fut_rmse"], marker="o", label=target)

    plt.xlabel("Horizon (days)")
    plt.ylabel("RMSE (future spike subset)")
    plt.title("Best hv_fut RMSE vs Horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path


def main() -> None:
    results_dir = Path("artifacts") / "results"
    figures_dir = Path("artifacts") / "figures"

    best_path = results_dir / "best_models_latest.csv"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing {best_path}. Run experiments first to generate it.")

    best_df = pd.read_csv(best_path)

    p1 = plot_rmse_vs_horizon(best_df, figures_dir)
    print(f"Saved: {p1}")

    p2 = plot_rmse_hv_fut_vs_horizon(best_df, figures_dir)
    if p2 is not None:
        print(f"Saved: {p2}")
    else:
        print("Skipped hv_fut plot (no best_hv_fut_rmse column or all NaN).")


if __name__ == "__main__":
    main()
