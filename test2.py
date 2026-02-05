from pathlib import Path
from src.experiments.per_regime_compare import PerRegimeCompareConfig, compute_per_regime_comparison

run_id = "results_2026-02-05T14-14-06Z"

processed_parquet = Path("data/processed/SPY_absret_h1-5-20_features.parquet")

winners = [
    ("y_absret_h1",  "hmm_ridge_soft_K3_normal"),
    ("y_absret_h20", "hmm_ridge_soft_K2_normal"),
    ("y_absret_h5",  "hmm_ridge_soft_K2_normal"),
]

for target, model in winners:
    base = Path("artifacts") / "regimes" / run_id / target / model
    cfg = PerRegimeCompareConfig(
        processed_parquet=processed_parquet,
        oos_probs_csv=base / "oos_regime_probs.csv",
        hmm_oos_predictions_csv=base / "oos_predictions.csv",
        target=target,
        out_compare_csv=base / "per_regime_compare_ridge_vs_hmm.csv",
        out_pubready_csv=base / "per_regime_pubready_table.csv",
        out_latex=base / "per_regime_pubready_table.tex",
        out_plot_path=base / "plots" / "hv_fut_concentration_by_regime.png",
        hv_fut_q=0.9,
    )
    outp = compute_per_regime_comparison(cfg)
    print(f"[ok] {target} | {model} -> {outp}")
