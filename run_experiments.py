from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.data.pipeline import build_and_save_processed_dataset

from src.eval.evaluate import evaluate_model
from src.eval.evaluate_regime import evaluate_hmm_regime_ridge, RegimeEvalConfig
from src.models.baselines import ZeroReturnBaseline, RollingMeanBaseline, RidgeBaseline
from src.models.regime_conditioned import RegimeConditionedRidge

from src.experiments.logging import utc_run_id, pack_params, save_results_csv


def _pretty(metrics: dict) -> dict:
    keys = [
        "rmse",
        "mae",
        "spearman",
        "top_decile_hit",
        "rmse_hv_now",
        "mae_hv_now",
        "rmse_hv_fut",
        "mae_hv_fut",
        "directional_accuracy",
    ]
    return {k: metrics.get(k) for k in keys if k in metrics}


def main() -> None:
    cfg = load_config()
    ds = build_and_save_processed_dataset(cfg)
    data = ds.df

    features = [c for c in data.columns if not c.startswith("y_")]

    run_id = utc_run_id("results")
    out_dir = Path("artifacts") / "results"

    rows: list[dict] = []

    for h in cfg.targets.horizons:
        target = f"y_{cfg.targets.target_type}_h{h}"
        print(f"\n=== Target: {target} ===")

        # --- Baselines ---
        print("\n=== Baselines ===")
        baseline_models = [
            ZeroReturnBaseline(),
            RollingMeanBaseline(),
            RidgeBaseline(alpha=1.0),
        ]

        for m in baseline_models:
            metrics = evaluate_model(
                df=data,
                features=features,
                target=target,
                model=m,
                train_years=cfg.evaluation.train_years,
                test_years=cfg.evaluation.test_years,
                step_years=cfg.evaluation.step_years,
            )

            print(m.name, _pretty(metrics))

            rows.append(
                {
                    "run_id": run_id,
                    "target": target,
                    "horizon": int(h),
                    "target_type": cfg.targets.target_type,
                    "model": m.name,
                    "family": "baseline",
                    "params": pack_params(getattr(m, "params", None)),
                    **metrics,
                }
            )

        # --- Regime-conditioned ---
        print("\n=== HMM Regime-Conditioned Ridge ===")
        for K in cfg.hmm.K_values:
            regime_cfg = RegimeEvalConfig(
                K=int(K),
                hmm_covariance_type=cfg.hmm.covariance_type,
                hmm_n_iter=cfg.hmm.n_iter,
                hmm_tol=cfg.hmm.tol,
                hmm_min_covar=cfg.hmm.min_covar,
                seed=cfg.project.seed,
            )

            for mode in ["hard", "soft"]:
                rc = RegimeConditionedRidge(alpha=1.0, mode=mode, min_points_per_regime=200)

                metrics = evaluate_hmm_regime_ridge(
                    df=data,
                    features=features,
                    target=target,
                    model=rc,
                    train_years=cfg.evaluation.train_years,
                    test_years=cfg.evaluation.test_years,
                    step_years=cfg.evaluation.step_years,
                    regime_cfg=regime_cfg,
                )

                name = f"hmm_ridge_{mode}_K{K}"
                print(name, _pretty(metrics))

                rows.append(
                    {
                        "run_id": run_id,
                        "target": target,
                        "horizon": int(h),
                        "target_type": cfg.targets.target_type,
                        "model": name,
                        "family": "hmm_ridge",
                        "params": pack_params(
                            {
                                "K": int(K),
                                "mode": mode,
                                "alpha": 1.0,
                                "min_points_per_regime": 200,
                                "hmm_covariance_type": cfg.hmm.covariance_type,
                                "hmm_n_iter": cfg.hmm.n_iter,
                                "hmm_tol": cfg.hmm.tol,
                                "hmm_min_covar": cfg.hmm.min_covar,
                                "seed": cfg.project.seed,
                            }
                        ),
                        **metrics,
                    }
                )

    out_path = save_results_csv(rows, out_dir=out_dir, run_id=run_id)
    print(f"\nSaved results to: {out_path}")

    try:
        config_src = Path("config.yaml")
        if config_src.exists():
            (out_dir / "latest_config.yaml").write_text(config_src.read_text(), encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    main()
