from src.config import load_config
from src.data.pipeline import build_and_save_processed_dataset

from src.eval.evaluate import evaluate_model
from src.models.baselines import ZeroReturnBaseline, RollingMeanBaseline, RidgeBaseline

from src.eval.evaluate_regime import evaluate_hmm_regime_ridge, RegimeEvalConfig
from src.models.regime_conditioned import RegimeConditionedRidge


def main() -> None:
    cfg = load_config()
    ds = build_and_save_processed_dataset(cfg)
    df = ds.df

    for h in cfg.targets.horizons:
        target = f"y_{cfg.targets.target_type}_h{h}"
        print(f"\n=== Target: {target} ===")

        # Features: exclude targets, and keep all engineered predictors
        features = [c for c in df.columns if not c.startswith("y_")]

        print("\n=== Baselines ===")
        for m in [ZeroReturnBaseline(), RollingMeanBaseline(), RidgeBaseline(alpha=1.0)]:
            metrics = evaluate_model(
                df=df,
                features=features,
                target=target,
                model=m,
                train_years=cfg.evaluation.train_years,
                test_years=cfg.evaluation.test_years,
                step_years=cfg.evaluation.step_years,
            )
            print(m.name, metrics)

        print("\n=== HMM Regime-Conditioned Ridge ===")
        for K in cfg.hmm.K_values:
            regime_cfg = RegimeEvalConfig(
                K=int(K),
                hmm_covariance_type=cfg.hmm.covariance_type,
                hmm_n_iter=cfg.hmm.n_iter,
                hmm_tol=cfg.hmm.tol,
                seed=cfg.project.seed,
            )

            for mode in ["hard", "soft"]:
                rc = RegimeConditionedRidge(alpha=1.0, mode=mode, min_points_per_regime=200)
                metrics = evaluate_hmm_regime_ridge(
                    df=df,
                    features=features,
                    target=target,
                    model=rc,
                    train_years=cfg.evaluation.train_years,
                    test_years=cfg.evaluation.test_years,
                    step_years=cfg.evaluation.step_years,
                    regime_cfg=regime_cfg,
                )
                print(f"hmm_ridge_{mode}_K{K}", metrics)


if __name__ == "__main__":
    main()
