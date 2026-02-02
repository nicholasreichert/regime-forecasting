from src.config import load_config
from src.data.pipeline import build_and_save_processed_dataset
from src.eval.evaluate import evaluate_model
from src.models.baselines import (
    ZeroReturnBaseline,
    RollingMeanBaseline,
    RidgeBaseline,
)


def main():
    cfg = load_config()
    ds = build_and_save_processed_dataset(cfg)

    df = ds.df
    target = "y_ret_h1"

    features = [
        c for c in df.columns
        if c.startswith("ret_")
        and not c.startswith("y_")
    ]

    models = [
        ZeroReturnBaseline(),
        RollingMeanBaseline(),
        RidgeBaseline(alpha=1.0),
    ]

    for model in models:
        metrics = evaluate_model(
            df=df,
            features=features,
            target=target,
            model=model,
            train_years=cfg.evaluation.train_years,
            test_years=cfg.evaluation.test_years,
            step_years=cfg.evaluation.step_years,
        )
        print(model.name, metrics)


if __name__ == "__main__":
    main()
