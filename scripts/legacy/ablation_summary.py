import pandas as pd

df = pd.read_csv("artifacts/results/latest.csv")
targets = ["y_absret_h1", "y_absret_h5", "y_absret_h20"]

df = df[df["target"].isin(targets)]

baseline = (
    df[df["model"] == "ridge"]
    .set_index(["target", "horizon"])
)

def best_by(group, metric):
    return group.loc[group[metric].idxmin()]

normal = (
    df[df["model"].str.endswith("_normal")]
    .groupby(["target", "horizon"])
    .apply(best_by, metric="rmse_hv_fut")
)
shuffle = (
    df[df["model"].str.endswith("_shuffle")]
    .groupby(["target", "horizon"])
    .apply(best_by, metric="rmse_hv_fut")
)

summary = pd.DataFrame({
    "ridge_rmse_hv_fut": baseline["rmse_hv_fut"],
    "normal_rmse_hv_fut": normal["rmse_hv_fut"],
    "shuffle_rmse_hv_fut": shuffle["rmse_hv_fut"],
})

summary["delta_normal_vs_ridge"] = (
    summary["normal_rmse_hv_fut"] - summary["ridge_rmse_hv_fut"]
)

summary["delta_shuffle_vs_ridge"] = (
    summary["shuffle_rmse_hv_fut"] - summary["ridge_rmse_hv_fut"]
)

summary["delta_normal_vs_shuffle"] = (
    summary["normal_rmse_hv_fut"] - summary["shuffle_rmse_hv_fut"]
)

summary.round(6).to_csv(
    "artifacts/results/ablation_summary.csv"
)
