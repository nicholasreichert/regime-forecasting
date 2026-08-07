"""Table and figure for the simulation study.

The point of the table is the contrast between columns: the descriptive
diagnostics look much the same whether or not the data contain regimes, while
the forecasting column separates cleanly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path("artifacts") / "study" / "simulation" / "simulation_results.csv"
TABLES = Path("paper") / "tables"
FIGURES = Path("paper") / "figures"
HORIZONS = [1, 5, 20]

LABEL = {
    "ms_slope": r"Markov switching (differing dynamics)",
    "ms_level": r"Markov switching (differing level)",
    "garch11": r"GARCH(1,1)-$t$",
    "iid_t": r"i.i.d.\ Student-$t$",
    "iid_normal": r"i.i.d.\ Gaussian",
    "real": r"\textbf{SPY (real data)}",
}
ORDER = ["ms_slope", "ms_level", "garch11", "iid_t", "iid_normal", "real"]


def real_data_row() -> dict:
    """Run the same diagnostics on SPY so the empirical case sits in the table."""
    from src.config import load_config
    from src.data.pipeline import build_and_save_processed_dataset
    from src.experiments.simulation_study import hmm_diagnostics

    cfg = load_config()
    data = build_and_save_processed_dataset(cfg).df
    diag = hmm_diagnostics(data, cfg, seed=cfg.project.seed)

    # forecasting improvement is already measured in the cross-asset study
    ma = Path("artifacts") / "study" / "multi_asset" / "multi_asset_results.csv"
    imp = {}
    if ma.exists():
        d = pd.read_csv(ma)
        d = d[(d["ticker"] == "SPY") & (d["model"] == "regime_normal")]
        for h in HORIZONS:
            s = d[d["horizon"] == h]["improvement_pct"]
            imp[h] = float(s.iloc[0]) if len(s) else np.nan
    return {"dgp": "real", "has_true_regimes": None, **diag,
            **{f"imp_h{h}": imp.get(h, np.nan) for h in HORIZONS}}


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.get("error").isna()] if "error" in df.columns else df
    rows = []
    for dgp, g in df.groupby("dgp"):
        row = {
            "dgp": dgp,
            "has_true_regimes": bool(g["has_true_regimes"].iloc[0]),
            "sep_ratio": g["sep_ratio"].mean(),
            "excess_persistence": g["excess_persistence"].mean(),
            "excess_persistence_no_vol": g["excess_persistence_no_vol"].mean(),
            "state_recovery_ari": g["state_recovery_ari"].mean(),
            "mean_self_transition": g["mean_self_transition"].mean(),
            "redundancy_acc": g["redundancy_acc"].mean(),
            "redundancy_base_rate": g["redundancy_base_rate"].mean(),
        }
        for h in HORIZONS:
            row[f"imp_h{h}"] = g[g["horizon"] == h]["improvement_pct"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def make_table(summary: pd.DataFrame) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{What the HMM learns from each world. All rows use the identical"
        r" pipeline. ``Separation'' is the ratio of the highest to the lowest"
        r" state's mean realized volatility; ``excess persistence'' is the average"
        r" self-transition probability minus the stationary probability, so zero"
        r" means states are temporally independent. Persistence is reported with"
        r" the standard emission vector and again with the 20-day rolling"
        r" volatility removed; on data with no dynamics at all the first is large"
        r" and the second is zero, so most of what the usual diagnostic reports is"
        r" autocorrelation contributed by an overlapping-window feature."
        r" ``State recovery'' is the adjusted Rand index against the true latent"
        r" path where one exists. The final columns are the out-of-sample RMSE"
        r" improvement from conditioning on the inferred state. Separation and"
        r" persistence do not distinguish the worlds containing regimes from those"
        r" that do not; forecasting does.}",
        r"\label{tab:simulation}",
        r"\small",
        r"\begin{tabular}{lccccrrrr}",
        r"\toprule",
        r"& True & & \multicolumn{2}{c}{Excess persistence} & State &"
        r" \multicolumn{3}{c}{Forecast improvement} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){7-9}",
        r"Data-generating process & regimes? & Separation & std.\ & no RV$_{20}$"
        r" & recovery & $h{=}1$ & $h{=}5$ & $h{=}20$ \\",
        r"\midrule",
    ]
    for dgp in ORDER:
        s = summary[summary["dgp"] == dgp]
        if s.empty:
            continue
        r = s.iloc[0]
        tr = r["has_true_regimes"]
        tr_s = "--" if tr is None or (isinstance(tr, float) and np.isnan(tr)) else (
            r"\checkmark" if tr else r"$\times$")
        if dgp == "real":
            lines.append(r"\midrule")
        ari = r.get("state_recovery_ari", np.nan)
        cells = [
            LABEL.get(dgp, dgp), tr_s,
            f"{r['sep_ratio']:.2f}",
            f"{r['excess_persistence']:+.3f}",
            f"{r['excess_persistence_no_vol']:+.3f}",
            "--" if not np.isfinite(ari) else f"{ari:.2f}",
        ]
        for h in HORIZONS:
            v = r.get(f"imp_h{h}", np.nan)
            cells.append("--" if not np.isfinite(v) else f"{v:+.2f}\\%")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "simulation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {TABLES / 'simulation.tex'}")


def make_figure(summary: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    dgps = [d for d in ORDER if d in set(summary["dgp"])]
    xs = np.arange(len(dgps))
    colour = ["#54A24B" if summary[summary["dgp"] == d]["has_true_regimes"].iloc[0] is True
              else ("#8C8C8C" if d == "real" else "#E45756") for d in dgps]
    short = {"ms_slope": "MS\n(dynamics)", "ms_level": "MS\n(level)", "garch11": "GARCH",
             "iid_t": "iid $t$", "iid_normal": "iid $N$", "real": "SPY"}

    # a diagnostic that does not discriminate
    w = 0.38
    ax1.bar(xs - w / 2, [summary[summary["dgp"] == d]["excess_persistence"].iloc[0] for d in dgps],
            w, color="#4C78A8", label="standard emissions")
    ax1.bar(xs + w / 2,
            [summary[summary["dgp"] == d]["excess_persistence_no_vol"].iloc[0] for d in dgps],
            w, color="#F58518", label="without RV$_{20}$")
    ax1.axhline(0, color="black", lw=1)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([short.get(d, d) for d in dgps], fontsize=8)
    ax1.set_ylabel("Excess persistence")
    ax1.set_title("Persistence is mostly a feature artefact", fontsize=9.5)
    ax1.legend(frameon=False, fontsize=7.5)

    # the one that does
    width = 0.27
    for i, h in enumerate(HORIZONS):
        vals = [summary[summary["dgp"] == d][f"imp_h{h}"].iloc[0] for d in dgps]
        ax2.bar(xs + (i - 1) * width, vals, width, label=f"$h={h}$")
    ax2.axhline(0, color="black", lw=1)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([short.get(d, d) for d in dgps], fontsize=8)
    ax2.set_ylabel("Forecast improvement (%)")
    ax2.set_title("What forecasting sees", fontsize=9.5)
    ax2.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "simulation.png", dpi=220)
    plt.close(fig)
    print(f"wrote {FIGURES / 'simulation.png'}")


def main() -> None:
    df = pd.read_csv(RESULTS)
    summary = build_summary(df)
    try:
        summary = pd.concat([summary, pd.DataFrame([real_data_row()])], ignore_index=True)
    except Exception as e:
        print(f"[warn] could not add real-data row: {type(e).__name__}: {e}")
    summary.to_csv(RESULTS.parent / "simulation_summary.csv", index=False)
    print(summary.to_string(index=False))
    make_table(summary)
    make_figure(summary)


if __name__ == "__main__":
    main()
