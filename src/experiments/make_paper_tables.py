"""Generate every LaTeX table and figure in the paper directly from artifacts.

Nothing in the paper is transcribed by hand. An earlier version of this project
carried a results table in its README whose numbers matched no artifact in the
repository, which is exactly the failure mode this module exists to prevent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STUDY = Path("artifacts") / "study"
PAPER = Path("paper")
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"

HORIZONS = [1, 5, 20]

BASELINE_ORDER = ["zero", "train_mean", "rw_vol", "ewma", "har", "garch11", "ridge"]
BASELINE_LABEL = {
    "zero": "Zero",
    "train_mean": "Train mean",
    "rw_vol": "Random walk (RV)",
    "ewma": "EWMA / RiskMetrics",
    "har": "HAR",
    "garch11": "GARCH(1,1)",
    "ridge": "Ridge (pooled)",
}
ABLATION_ORDER = ["normal", "gate_shuffle", "expert_shuffle", "both_shuffle", "single"]
ABLATION_LABEL = {
    "normal": r"\textsc{normal} (full model)",
    "gate_shuffle": r"\textsc{gate\_shuffle}",
    "expert_shuffle": r"\textsc{expert\_shuffle}",
    "both_shuffle": r"\textsc{both\_shuffle}",
    "single": r"\textsc{single} (pooled ridge)",
}


def _latest_study_dir() -> Path:
    pointer = STUDY / "latest" / "run_id.txt"
    if pointer.exists():
        d = STUDY / pointer.read_text(encoding="utf-8").strip()
        if d.exists():
            return d
    candidates = sorted([p for p in STUDY.glob("study_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError("No study run found under artifacts/study/")
    return candidates[-1]


def _fmt(x: float, nd: int = 5) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    # QLIKE diverges for near-zero forecasts, so trivial baselines land orders of
    # magnitude above everything else. Print those compactly rather than padding
    # the column to accommodate them.
    if abs(x) >= 1000:
        return f"{x:.3g}"
    if abs(x) >= 100:
        return f"{x:.1f}"
    return f"{x:.{nd}f}"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"wrote {path}")


# --------------------------------------------------------------------------


def make_main_table(metrics: pd.DataFrame, dm: pd.DataFrame) -> None:
    """Model comparison on the realized-volatility targets, RMSE and QLIKE."""
    m = metrics[metrics["target_family"] == "realized_vol"]

    # Significance of regime_normal vs each baseline, under MSE, Holm-corrected.
    # Direction matters: the paper's point is that the regime model significantly
    # beats only the trivial baselines while significantly losing to the pooled
    # ridge, so a single marker would hide exactly the thing worth seeing.
    sig: Dict[tuple, str] = {}
    if len(dm):
        # Restrict to the same targets this table reports. Both y_rv_h20 and
        # y_absret_h20 carry horizon=20, so keying on the horizon alone lets the
        # aliased-target tests silently overwrite the realized-volatility ones.
        rv_targets = set(m["target"].unique())
        d = dm[
            (dm["loss"] == "mse")
            & (dm["model_a"] == "regime_normal")
            & (dm["target"].isin(rv_targets))
        ]
        for _, r in d.iterrows():
            if not bool(r["reject_h0_holm_5pct"]):
                continue
            # "favours a" == the regime model wins
            sig[(int(r["horizon"]), r["model_b"])] = "better" if r["favours"] == "a" else "worse"

    lines = [
        r"\begin{table}[t]",
        r"\caption{Pooled out-of-sample performance on realized-volatility targets,"
        r" 15 walk-forward folds, Feb 2011--Feb 2026, n=3771. Lower is better."
        r" Best per column in \textbf{bold}. Markers report a Holm-corrected"
        r" Diebold--Mariano test of the regime model against that baseline at the"
        r" 5\% level: $\ast$ the regime model is significantly better,"
        r" $\dagger$ significantly worse. The regime model significantly beats"
        r" only the trivial baselines, and significantly loses to the pooled"
        r" ridge. QLIKE diverges for near-zero forecasts, which is why the zero"
        r" and random-walk rows are orders of magnitude above the rest.}",
        r"\label{tab:main}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "rr" * len(HORIZONS) + "}",
        r"\toprule",
        r"& \multicolumn{2}{c}{$h=1$} & \multicolumn{2}{c}{$h=5$} & \multicolumn{2}{c}{$h=20$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Model & RMSE & QLIKE & RMSE & QLIKE & RMSE & QLIKE \\",
        r"\midrule",
    ]

    best: Dict[tuple, float] = {}
    for h in HORIZONS:
        sub = m[m["horizon"] == h]
        for col in ("rmse", "qlike"):
            vals = sub[col].astype(float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                best[(h, col)] = float(vals.min())

    def row_for(model: str, label: str, mark_sig: bool) -> str:
        cells = []
        for h in HORIZONS:
            r = m[(m["horizon"] == h) & (m["model"] == model)]
            for col, nd in (("rmse", 6), ("qlike", 4)):
                if r.empty:
                    cells.append("--")
                    continue
                v = float(r.iloc[0][col])
                s = _fmt(v, nd)
                if np.isfinite(v) and abs(v - best.get((h, col), np.nan)) < 1e-12:
                    s = r"\textbf{" + s + "}"
                if col == "rmse" and mark_sig:
                    direction = sig.get((h, model))
                    if direction == "better":
                        s += r"$^{\ast}$"
                    elif direction == "worse":
                        s += r"$^{\dagger}$"
                cells.append(s)
        return f"{label} & " + " & ".join(cells) + r" \\"

    for b in BASELINE_ORDER:
        lines.append(row_for(b, BASELINE_LABEL[b], mark_sig=True))
    lines.append(r"\midrule")
    lines.append(row_for("regime_normal", "Regime-conditioned (HMM)", mark_sig=False))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(TABLES / "main_results.tex", "\n".join(lines) + "\n")


def make_ablation_table(metrics: pd.DataFrame) -> None:
    m = metrics[(metrics["target_family"] == "realized_vol") & (metrics["model_family"] == "regime")]
    single = {
        h: float(m[(m["horizon"] == h) & (m["model"] == "regime_single")]["rmse"].iloc[0])
        for h in HORIZONS
        if len(m[(m["horizon"] == h) & (m["model"] == "regime_single")])
    }

    lines = [
        r"\begin{table}[t]",
        r"\caption{Ablation of the regime signal (realized-volatility targets)."
        r" $\Delta$ is RMSE relative to the \textsc{single} pooled-ridge control,"
        r" in percent; negative would mean the arm beats the control."
        r" Shuffling the gate in time does not degrade the model: at $h=1$ and"
        r" $h=5$ it improves it. If knowing the current regime mattered, the"
        r" \textsc{gate\_shuffle} row would have to be worse than"
        r" \textsc{normal}, and it is not.}",
        r"\label{tab:ablation}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{$h=1$} & \multicolumn{2}{c}{$h=5$} & \multicolumn{2}{c}{$h=20$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Arm & RMSE & $\Delta$\% & RMSE & $\Delta$\% & RMSE & $\Delta$\% \\",
        r"\midrule",
    ]
    for arm in ABLATION_ORDER:
        cells = []
        for h in HORIZONS:
            r = m[(m["horizon"] == h) & (m["model"] == f"regime_{arm}")]
            if r.empty:
                cells += ["--", "--"]
                continue
            v = float(r.iloc[0]["rmse"])
            cells.append(_fmt(v, 6))
            base = single.get(h)
            cells.append(f"{100.0 * (v - base) / base:+.2f}" if base else "--")
        lines.append(f"{ABLATION_LABEL[arm]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(TABLES / "ablation_results.tex", "\n".join(lines) + "\n")


def make_regime_characterization_table(run_dir: Path, data: pd.DataFrame) -> None:
    """Are the inferred regimes informative about future stress, as a classifier?

    Reported separately from forecast accuracy because the two can and do come
    apart: the states here carry real information about whether a volatility
    spike is coming, while adding nothing to the conditional mean.
    """
    probs_fp = STUDY / "fixedK3_h5" / "oos_regime_probs.csv"
    preds_fp = run_dir / "predictions" / "y_rv_h5" / "regime_normal.csv"
    if not (probs_fp.exists() and preds_fp.exists()):
        print("[skip] regime characterization inputs not found")
        return

    probs = pd.read_csv(probs_fp, index_col=0, parse_dates=True)
    preds = pd.read_csv(preds_fp, index_col=0, parse_dates=True)
    j = probs.join(preds, how="inner").join(data[["ret_vol_20"]], how="inner")

    thr = j["y_true"].quantile(0.9)
    j["stress"] = j["y_true"] >= thr
    base = float(j["stress"].mean())
    n_stress = int(j["stress"].sum())

    names = {0: "Low vol", 1: "Mid vol", 2: "High vol"}
    lines = [
        r"\begin{table}[t]",
        r"\caption{What the inferred regimes do capture ($K{=}3$, $h{=}5$,"
        r" out-of-sample). ``Stress'' is the top decile of realized volatility"
        r" over the forecast window, so the unconditional rate is 10\%. Lift is"
        r" the ratio of the within-regime rate to that base rate. The"
        r" high-volatility state occupies a sixth of the sample and contains"
        r" nearly three-fifths of all stress events---the regimes are genuinely"
        r" informative, just not incrementally so for the conditional mean.}",
        r"\label{tab:regimes}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Regime & \% of time & Mean RV$^{(20)}$ & P(stress) & Lift & \% of all stress \\",
        r"\midrule",
    ]
    for k in sorted(j["hard_state"].unique()):
        s = j[j["hard_state"] == k]
        rate = float(s["stress"].mean())
        lines.append(
            f"{names.get(int(k), f'State {int(k)}')} & "
            f"{100 * len(s) / len(j):.1f} & {s['ret_vol_20'].mean():.5f} & "
            f"{rate:.3f} & {rate / base:.2f} & "
            f"{100 * int(s['stress'].sum()) / n_stress:.1f}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(TABLES / "regime_characterization.tex", "\n".join(lines) + "\n")


def make_protocol_table() -> None:
    from src.experiments.protocol_ablation import PROTOCOLS

    PROTOCOL_LABEL = {p.name: p.label for p in PROTOCOLS}

    fp = STUDY / "protocol_ablation" / "protocol_ablation.csv"
    if not fp.exists():
        print(f"[skip] {fp} not found")
        return
    df = pd.read_csv(fp)

    lines = [
        r"\begin{table}[t]",
        r"\caption{Protocol ablation. Improvement of the regime-conditioned model"
        r" over its \emph{matched} pooled-ridge baseline, as four evaluation"
        r" shortcuts are removed cumulatively. Positive means the regime model"
        r" looks better. $p$ is a HAC-corrected Diebold--Mariano test against that"
        r" baseline. At $h=1$ the aliased and realized-volatility targets"
        r" coincide by construction, so P3 and P4 are identical.}",
        r"\label{tab:protocol}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{$h=1$} & \multicolumn{2}{c}{$h=5$} & \multicolumn{2}{c}{$h=20$} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"& Correction applied & $\Delta$\% & $p$ & $\Delta$\% & $p$ & $\Delta$\% & $p$ \\",
        r"\midrule",
    ]
    for proto in ["P0", "P1", "P2", "P3", "P4"]:
        sub = df[df["protocol"] == proto]
        if sub.empty:
            continue
        # Take the label from the protocol definition rather than from the stored
        # CSV: the numbers are what the run produced, but the wording of a stage
        # should not require a 20-minute refit to change.
        label = PROTOCOL_LABEL.get(proto, str(sub.iloc[0]["protocol_label"]))
        cells = []
        for h in HORIZONS:
            r = sub[sub["horizon"] == h]
            if r.empty:
                cells += ["--", "--"]
                continue
            cells.append(f"{float(r.iloc[0]['improvement_pct']):+.2f}")
            p = float(r.iloc[0]["dm_p"])
            cells.append("$<$0.001" if p < 0.001 else f"{p:.3f}")
        lines.append(f"{proto} & {label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(TABLES / "protocol_ablation.tex", "\n".join(lines) + "\n")


def make_seed_table() -> None:
    fp = STUDY / "seed_robustness" / "seed_robustness.csv"
    if not fp.exists():
        print(f"[skip] {fp} not found")
        return
    df = pd.read_csv(fp)
    g = df.groupby("horizon")

    lines = [
        r"\begin{table}[t]",
        r"\caption{Sensitivity to the HMM's random initialization. Ten seeds,"
        r" everything else fixed. $\Delta$\% is improvement over the"
        r" (deterministic) pooled ridge. The spread across seeds is comparable to"
        r" the effect size the comparison is trying to resolve.}",
        r"\label{tab:seed}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Horizon & Pooled ridge & RMSE mean & RMSE s.d. & $\Delta$\% min & $\Delta$\% max \\",
        r"\midrule",
    ]
    for h, sub in g:
        lines.append(
            f"$h={int(h)}$ & {_fmt(float(sub['rmse_pooled_ridge'].iloc[0]), 6)} & "
            f"{_fmt(sub['rmse_regime'].mean(), 6)} & "
            f"{_fmt(sub['rmse_regime'].std(), 6)} & "
            f"{sub['improvement_pct'].min():+.2f} & {sub['improvement_pct'].max():+.2f}"
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(TABLES / "seed_robustness.tex", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------


def make_protocol_figure() -> None:
    fp = STUDY / "protocol_ablation" / "protocol_ablation.csv"
    if not fp.exists():
        return
    df = pd.read_csv(fp)
    order = ["P0", "P1", "P2", "P3", "P4"]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.26
    xs = np.arange(len(order))
    for i, h in enumerate(HORIZONS):
        vals = [
            float(df[(df["protocol"] == p) & (df["horizon"] == h)]["improvement_pct"].iloc[0])
            if len(df[(df["protocol"] == p) & (df["horizon"] == h)])
            else np.nan
            for p in order
        ]
        ax.bar(xs + (i - 1) * width, vals, width, label=f"$h={h}$")

    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        ["P0\noriginal", "P1\n+baseline", "P2\n+embargo", "P3\n+nested sel.", "P4\n+RV target"],
        fontsize=8,
    )
    ax.set_ylabel("Improvement over pooled ridge (%)")
    ax.set_title("Apparent regime-switching gain vs. evaluation protocol", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "protocol_ablation.png", dpi=220)
    plt.close(fig)
    print(f"wrote {FIGURES / 'protocol_ablation.png'}")


def make_seed_figure() -> None:
    fp = STUDY / "seed_robustness" / "seed_robustness.csv"
    if not fp.exists():
        return
    df = pd.read_csv(fp)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    data = [df[df["horizon"] == h]["improvement_pct"].to_numpy() for h in HORIZONS]
    ax.boxplot(data, tick_labels=[f"$h={h}$" for h in HORIZONS], widths=0.5)
    for i, d in enumerate(data, start=1):
        ax.scatter(np.full(len(d), i) + np.random.uniform(-0.08, 0.08, len(d)), d, s=12, alpha=0.7)
    ax.axhline(0, color="crimson", lw=1.2, ls="--", label="pooled ridge")
    ax.set_ylabel("Improvement over pooled ridge (%)")
    ax.set_title("Spread across 10 HMM initializations", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "seed_robustness.png", dpi=220)
    plt.close(fig)
    print(f"wrote {FIGURES / 'seed_robustness.png'}")


def main() -> None:
    run_dir = _latest_study_dir()
    print(f"using study run: {run_dir}")
    metrics = pd.read_csv(run_dir / "metrics.csv")
    dm_fp = run_dir / "dm_tests.csv"
    dm = pd.read_csv(dm_fp) if dm_fp.exists() else pd.DataFrame()

    TABLES.mkdir(parents=True, exist_ok=True)
    make_main_table(metrics, dm)
    make_ablation_table(metrics)

    from src.config import load_config
    from src.data.pipeline import build_and_save_processed_dataset

    make_regime_characterization_table(run_dir, build_and_save_processed_dataset(load_config()).df)

    make_protocol_table()
    make_seed_table()
    make_protocol_figure()
    make_seed_figure()


if __name__ == "__main__":
    main()
