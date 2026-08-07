"""Tables and figures for the cross-asset study."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path("artifacts") / "study" / "multi_asset" / "multi_asset_results.csv"
TABLES = Path("paper") / "tables"
FIGURES = Path("paper") / "figures"
HORIZONS = [1, 5, 20]


def _load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS)
    if "error" in df.columns:
        df = df[df["error"].isna()] if df["error"].notna().any() else df
    return df


def make_table(df: pd.DataFrame) -> None:
    """Per-asset improvement of the regime model over its pooled-ridge control."""
    reg = df[df["model"] == "regime_normal"].copy()

    lines = [
        r"\begin{table}[t]",
        r"\caption{Cross-asset replication. Each cell is the out-of-sample RMSE"
        r" improvement of the regime-conditioned model over the pooled ridge"
        r" fitted on identical features, in percent; positive would favour regime"
        r" conditioning. $^{\dagger}$ marks a Diebold--Mariano rejection at the"
        r" 5\% level, always in the direction of the pooled ridge. Assets are"
        r" sorted by mean trailing volatility.}",
        r"\label{tab:multiasset}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Ticker & Class & Vol & $h{=}1$ & $h{=}5$ & $h{=}20$ \\",
        r"\midrule",
    ]

    order = reg.groupby("ticker")["mean_vol"].first().sort_values().index
    for t in order:
        sub = reg[reg["ticker"] == t]
        if sub.empty:
            continue
        cls = str(sub.iloc[0]["asset_class"])
        vol = float(sub.iloc[0]["mean_vol"])
        cells = []
        for h in HORIZONS:
            r = sub[sub["horizon"] == h]
            if r.empty:
                cells.append("--")
                continue
            v = float(r.iloc[0]["improvement_pct"])
            s = f"{v:+.2f}"
            p = r.iloc[0].get("dm_p_vs_pooled", np.nan)
            if pd.notna(p) and float(p) < 0.05:
                s += r"$^{\dagger}$"
            cells.append(s)
        safe = t.replace("-", "--")
        lines.append(f"{safe} & {cls} & {vol:.4f} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    med = [reg[reg["horizon"] == h]["improvement_pct"].median() for h in HORIZONS]
    lines.append("Median & & & " + " & ".join(f"{v:+.2f}" for v in med) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "multi_asset.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {TABLES / 'multi_asset.tex'}")


def make_figure(df: pd.DataFrame) -> None:
    reg = df[df["model"] == "regime_normal"]
    order = list(reg.groupby("ticker")["improvement_pct"].mean().sort_values().index)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.5, 5.2), gridspec_kw={"width_ratios": [2.4, 1]}
    )

    colors = {1: "#4C78A8", 5: "#F58518", 20: "#54A24B"}
    ys = np.arange(len(order))
    for h in HORIZONS:
        vals = [
            float(reg[(reg["ticker"] == t) & (reg["horizon"] == h)]["improvement_pct"].iloc[0])
            if len(reg[(reg["ticker"] == t) & (reg["horizon"] == h)])
            else np.nan
            for t in order
        ]
        ax1.scatter(vals, ys, s=34, color=colors[h], label=f"$h={h}$", zorder=3)

    ax1.axvline(0, color="crimson", lw=1.3, ls="--", zorder=2)
    for y in ys:
        ax1.axhline(y, color="0.92", lw=0.8, zorder=1)
    ax1.set_yticks(ys)
    ax1.set_yticklabels(order, fontsize=8)
    ax1.set_xlabel("RMSE improvement over pooled ridge (%)")
    ax1.set_title("Regime conditioning by asset", fontsize=10)
    ax1.legend(frameon=False, fontsize=8, loc="lower left")

    data = [reg[reg["horizon"] == h]["improvement_pct"].dropna().to_numpy() for h in HORIZONS]
    ax2.boxplot(data, tick_labels=[f"$h={h}$" for h in HORIZONS], widths=0.55)
    for i, d in enumerate(data, start=1):
        ax2.scatter(np.full(len(d), i) + np.random.uniform(-0.09, 0.09, len(d)),
                    d, s=16, alpha=0.65, color="#4C78A8")
    ax2.axhline(0, color="crimson", lw=1.3, ls="--")
    ax2.set_ylabel("Improvement (%)")
    ax2.set_title("Distribution", fontsize=10)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "multi_asset.png", dpi=220)
    plt.close(fig)
    print(f"wrote {FIGURES / 'multi_asset.png'}")


def make_gate_shuffle_figure(df: pd.DataFrame) -> None:
    """How much is regime *timing* worth, and how does that compare to its cost?

    Plotted as the paired difference rather than as raw RMSE against RMSE: the
    assets span an order of magnitude in volatility level, so on a scatter of
    RMSE-vs-RMSE every point sits on the diagonal and an effect of well under a
    percent is invisible. The quantity of interest is the within-pair gap.
    """
    from scipy import stats

    cols = ["ticker", "horizon", "rmse"]
    a = df[df["model"] == "regime_normal"][cols]
    b = df[df["model"] == "regime_gate_shuffle"][cols]
    s = df[df["model"] == "regime_single"][cols].rename(columns={"rmse": "rmse_pooled"})
    m = a.merge(b, on=["ticker", "horizon"], suffixes=("_true", "_shuf")).merge(
        s, on=["ticker", "horizon"]
    )
    if m.empty:
        return

    # Everything in percent of the pooled-ridge RMSE, so assets are comparable.
    m["rel_true"] = 100 * (m["rmse_pooled"] - m["rmse_true"]) / m["rmse_pooled"]
    m["rel_shuf"] = 100 * (m["rmse_pooled"] - m["rmse_shuf"]) / m["rmse_pooled"]
    m["gap"] = m["rel_true"] - m["rel_shuf"]  # > 0 => true timing helps

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.6))

    colors = {1: "#4C78A8", 5: "#F58518", 20: "#54A24B"}
    for i, h in enumerate(HORIZONS, start=1):
        g = m[m["horizon"] == h]["gap"].to_numpy()
        ax1.scatter(np.full(len(g), i) + np.random.uniform(-0.11, 0.11, len(g)),
                    g, s=26, alpha=0.75, color=colors[h])
        ax1.hlines(g.mean(), i - 0.26, i + 0.26, color="black", lw=2, zorder=4)
    ax1.axhline(0, color="crimson", lw=1.3, ls="--")
    ax1.set_xticks(list(range(1, len(HORIZONS) + 1)))
    ax1.set_xticklabels([f"$h={h}$" for h in HORIZONS])
    ax1.set_ylabel("Value of true timing (pp of RMSE)")
    w = stats.wilcoxon(m["gap"])
    ax1.set_title(f"Regime timing is worth a little\n(Wilcoxon $p$={w.pvalue:.3f})", fontsize=9)

    # the comparison that matters: signal gained vs. efficiency lost
    ax2.bar(["value of\ntrue timing", "cost of\npartitioning"],
            [m["gap"].mean(), -m["rel_true"].mean()],
            color=["#54A24B", "#E45756"])
    ax2.axhline(0, color="black", lw=1)
    ax2.set_ylabel("pp of RMSE")
    ax2.set_title("...but far less than it costs", fontsize=9)
    for i, v in enumerate([m["gap"].mean(), -m["rel_true"].mean()]):
        ax2.text(i, v + 0.08, f"{v:+.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "multi_asset_gate_shuffle.png", dpi=220)
    plt.close(fig)
    print(f"wrote {FIGURES / 'multi_asset_gate_shuffle.png'}")


def print_summary(df: pd.DataFrame) -> None:
    reg = df[df["model"] == "regime_normal"]
    print("\n=== cross-asset summary (regime vs pooled ridge) ===")
    for h in HORIZONS:
        s = reg[reg["horizon"] == h]["improvement_pct"].dropna()
        pos = int((s > 0).sum())
        sig_worse = int(
            (
                (reg[reg["horizon"] == h]["dm_p_vs_pooled"] < 0.05)
                & (reg[reg["horizon"] == h]["improvement_pct"] < 0)
            ).sum()
        )
        sig_better = int(
            (
                (reg[reg["horizon"] == h]["dm_p_vs_pooled"] < 0.05)
                & (reg[reg["horizon"] == h]["improvement_pct"] > 0)
            ).sum()
        )
        print(f"h={h:2d}  n={len(s):2d}  median={s.median():+.2f}%  "
              f"range=[{s.min():+.2f}, {s.max():+.2f}]  "
              f"positive={pos}/{len(s)}  sig.better={sig_better}  sig.worse={sig_worse}")

    a = df[df["model"] == "regime_normal"][["ticker", "horizon", "rmse"]]
    b = df[df["model"] == "regime_gate_shuffle"][["ticker", "horizon", "rmse"]]
    m = a.merge(b, on=["ticker", "horizon"], suffixes=("_n", "_s"))
    if len(m):
        print(f"\ngate_shuffle beats true timing in {int((m['rmse_s'] < m['rmse_n']).sum())}"
              f"/{len(m)} asset-horizon pairs")

    # is there any relationship between asset volatility and the effect?
    piv = reg.dropna(subset=["improvement_pct", "mean_vol"])
    for h in HORIZONS:
        s = piv[piv["horizon"] == h]
        if len(s) > 3:
            c = float(np.corrcoef(s["mean_vol"], s["improvement_pct"])[0, 1])
            print(f"corr(mean_vol, improvement) at h={h}: {c:+.3f}")


def main() -> None:
    df = _load()
    make_table(df)
    make_figure(df)
    make_gate_shuffle_figure(df)
    print_summary(df)


if __name__ == "__main__":
    main()
