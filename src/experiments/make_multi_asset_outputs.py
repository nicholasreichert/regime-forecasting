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

    # Only the paired gap is shown here; the full cost accounting lives in
    # cost_decomposition.png, which separates the partition cost from the cost of
    # the regime signal itself. Putting a single "cost" bar next to this panel
    # would conflate the two.
    fig, ax1 = plt.subplots(figsize=(5.0, 3.6))

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
    ax1.set_title(f"Correct regime timing is worth a little\n"
                  f"(mean $+${m['gap'].mean():.2f} pp, Wilcoxon $p$={w.pvalue:.3f})", fontsize=9)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "multi_asset_gate_shuffle.png", dpi=220)
    plt.close(fig)
    print(f"wrote {FIGURES / 'multi_asset_gate_shuffle.png'}")


VARIANT_ORDER = ["regime_normal", "regime_gate_shuffle", "regime_shrunk", "regime_features",
                 "har", "garch11", "ewma"]
VARIANT_LABEL = {
    "regime_normal": r"Regime gating (full mixture)",
    "regime_gate_shuffle": r"\quad with time-shuffled gate",
    "regime_shrunk": r"Regime gating, shrunk to pooled",
    "regime_features": r"Regime posteriors as features",
    "har": "HAR",
    "garch11": "GARCH(1,1)",
    "ewma": "EWMA",
}


def make_variant_table(df: pd.DataFrame) -> None:
    """Every route the regime signal can take into the model, plus the benchmarks."""
    from scipy import stats

    piv = df.pivot_table(index=["ticker", "horizon"], columns="model", values="improvement_pct")

    lines = [
        r"\begin{table}[t]",
        r"\caption{Every way of using the regime signal, against the pooled ridge,"
        r" over 60 asset-horizon pairs. ``Median'' is the median improvement in"
        r" percent (positive favours the row). ``Wins'' counts pairs where the row"
        r" beats the pooled ridge; ``sig.\ better'' and ``sig.\ worse'' count"
        r" Diebold--Mariano rejections at 5\%. No regime variant achieves a single"
        r" significant improvement in 240 tests, whereas GARCH and EWMA---which"
        r" ignore regimes entirely---beat the pooled ridge more often than not.}",
        r"\label{tab:variants}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Median & Wins & sig.\ better & sig.\ worse \\",
        r"\midrule",
    ]
    for m in VARIANT_ORDER:
        s = df[df["model"] == m]
        if s.empty:
            continue
        imp = s["improvement_pct"].dropna()
        sig_b = int(((s["dm_p_vs_pooled"] < 0.05) & (s["improvement_pct"] > 0)).sum())
        sig_w = int(((s["dm_p_vs_pooled"] < 0.05) & (s["improvement_pct"] < 0)).sum())
        if m == "regime_features":
            lines.append(r"\midrule")
        lines.append(
            f"{VARIANT_LABEL[m]} & {imp.median():+.2f}\\% & {int((imp > 0).sum())}/{len(imp)} "
            f"& {sig_b} & {sig_w}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "variants.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {TABLES / 'variants.tex'}")

    # decomposition, reported as paired medians so the pieces are comparable
    norm, feat = piv["regime_normal"], piv["regime_features"]
    partition = float((feat - norm).median())
    residual = float(-feat.median())
    p_part = stats.wilcoxon((feat - norm).dropna()).pvalue
    p_res = stats.wilcoxon(feat.dropna()).pvalue
    print(f"  partition cost {partition:.2f}pp (p={p_part:.1e}), "
          f"residual signal cost {residual:.2f}pp (p={p_res:.1e})")


def make_decomposition_figure(df: pd.DataFrame) -> None:
    """Where the loss comes from: the signal itself, and then the gating on it.

    Uses means, not medians. The decomposition
    ``normal = features + (normal - features)`` is exactly additive in means but
    not in medians, and a chart whose bars visibly fail to sum to its own total
    invites the reader to distrust the rest of the paper.

    The timing term is deliberately drawn apart from the other two: it is a
    different comparison (gated vs. gated-with-shuffled-gate), not a component of
    the total, and stacking it alongside would imply an accounting relationship
    that does not hold.
    """
    piv = df.pivot_table(index=["ticker", "horizon"], columns="model", values="improvement_pct")
    norm, feat, gsh = piv["regime_normal"], piv["regime_features"], piv["regime_gate_shuffle"]

    signal = float(feat.mean())                 # cost of injecting the signal at all
    gating = float((norm - feat).mean())        # extra cost of gating on it
    total = float(norm.mean())                  # == signal + gating, exactly
    timing = float((norm - gsh).mean())

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    xs = [0, 1, 2, 3.4]
    vals = [signal, gating, total, timing]
    labels = ["cost of the\nregime signal", "extra cost\nof gating",
              "= total", "value of\ncorrect timing"]
    colors = ["#F58518", "#E45756", "#8C8C8C", "#54A24B"]
    bars = ax.bar(xs, vals, color=colors, width=0.72)
    ax.axhline(0, color="black", lw=1)
    ax.axvline(2.7, color="0.75", lw=1, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("pp of RMSE vs pooled ridge")
    ax.set_title("Why regime conditioning loses (mean over 60 asset-horizon pairs)",
                 fontsize=9.5)

    span = max(abs(min(vals)), abs(max(vals)))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.06 * span if v > 0 else -0.12 * span),
                f"{v:+.2f}", ha="center", fontsize=9)
    ax.set_ylim(min(vals) - 0.32 * span, max(vals) + 0.28 * span)
    ax.text(3.4, max(vals) + 0.12 * span, "separate\ncomparison",
            ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "cost_decomposition.png", dpi=220)
    plt.close(fig)
    print(f"wrote {FIGURES / 'cost_decomposition.png'}")


def make_equivalence_table(df: pd.DataFrame) -> None:
    """What size of improvement can the data actually exclude?"""
    lines = [
        r"\begin{table}[t]",
        r"\caption{Equivalence bounds. For each asset-horizon pair we form a 95\%"
        r" confidence interval on the regime model's RMSE improvement over the"
        r" pooled ridge and report the upper limit, i.e.\ the largest improvement"
        r" the data cannot exclude. A negative bound means \emph{no} improvement of"
        r" any size is consistent with the data at 95\%. This is the statement a"
        r" null result requires: failing to reject equal accuracy is also what an"
        r" underpowered test produces.}",
        r"\label{tab:equivalence}",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Variant & Horizon & Median bound & Worst case & Bound $<0$ \\",
        r"\midrule",
    ]
    for m in ["regime_normal", "regime_features", "regime_shrunk"]:
        s = df[df["model"] == m].dropna(subset=["ci_upper_pct"])
        if s.empty:
            continue
        for h in HORIZONS:
            x = s[s["horizon"] == h]["ci_upper_pct"]
            if not len(x):
                continue
            label = VARIANT_LABEL.get(m, m) if h == HORIZONS[0] else ""
            lines.append(
                f"{label} & $h={h}$ & {x.median():+.2f}\\% & {x.max():+.2f}\\% "
                f"& {int((x < 0).sum())}/{len(x)}" + r" \\"
            )
        lines.append(r"\midrule")
    lines = lines[:-1]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "equivalence.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {TABLES / 'equivalence.tex'}")


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
    make_variant_table(df)
    make_equivalence_table(df)
    make_decomposition_figure(df)
    make_figure(df)
    make_gate_shuffle_figure(df)
    print_summary(df)


if __name__ == "__main__":
    main()
