"""Table for the joint-vs-two-stage comparison."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS = Path("artifacts") / "study" / "joint_vs_twostage" / "joint_vs_twostage.csv"
TABLES = Path("paper") / "tables"
HORIZONS = [1, 5, 20]

LABEL = {
    "ms_level": r"Markov switching (level)",
    "ms_slope": r"Markov switching (dynamics)",
    "garch11": r"GARCH(1,1)-$t$ \quad(no regimes)",
    "SPY": r"\textbf{SPY (real data)}",
}
ORDER = ["ms_level", "ms_slope", "garch11", "SPY"]


def main() -> None:
    df = pd.read_csv(RESULTS)
    if "error" in df.columns:
        df = df[df["error"].isna()]

    lines = [
        r"\begin{table}[t]",
        r"\caption{Two-stage versus jointly-estimated regimes. Both are RMSE"
        r" improvement over the same pooled ridge, in percent, averaged over four"
        r" replications for the synthetic worlds. Both arms use the identical"
        r" protocol: embargoed walk-forward splits, $K$ and the ridge penalty"
        r" selected on the inner validation block, and causal prediction. Joint"
        r" estimation is much better where regimes exist and differ in level, no"
        r" better where they differ only in dynamics, and markedly worse where"
        r" there are no regimes to find.}",
        r"\label{tab:joint}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Two-stage} & \multicolumn{3}{c}{Joint} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Data-generating process & $h{=}1$ & $h{=}5$ & $h{=}20$"
        r" & $h{=}1$ & $h{=}5$ & $h{=}20$ \\",
        r"\midrule",
    ]
    for case in ORDER:
        s = df[df["case"] == case]
        if s.empty:
            continue
        if case == "SPY":
            lines.append(r"\midrule")
        cells = []
        for col in ("twostage_vs_pooled_pct", "joint_vs_pooled_pct"):
            for h in HORIZONS:
                v = s[s["horizon"] == h][col]
                cells.append("--" if not len(v) else f"{v.mean():+.2f}")
        lines.append(f"{LABEL.get(case, case)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "joint.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {TABLES / 'joint.tex'}")

    print("\n=== joint minus two-stage, paired over reps and horizons ===")
    for case in ORDER:
        s = df[df["case"] == case]
        if s.empty:
            continue
        d = s["joint_vs_pooled_pct"] - s["twostage_vs_pooled_pct"]
        print(f"  {case:<10} mean {d.mean():+6.2f}pp   joint better in "
              f"{int((d > 0).sum())}/{len(d)}")


if __name__ == "__main__":
    main()
