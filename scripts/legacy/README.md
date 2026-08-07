# Legacy pipeline

These are the original experiment scripts, kept for provenance. They are
**superseded** and should not be used to produce results.

| File | Superseded by |
|---|---|
| `run_experiments.py` | `src/experiments/run_study.py` |
| `ablation_summary.py` (was `test.py`) | `src/experiments/make_paper_tables.py` |
| `rerun_per_regime_compare.py` (was `test2.py`) | `src/experiments/run_study.py` |

They are retained because the paper's protocol ablation reconstructs the
evaluation these scripts implemented, and it is useful to be able to point at
the original. Note in particular that `run_experiments.py`:

* selected the number of regimes and the gating rule by lowest **out-of-sample**
  RMSE and then reported that same number;
* used no embargo between the training and test windows;
* compared against a ridge with a fixed penalty on unstandardized features;
* forecast `|r_{t+h}|` rather than realized volatility over `[t+1, t+h]`;
* included `uniform` and `no_regime` ablation arms which are provably identical
  to each other and to the pooled ridge, because experts are assigned by
  `argmax` and a uniform probability vector always selects state 0.

Those five properties are stages P0–P4 of the protocol ablation in the paper.
The two ad-hoc scripts additionally reference hard-coded run ids that no longer
exist in `artifacts/`.
