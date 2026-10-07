# Three reproduction levels

| Level | Runnable entry | Actual scope |
|---|---|---|
| 1: replay frozen tables | `python scripts/replay_figures.py --out outputs/figures` | Figures 2–8 from the packaged full-precision tables, no source producer. Figure 1 uses `--include-figure1` and local TeX. |
| 2: inspect stored results | `python scripts/verify_stored_results.py --out outputs/stored_checks.json` | File integrity and declared current summary/regression anchors. No new ranks, confidence intervals or fold certification. |
| 3: compute from inputs, small example | `python examples/hb_frozen_demo.py --out outputs/hb_demo.json`; `python examples/ccle_fixed_score_smoke.py --out outputs/ccle_smoke.json` | One real HB frozen-fit regimen stage/score/partition replay, and one synthetic fixed-score classifier-only cohort. This is not an all-experiment reproduction. |

Full scientific recomputation remains an unvalidated path. Available original scripts are indexed under `archived_sources`; do not launch their mains as installation smoke checks. For a complete rerun, assemble hash-matched original input/fit/fold banks, legacy CCLE inputs and missing dependencies and obtain separate execution approval. Do not use figure tables as substitutes for raw observations. Research application should also resolve the project licence.
