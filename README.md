# TRIOS reproduction materials

TRIOS provides ordered low, intermediate and high sensitivity strata from multidose response profiles in small cohorts. It separates the pharmacodynamic domain retained after reconstruction, the finite response-evaluation resolution, and experimentally acquired concentrations. The paper is **Reliable ordered stratification of multidose response profiles in small cohorts under limited dose information** by Qiong Liu, Siwei Mao, Ji Ma, Qiuhui Pan, Xiaoling Peng and Ping He. No DOI/publication status is asserted here.

**Release scope:** this is a scoped repository of bound code, authorized de-identified HB-PDO drug-response data, frozen summary/figure source tables and validated small examples. It is **not a demonstrated complete re-execution** of all experiments. See [reproduction levels](docs/REPRODUCTION_LEVELS.md), [local validation report](docs/LOCAL_VALIDATION_REPORT.md) and [known limitations](docs/KNOWN_LIMITATIONS.md).

## Layout

- `src/trios_release/`: exact-log scoring geometry, frozen reconstruction, distinct hash-bound AB implementations, classifiers and ORIGINAL5 evaluation functions.
- `data/hb/`: authorized measurements, source-to-eligible membership and stored fit parameters/status; [data dictionary](data/hb/DATA_DICTIONARY.md).
- `data/ccle/`: exact legacy input names/hashes, official acquisition instructions and locked subset membership. Third-party raw files are not mirrored.
- `configs/`: available frozen simulation generator/stress locks. [Current analysis contracts](configs/ANALYSIS_CONTRACTS.json) distinguish current results and perturbation protocols.
- `results/`: study-derived current HB, Q50 81-method simulation, prospective and CCLE frozen aggregates. No old QSC/native Option C comparison replaces the current CCLE authority.
- `figures/`: accepted drawing code and full-precision source tables for Figures 1–8. `reference_figures/` holds accepted quantitative PDFs for visual comparison.
- `archived_sources/`: available historical generation, analysis and prospective scripts with host paths redacted. Missing original inputs/dependencies prevent a claim of complete runnable reproduction.
- `scripts/`, `examples/`: safe manifest checks, stored-result diagnostics, small examples and plot replay. Outputs go to the directory explicitly supplied by the user.
- `provenance/`: source/public hashes, computational function bindings, result index and payload manifest.

## Installation

Python **3.13** is required; the small-release validation used Python **3.13.5**. Historical CCLE scientific versions are separately recorded in `environment/FROZEN_CCLE_EXECUTION_ENVIRONMENT.json`. Tested numeric/drawing dependencies are pinned in `environment/requirements.txt`; the complete installed dependency set is recorded in `environment/validated-lock.txt`. This was tested in a new Windows virtual environment and a fresh directory copy, not every platform.

Clone [TRIOS_v1](https://github.com/JuneQiong-creator/TRIOS_v1) or [download the main-branch ZIP](https://github.com/JuneQiong-creator/TRIOS_v1/archive/refs/heads/main.zip), then run from its root:

```bash
git clone https://github.com/JuneQiong-creator/TRIOS_v1.git
cd TRIOS_v1
```

```bash
python -m venv .venv
```

Activate it on Windows PowerShell with `.venv/Scripts/Activate.ps1`, or on POSIX with `source .venv/bin/activate`. Then:

```bash
python -m pip install -r environment/validated-lock.txt
python -m pip install .
python scripts/verify_manifest.py
```

No token, private credentials or TeX installation is performed by these scripts. Code/data licences remain to be confirmed by the authors; public deposit does not assign MIT/CC-BY or override third-party terms. See [licence status](docs/LICENSE_STATUS.md).

## 1. Replay figures from frozen source tables

```bash
python scripts/replay_figures.py --out outputs/figures
```

This redraws Figures 2–8 from packaged tables without simulation, fitting, domain selection, classification, LOO, tests or bootstrap. Figure 1 is the unchanged TikZ schematic; with an existing TeX installation supporting `article`, `geometry`, `amsmath` and `tikz`:

```bash
python scripts/replay_figures.py --out outputs/figures --include-figure1
```

`--group 4` can replay one group only. Generated outputs are never written to frozen source-data folders. Fonts can differ if Arial is unavailable; font binaries are not redistributed. [Figure contract](figures/FIGURE_REPLAY_CONTRACT.md) records source scope and display conventions.

## 2. Check stored results and bytes

```bash
python scripts/verify_stored_results.py --out outputs/stored_checks.json
python scripts/ccle_entry.py --mode verify-stored --out outputs/ccle_stored_checks.json
```

These inspect exact payload bytes, declared current contract, stored task coverage and numerical regression anchors. They do not independently rerank, certify all folds or reproduce the original experiments. SHA validation excludes the manifest itself, Git metadata, virtual environments and generated `outputs/`.

## 3. Small computation examples

```bash
python examples/hb_frozen_demo.py --out outputs/hb_demo.json
python examples/ccle_fixed_score_smoke.py --out outputs/ccle_smoke.json
```

The HB example uses one existing 5-FU cohort: original frozen profile-level stage quantities select the domain, stored spline parameters are reloaded, exact-log K=7 CRS is evaluated and the historical HB AB partition is compared with its frozen labels/cutoffs. It does not fit from measurements, run LOO or reconstruct all 81 benchmark arms. Scores are checked against a predeclared 1e-12 floating diagnostic tolerance, not certified as ideal real-valued quantities.

The CCLE example uses a synthetic n=12 vector and supplied full labels/cutoffs. Only classifier-only folds are trained. It exercises the current five-component evaluator and failure accounting, and preserves the historical-versus-current artificial tie-fixture difference. It is not an evaluation of the seven real datasets. `src/trios_release/fixed_score.py` accepts supplied frozen full states; it never reselects a domain or regenerates a response representation.

## Complete scientific recomputation — not yet validated

Available simulation-generation, exact-log analysis, current CCLE engine and prospective-extension scripts are indexed in `archived_sources/` and `provenance/SOURCE_BINDINGS.csv`. Their original worker banks, some binary fit/fold states, historical inputs and dependency snapshots are not all packaged. Original mains are provided as source evidence, **not validated README execution commands**. `scripts/ccle_entry.py --mode full-recompute` deliberately reports the missing-input limitation rather than launching a substitute benchmark.

Current CCLE FULL/B100 comparison is fixed D2 domains and fixed native scalar scores, subset-specific partitions and fixed-score classifier-only LOO under ORIGINAL5. HB retains its own method-native frozen-fit domain-reselection/rescoring protocol. Simulation E3 retains its own frozen definitions and weighting. Equal component names or algebra do not unify these protocols. Prospective Stage 1 current selection uses overall A_valid >=0.95; H250-reference fidelity is descriptive sensitivity evidence, not a second current selection gate. Rounded J5 remains separate. No thresholds, weights, seeds or frozen numeric results are changed for publication.

## Data, rights and remaining limitations

HB raw clinical records, direct identifiers, identity mapping and restricted associated material are excluded. Contact Ping He at heping@bnbu.edu.cn for restricted associated material; no invented access conditions are imposed. The same de-identified measurement resource underlies the earlier HB-PDO study, but this repository preserves TRIOS-specific source eligibility and strata.

Exact legacy CCLE raw files are obtained from the official provider according to [data/ccle/README.md](data/ccle/README.md); no newer release is silently substituted. No exact-file mirror permission was confirmed. Licences remain to be supplied by the rights holders.

The contemporaneous external stress-generator snapshot/hash is still unavailable; present code is not advertised as recovered historical source. Stored-score exact checks do not certify ideal real-valued CRS, and guarded-shortlist AB refinement has no general proven global-optimum inclusion guarantee. S8 missing NP_PCHIP states, empty joint certified sets and incomplete strict real-function/ideal-node bridge remain explicitly documented. No universal pipeline consistency, biological-truth recovery or complete reproducibility claim is made.
