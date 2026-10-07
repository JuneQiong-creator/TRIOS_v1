# Local release validation

Validation date: 2026-10-07. Platform: Windows; Python 3.13.5. A new virtual environment was created, pinned requirements were installed, and the package was built and installed. The complete lock file was then installed and the package rebuilt from a separate fresh directory copy. `pip check` found no broken requirements. All 14 package modules imported successfully. This is validation on one platform, not a cross-platform certification.

## Commands actually checked from the fresh copy

```bash
python -m pip install -r environment/validated-lock.txt
python -m pip install .
python -m pip check
python scripts/verify_manifest.py
python scripts/verify_stored_results.py --out outputs/stored_checks.json
python scripts/ccle_entry.py --mode verify-stored --out outputs/ccle_stored_checks.json
python examples/hb_frozen_demo.py --out outputs/hb_demo.json
python examples/ccle_fixed_score_smoke.py --out outputs/ccle_smoke.json
python scripts/replay_figures.py --out outputs/figures --include-figure1
```

All listed commands passed. Figure 1 used the already installed pdfLaTeX/TikZ environment; the release script does not install TeX. Its new article/geometry wrapper was verified to produce one unclipped page while the drawing source remained byte-identical.

## Verified computational scope

- One existing 22-profile 5-FU cohort: original frozen stage quantities selected D=50; frozen spline coefficients were evaluated with exact-log K=7 geometry; score maximum absolute difference was 0 and stored labels/cutoffs matched. No fits, new observations or LOO were performed for this example.
- Synthetic n=12 fixed-score input: 12 classifier-only folds, original five components and failed-fold/full-invalid controls passed. The historical HB artificial objective-tie fixture returned (5,9); the current CCLE version returned (4,9). These differences are retained, not patched.
- Stored result checks: current CCLE FULL 567 tasks and B100 56,700 intended tasks, 81 methods and seven datasets; TRIOS applicability and both stored ranks; current simulation 81-method anchors and HB operational anchor. These are checks of packaged stored fields, not independent recomputation or certification of all folds.
- Figures 1–8: eight single-page vector PDFs with embedded fonts; 22 figure products (Figure 1 PDF; Figures 2–8 PDF/SVG/300-dpi PNG). Figures 2–8 PNG pixels were identical to the accepted manuscript figures. The contact sheet and dense Figure 7 were visually inspected without clipping or overlap. The original drawing code, source values, scientific annotations and layout were retained.
- Input integrity: 137 source bindings and 15 verbatim computational functions checked; 83 packaged Python files parsed. Protected source inputs (137 records) and manuscript sources/figures (243 files) were unchanged. The public HB column allowlist yields 1,832 source rows/114 profiles and 1,817 eligible rows/113 profiles.
- Payload scanning found no host absolute paths, recognized credential/private-key patterns or disallowed raw CCLE inputs. The maximum file size was below 4 MB. Manifest checking verifies both hashes and complete file-set membership, excluding generated outputs and build/Git metadata. Relative Markdown file links were checked before release.

## Limits

No complete bank, benchmark, native LOO archive, bootstrap or molecular test was rerun. Original full-recompute producers remain evidence-only source copies with missing inputs and dependency states. Exact legacy CCLE redistribution rights were not confirmed; raw third-party files are omitted. The project licence remains author-confirmation pending. Missing historical stress-generator provenance, missing NP_PCHIP states, empty composed certified sets and the unfinished ideal-real-function bridge remain limitations. See `KNOWN_LIMITATIONS.md` and the reproduction-level documentation.

The release checks establish the explicitly listed small/replay/stored-byte scope only. They do not establish complete experimental reproducibility, universal AB optimality or biological-truth recovery.
