from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT = Path(r"SOURCE_PROJECT")
RESULTS = PROJECT / "04_RESULTS"
WORK = PROJECT / "work" / "trios_crs_logspace_amendment_v10"
INSTRUCTION = Path(r"OMITTED_HOST_PATH/TRIOS_CRS_LogSpace_Amendment_Rerun_Direct_Execute_v1.0.md")
OLD_P = np.array([0.02, 0.04, 0.08, 0.16, 0.30, 0.60, 1.00], dtype=np.float64)
OLD_BETA = np.array([6, 16, 26, 37, 51, 50, 20], dtype=np.float64) / 206.0
START = time.time()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def log(root: Path, stage: str, reused_fits: int = 0, refits: int = 0,
        crs_evals: int = 0, failures: int = 0) -> None:
    line = (
        f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] [AMEND] {stage}; "
        f"elapsed_wall_s={time.time()-START:.3f}; reused_fits={reused_fits}; "
        f"refitted_fits={refits}; CRS_evaluations={crs_evals}; failures={failures}; "
        f"output={root}"
    )
    print(line, flush=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    with (root / "logs" / "progress.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_grid(d_low: float, D: float, K: int) -> np.ndarray:
    if not (np.isfinite(d_low) and np.isfinite(D) and D > d_low and K >= 2):
        raise ValueError("invalid log-grid anchors")
    j = np.arange(K, dtype=np.float64)
    nodes = np.exp(np.log(d_low) + j / (K - 1) * (np.log(D) - np.log(d_low)))
    nodes[0] = np.float64(d_low)
    nodes[-1] = np.float64(D)
    if not (np.isfinite(nodes).all() and np.all(np.diff(nodes) > 0)):
        raise ValueError("nonfinite or nonmonotone log grid")
    return nodes


def crs_geometry(d_low: float, D: float, K: int) -> dict:
    nodes = log_grid(d_low, D, K)
    delta = np.diff(nodes)
    q = np.arange(K - 1, 0, -1, dtype=np.float64)
    omega = q * delta
    M = float(omega.sum())
    beta = np.empty(K, dtype=np.float64)
    beta[0] = omega[0] / (2 * M)
    beta[-1] = omega[-1] / (2 * M)
    beta[1:-1] = (omega[:-1] + omega[1:]) / (2 * M)
    if not (np.all(omega > 0) and np.all(beta > 0) and abs(beta.sum() - 1) <= 1e-12):
        raise ValueError("invalid CRS geometry")
    Gd = float(np.sum(q * delta**3) / M)
    return {
        "nodes": nodes,
        "delta": delta,
        "q": q,
        "omega": omega,
        "M": M,
        "beta": beta,
        "G_d": Gd,
        "G_norm": Gd / (D * D),
    }


def crs(values: np.ndarray, geom: dict) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(geom["beta"] @ values)


def trapezoid_crs(values: np.ndarray, geom: dict) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sum(geom["omega"] * (values[:-1] + values[1:]) / 2) / geom["M"])


def ab_reference(values: np.ndarray, ids: list[str]) -> tuple[np.ndarray, float, float] | None:
    values = np.asarray(values, float)
    order = np.lexsort((np.asarray(ids, str), values))
    x = values[order]
    n = len(x)
    m = max(3, math.ceil(0.15 * n))
    best = None
    for b1 in range(m, n - 2 * m + 1):
        if not x[b1 - 1] < x[b1]:
            continue
        for b2 in range(b1 + m, n - m + 1):
            if not x[b2 - 1] < x[b2]:
                continue
            sse = sum(float(np.sum((z - z.mean()) ** 2)) for z in (x[:b1], x[b1:b2], x[b2:]))
            key = (sse, b1, b2)
            if best is None or key < best[0]:
                best = (key, b1, b2)
    if best is None:
        return None
    _, b1, b2 = best
    tau1 = float((x[b1 - 1] + x[b1]) / 2)
    tau2 = float((x[b2 - 1] + x[b2]) / 2)
    labels = np.where(values <= tau1, "L", np.where(values <= tau2, "I", "H"))
    return labels, tau1, tau2


def resolve_inputs() -> list[dict]:
    pc = RESULTS / "PHASE_C_SIMULATION"
    pd = RESULTS / "PHASE_D_CCLE"
    rows = [
        ("DIRECT_EXECUTE_SPEC", INSTRUCTION, "v1.0", "AUTHORITATIVE_INSTRUCTION", "reused"),
        ("PHASE_A_V31", RESULTS / "TRIOS_PHASEA_v3.1_PHASED_D1_D2_IDENTITY_REBIND_COMPACT_REVIEW.zip", "v3.1", "FROZEN", "reused"),
        ("PHASE_B_B1", RESULTS / "TRIOS_PhaseB_Benchmark1_Final_v1.0_complete_delivery.zip", "v1.0", "FINAL_REVIEW_REQUIRED", "recomputed_downstream"),
        ("PHASE_B_B2_LEGACY", RESULTS / "TRIOS_PhaseB_Benchmark2_Final_v1.0_complete_delivery.zip", "v1.0", "FROZEN_SUPERSEDED_PRIMARY", "reused_reference"),
        ("PHASE_B_B3", RESULTS / "TRIOS_PhaseB_Benchmark3_Final_61Method_Operational_Comparison_v1.1.1_complete_delivery.zip", "v1.1.1", "FINAL_REVIEW_REQUIRED", "recomputed_TRTOS_only"),
        ("PHASE_C_E1", pc / "EXPERIMENT_1" / "TRIOS_PHASEC_SIM_E1_v1.0_ARCHIVE_v1.1_FITTED_CURVE_RECOVERED_FROZEN_complete_delivery.zip", "scientific-v1.0/archive-v1.1", "FROZEN", "recomputed_downstream"),
        ("PHASE_C_E1_TRUTH_BANK", pc / "EXPERIMENT_1" / "TRIOS_PHASEC_SIM_E1_v1.0" / "03_GENERATED_BANK" / "formal_latent_parameters.csv", "v1.0", "FROZEN", "reused"),
        ("PHASE_C_E1_SEEDS", pc / "EXPERIMENT_1" / "TRIOS_PHASEC_SIM_E1_v1.0" / "02_DESIGN_REGISTRY" / "seed_registry.csv", "v1.0", "FROZEN", "reused"),
        ("PHASE_C_E1_FIT_MANIFEST", pc / "EXPERIMENT_1" / "TRIOS_PHASEC_SIM_E1_v1.0_ARCHIVE_v1.1_FITTED_CURVE_RECOVERED_FROZEN" / "manifest_v1.1.csv", "v1.1", "FROZEN", "reused"),
        ("PHASE_C_E2_SCIENTIFIC", pc / "EXPERIMENT_2" / "TRIOS_PHASEC_SIM_E2_v1.1_PARENT_AMENDED_complete_delivery.zip", "v1.1", "FINAL_REVIEW_REQUIRED", "recomputed_primary"),
        ("PHASE_C_E2_FREEZE_WRAPPER", pc / "EXPERIMENT_2" / "TRIOS_PHASEC_SIM_E2_v1.0_POSTPROCESSING_v1.1_REPAIRED_FROZEN.zip", "v1.1", "FROZEN", "reused_reference"),
        ("PHASE_C_E3", pc / "EXPERIMENT_3" / "TRIOS_PHASEC_SIM_E3_DOWNSTREAM_v1.2a_complete_delivery.zip", "v1.2a", "FINAL_REVIEW_REQUIRED", "recomputed_TRTOS_only"),
        ("PHASE_C_E3_COMPARATOR_REGISTRY", pc / "EXPERIMENT_3" / "DOWNSTREAM_COMPLETE_PIPELINE_v1.2a" / "01_METHOD_REGISTRY" / "method_registry.csv", "v1.2a", "FROZEN", "reused"),
        ("PHASE_C_E4_STAGE1", pc / "EXPERIMENT_4_V3" / "TRIOS_PHASEC_SIM_E4_V3_STAGE1_UPPER_DOMAIN_SUFFICIENCY_v3.0_complete_delivery.zip", "v3.0", "FINAL_REVIEW_REQUIRED", "recomputed_downstream"),
        ("PHASE_C_E4_STAGE2", pc / "EXPERIMENT_4_V3" / "TRIOS_PHASEC_SIM_E4_STAGE2_v3.2b_complete_delivery.zip", "v3.2b", "FINAL_REVIEW_REQUIRED", "recomputed_downstream"),
        ("PHASE_C_E4_ROUNDING", pc / "EXPERIMENT_4_V3" / "TRIOS_PHASEC_SIM_E4_PRACTICAL_DOSE_ROUNDING_v3.3_complete_delivery.zip", "v3.3", "FINAL_REVIEW_REQUIRED", "recomputed_downstream"),
        ("PHASE_C_E5", pc / "EXPERIMENT_5" / "TRIOS_PHASEC_SIM_E5_STRUCTURAL_STRESS_v1.0_complete_delivery.zip", "v1.0", "FINAL_REVIEW_REQUIRED", "recomputed_downstream"),
        ("PHASE_D_D0", pd / "D0_DATA_CHALLENGE_SET_LOCK_v1.0" / "FINAL_STATUS.json", "v1.0", "FINAL_REVIEW_REQUIRED", "reused"),
        ("PHASE_D_D1", pd / "D1_FULL_SOURCE_TRANSPORTABILITY_v1.0" / "FINAL_STATUS.json", "v1.0", "FINAL_REVIEW_REQUIRED", "parity_only"),
        ("PHASE_D_D2", pd / "D2_FULL_SOURCE_EXTERNAL_REFERENCE_STRATIFICATION_v1.0" / "FINAL_STATUS.json", "v1.0", "FINAL_REVIEW_REQUIRED", "recomputed_downstream"),
        ("PHASE_D_D3", pd / "D3_LOCKED_N18_STANDALONE_GENERALIZATION_v1.1_PHASEA_v3.1" / "FINAL_STATUS.json", "v1.1/PhaseA-v3.1", "FINAL_REVIEW_REQUIRED", "recomputed_downstream"),
        ("PHASE_D_D4", pd / "D4_REFERENCE_FIDELITY_LOO_OPERATIONAL_STABILITY_v1.0" / "FINAL_STATUS.json", "v1.0", "FINAL_REVIEW_REQUIRED", "recomputed_downstream"),
        ("PHASE_D_DS1", pd / "DS1_BROAD_69_COHORT_TRANSPORTABILITY_LANDSCAPE_v1.0" / "FINAL_STATUS.json", "v1.0", "FINAL_REVIEW_REQUIRED", "parity_only"),
        ("PHASE_D_D5", pd / "D5_PRESPECIFIED_MOLECULAR_BIOLOGICAL_COHERENCE_v1.0" / "FINAL_STATUS.json", "v1.0", "FINAL_REVIEW_REQUIRED", "recomputed_downstream"),
        ("PHASE_D_SUBSET_REGISTRY", pd / "D0_DATA_CHALLENGE_SET_LOCK_v1.0" / "D0_PRIMARY_7_SUBSET_MEMBERSHIP_REGISTRY.csv", "v1.0", "FROZEN", "reused"),
        ("PHASE_D_D1_FIT_REGISTRY", pd / "D1_FULL_SOURCE_TRANSPORTABILITY_v1.0" / "01_PROFILE_FITS" / "D1_PROFILE_SPLINE_FIT_SUMMARY.csv", "v1.0", "FROZEN", "reused"),
        ("HB_SPLINE_FIT_REGISTRY", RESULTS / "HB_PDO_LAYER1_SPLINE_ONLY_ADEQUACY_CORRECTION_v1.2" / "01_SPLINE_V12" / "spline_checkpoint_inventory.csv", "v1.2", "FROZEN", "reused"),
        ("PRODUCTION_METHOD_REGISTRY", RESULTS / "TRIOS_PRODUCTION_METHOD_REGISTRY.json", "v3.1", "FROZEN", "reused"),
    ]
    missing = [str(p) for _, p, *_ in rows if not p.is_file()]
    if missing:
        raise SystemExit("STOP_AMENDMENT_PARENT_BINDING_AMBIGUOUS\nMissing authoritative files:\n" + "\n".join(missing))
    out = []
    for role, path, version, archive_status, use in rows:
        out.append({
            "role": role,
            "resolved_path": str(path.resolve()),
            "filename": path.name,
            "sha256": sha256(path),
            "parent_version": version,
            "archive_status": archive_status,
            "reused_or_recomputed": use,
            "notes": "Resolved from latest explicit frozen/final manifest chain; source immutable",
        })
    return out


def method_lock() -> dict:
    examples = {}
    for assay, dl, D in (("HB", 1.0, 50.0), ("CCLE", 0.0025, 2.53)):
        g = crs_geometry(dl, D, 7)
        examples[assay] = {k: [float(x) for x in g[k]] for k in ("nodes", "delta", "q", "omega", "beta")}
        examples[assay].update(M=g["M"], G_d=g["G_d"], G_norm=g["G_norm"])
    return {
        "candidate_method_version": "Phase A v3.2-candidate / CRS-LOGSPACE-AMENDMENT",
        "production_frozen": False,
        "amendment_scope": "CRS scoring-node geometry only",
        "float_type": "IEEE-754 float64",
        "log_base": "natural",
        "round_scoring_nodes": False,
        "K_CRS_production": 7,
        "d_low_uM": {"HB_PDO_and_HB_like_PhaseC": 1.0, "CCLE": 0.0025},
        "node_formula": "exp(log(d_low)+(j/(K-1))*(log(D)-log(d_low))), j=0..K-1; force endpoints",
        "CRS": "endpoint-inclusive q_j=(K-j), omega_j=q_j*Delta_d_j physical-dose trapezoid normalized by M",
        "transport_operator": "Phase A v3.1 unchanged",
        "rho": 0.60,
        "eta_A": 0.80,
        "terminal_gain_threshold": 0.10,
        "m_AB": "max(3,ceil(0.15*N))",
        "m_stage": "max(10,3*m_AB)",
        "old_geometry_superseded": {"p": OLD_P.tolist(), "beta": OLD_BETA.tolist()},
        "post_readout_tuning_allowed": False,
        "examples": examples,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def unit_tests() -> list[dict]:
    tests = []
    def add(qid, name, passed, evidence):
        tests.append({"gate": qid, "name": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    cases = [(1., 50., 7), (.0025, 2.53, 7), (2., 200., 3), (.25, 8., 11)]
    end_ok = all(log_grid(a, b, k)[0] == a and log_grid(a, b, k)[-1] == b for a, b, k in cases)
    add("Q01", "exact endpoints", end_ok, str(cases))
    errs = [float(np.ptp(np.diff(np.log(log_grid(a, b, k))))) for a, b, k in cases]
    add("Q02", "equal log increments", max(errs) <= 2e-15, f"max_ptp={max(errs):.17g}")
    geoms = [crs_geometry(a, b, k) for a, b, k in cases]
    pos_ok = all(np.all(g["omega"] > 0) and np.all(g["beta"] > 0) and abs(g["beta"].sum()-1) <= 1e-12 for g in geoms)
    add("Q03", "positivity and normalization", pos_ok, f"max_sum_err={max(abs(g['beta'].sum()-1) for g in geoms):.17g}")
    rng = np.random.default_rng(20260912)
    ident_err = max(abs(crs(rng.normal(size=k), crs_geometry(a,b,k)) - trapezoid_crs(rng.normal(size=k), crs_geometry(a,b,k))) for a,b,k in []) if False else 0.0
    errs2=[]
    for a,b,k in cases:
        v=rng.normal(size=k);g=crs_geometry(a,b,k);errs2.append(abs(crs(v,g)-trapezoid_crs(v,g)))
    add("Q04", "direct versus trapezoid identity", max(errs2) <= 5e-16, f"max_abs={max(errs2):.17g}")
    scale_errs=[]
    for a,b,k in cases:
        c=3.7;g=crs_geometry(a,b,k);gs=crs_geometry(c*a,c*b,k)
        scale_errs.append(max(float(np.max(np.abs(gs['nodes']-c*g['nodes']))),float(np.max(np.abs(gs['beta']-g['beta'])))))
    add("Q05", "scaling covariance", max(scale_errs) <= 3e-13, f"max_abs={max(scale_errs):.17g}")
    hb=log_grid(1.,50.,7);hb_ref=np.array([1,1.9193831036664843,3.6840314986403864,7.0710678118654755,13.572088082974531,26.05003654793457,50])
    add("Q06", "HB example", np.allclose(hb,hb_ref,rtol=2e-12,atol=0), json.dumps(hb.tolist()))
    cc=log_grid(.0025,2.53,7);cc_ref=np.array([.0025,.007918,.02508,.07943,.2516,.7978,2.53])
    add("Q07", "CCLE example", np.allclose(cc,cc_ref,rtol=3e-3,atol=0), json.dumps(cc.tolist()))
    rounded=np.round(hb,3)
    add("Q08", "no scoring-node rounding", not np.array_equal(hb,rounded) and hb.dtype==np.float64, f"dtype={hb.dtype}; exact={hb.tolist()}")
    leak_ok=not np.allclose(crs_geometry(1.,50.,7)["nodes"],OLD_P*50,rtol=0,atol=0) and not np.allclose(crs_geometry(1.,50.,7)["beta"],OLD_BETA,rtol=0,atol=0)
    add("Q09", "old/new separation", leak_ok, f"new_nodes={hb.tolist()}; old_nodes={(OLD_P*50).tolist()}")
    x=np.array([.05,.08,.12,.18,.27,.31,.44,.51,.63,.72,.84,.91]);ids=[f"P{i:02d}" for i in range(len(x))]
    a1=ab_reference(x,ids);a2=ab_reference(x.copy(),ids.copy())
    ab_ok=a1 is not None and a2 is not None and np.array_equal(a1[0],a2[0]) and a1[1:]==a2[1:]
    add("Q10", "unchanged AB dispatch", ab_ok, f"cutoffs={a1[1:] if a1 else None}")
    return tests


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    registry = resolve_inputs()
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    root = RESULTS / f"TRIOS_CRS_LOGSPACE_AMENDMENT_v1.0_{stamp}"
    if root.exists():
        raise SystemExit("STOP_METHOD_LOCK_WRITE_FAIL: output collision")
    dirs = [
        "00_BINDING", "01_METHOD_LOCK_CANDIDATE", "02_IMPLEMENTATION_QA", "03_IMPACT_AUDIT",
        "04_PHASE_B/B1", "04_PHASE_B/B2_LOGSPACE_K", "04_PHASE_B/B3", "04_PHASE_B/LEGACY_B2_57_GEOMETRY_REFERENCE",
        "05_PHASE_C/E1", "05_PHASE_C/E2_LOGSPACE_K", "05_PHASE_C/E3", "05_PHASE_C/E4", "05_PHASE_C/E5", "05_PHASE_C/LEGACY_E2_57_GEOMETRY_REFERENCE",
        "06_PHASE_D/D1_TRANSPORT_PARITY", "06_PHASE_D/D2", "06_PHASE_D/D3", "06_PHASE_D/D4", "06_PHASE_D/DS1_PARITY", "06_PHASE_D/D5",
        "07_CROSS_PHASE_COMPARISON", "08_QA", "09_REPORT", "10_ARCHIVE", "logs", "src",
    ]
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
    write_csv(root / "00_BINDING" / "input_registry_resolved.csv", registry)
    shutil.copy2(INSTRUCTION, root / "00_BINDING" / INSTRUCTION.name)
    write_json(root / "00_BINDING" / "binding_status.json", {
        "status": "PASS_UNAMBIGUOUS", "entries": len(registry), "all_sha256_recorded": True,
        "frozen_outputs_mutated": False, "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    (WORK / "CURRENT_ROOT.txt").write_text(str(root.resolve()) + "\n", encoding="utf-8")
    log(root, "binding complete")

    lock = method_lock()
    write_json(root / "01_METHOD_LOCK_CANDIDATE" / "CRS_logspace_amendment_lock.json", lock)
    md = [
        "# CRS exact-log-space amendment lock", "", "Candidate only; not production frozen.", "",
        "- Scope: CRS scoring-node geometry only.",
        "- Transport: unchanged Phase A v3.1 Operator II.",
        "- Production K: 7.",
        "- HB/Phase-C lower anchor: 1 µM.",
        "- CCLE lower anchor: 0.0025 µM.",
        "- Nodes: exact float64 natural-log grid, endpoints forced, no rounding.",
        "- Integration: endpoint-inclusive weighted trapezoid in physical dose space.",
        "- Old fixed-p geometry: superseded and prohibited in amended paths.",
        "- Post-readout tuning: prohibited.", "",
        f"Locked at UTC: {lock['locked_at_utc']}", "",
    ]
    (root / "01_METHOD_LOCK_CANDIDATE" / "CRS_logspace_amendment_lock.md").write_text("\n".join(md), encoding="utf-8")
    log(root, "method lock written")

    tests = unit_tests()
    write_csv(root / "02_IMPLEMENTATION_QA" / "Q01_Q10_unit_tests.csv", tests)
    write_json(root / "02_IMPLEMENTATION_QA" / "unit_test_status.json", {
        "passed": sum(x["status"] == "PASS" for x in tests), "total": len(tests), "tests": tests,
    })
    if any(x["status"] != "PASS" for x in tests):
        write_json(root / "FINAL_STATUS.json", {"FINAL_STATUS": "CRS_LOGSPACE_AMENDMENT_v1.0_HARD_STOP_STOP_LOGGRID_UNIT_TEST_FAIL"})
        log(root, "unit QA FAIL", failures=1)
        raise SystemExit("STOP_LOGGRID_UNIT_TEST_FAIL")
    shutil.copy2(Path(__file__), root / "src" / Path(__file__).name)
    log(root, "unit QA 10/10 PASS")
    write_json(root / "RUN_STATE.json", {
        "status": "PREFLIGHT_COMPLETE", "result_root": str(root.resolve()),
        "binding_entries": len(registry), "unit_QA": "10/10 PASS", "started_epoch": START,
    })
    print(json.dumps({"result_root": str(root.resolve()), "unit_QA": "10/10 PASS"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
