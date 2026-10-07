from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import expit
from scipy.stats import beta

ROOT = Path(r"SOURCE_PROJECT")
OUT = ROOT / "04_RESULTS/PHASE_C_SIMULATION/EXPERIMENT_1/TRIOS_PHASEC_SIM_E1_v1.0"
CAL = OUT / "01_CALIBRATION"
MASTER_SEED = 42
NS = "TRIOS_PHASEC_SIM_E1_CALIBRATION_V1"
DOSES = np.r_[0.0, 250.0 ** (np.arange(7) / 6.0)]
CANDIDATES = np.array([30., 50., 100., 150., 200., 250.])
CENTERS = np.geomspace(3., 15., 121)
DELTAS = np.round(np.arange(.50, 6.0001, .05), 2)
SEPS = ("weak", "moderate", "strong")
FAMILIES = ("EMAX", "HILL", "LOGISTIC", "WEIBULL", "NP_PCHIP")
A_STAR = 0.810818275087736
T_STAR = 0.09640876007717228


def rng(*parts: object) -> np.random.Generator:
    key = "|".join(map(str, (MASTER_SEED, NS, *parts))).encode()
    words = np.frombuffer(hashlib.sha256(key).digest(), dtype=">u4")
    return np.random.default_rng(np.random.SeedSequence(words.tolist()))


def make_bank(sep: str) -> list[dict[str, np.ndarray]]:
    out = []
    truth = np.array(list("LLLLLLIIIIIIHHHHHH"))
    for b in range(1, 201):
        rr = rng(sep, b, "truth_shuffle")
        g = truth.copy(); rr.shuffle(g)
        xi = rng(sep, b, "location_z").normal(size=18)
        u = rng(sep, b, "amax_quantile").uniform(size=18)
        amp = beta.ppf(u, 17, 3)
        zshape = rng(sep, b, "shape_z").normal(size=18)
        znp = rng(sep, b, "np_shape_z").normal(size=(18, 8))
        knots = np.array([0., .25, .5, 1., 2., 4., 8., 16., 32.])
        base = knots ** 1.5 / (1 + knots ** 1.5); base[0] = 0
        inc = np.diff(base)[None, :] * np.exp(.25 * znp)
        inc /= inc.sum(axis=1, keepdims=True)
        ky = np.c_[np.zeros(18), np.cumsum(inc, axis=1)]
        # Frozen profile PCHIP objects are candidate-independent. Caching changes
        # scheduling cost only; knots, ordinates and SciPy evaluation are exact.
        np_pchip = [PchipInterpolator(knots, ky[i], extrapolate=False) for i in range(18)]
        out.append({"g": g, "xi": xi, "amp": amp, "zshape": zshape, "znp": znp,
                    "np_pchip": np_pchip})
    return out


BANK = {s: make_bank(s) for s in SEPS}


def curves(family: str, center: float, delta: float, block: dict, doses: np.ndarray) -> np.ndarray:
    g, xi, amp = block["g"], block["xi"], block["amp"]
    offsets = np.where(g == "H", -.4 * delta, np.where(g == "L", .4 * delta, 0.))
    K = np.exp(math.log(center) + offsets + .4 * xi)
    r = doses[None, :] / K[:, None]
    if family == "EMAX":
        y = r / (1 + r)
    elif family == "HILL":
        h = np.exp(math.log(1.5) + .2 * block["zshape"])
        rh = r ** h[:, None]; y = rh / (1 + rh)
    elif family == "LOGISTIC":
        k = np.exp(math.log(1.13519432989465) + .2 * block["zshape"])
        lo = expit(-k)[:, None]
        y = (expit(k[:, None] * (r - 1)) - lo) / (1 - lo)
    elif family == "WEIBULL":
        p = np.exp(math.log(.75 / math.log(2)) + .2 * block["zshape"])
        y = 1 - np.exp(-math.log(2) * r ** p[:, None])
    elif family == "NP_PCHIP":
        y = np.empty_like(r)
        for i in range(18):
            y[i] = block["np_pchip"][i](np.minimum(r[i], 32))
    else:
        raise ValueError(family)
    return amp[:, None] * y


def obs_sep(family: str, center: float, delta: float, sep: str) -> float:
    vals = []
    for b in BANK[sep]:
        x = curves(family, center, delta, b, DOSES)
        overall = x.mean(axis=0)
        sb = sw = 0.
        for lab in "LIH":
            z = x[b["g"] == lab]; cen = z.mean(axis=0)
            sb += len(z) * float(np.sum((cen - overall) ** 2))
            sw += float(np.sum((z - cen) ** 2))
        vals.append(sb / sw)
    return float(np.median(vals))


def stage_summary(family: str, center: float, ds: dict[str, float]) -> dict[str, float]:
    ae50, te50, stages = [], [], []
    eval_d = np.unique(np.r_[0., CANDIDATES, .6 * CANDIDATES, 250.])
    for sep in SEPS:
        for b in BANK[sep]:
            x = curves(family, center, ds[sep], b, eval_d)
            ix = {float(d): j for j, d in enumerate(eval_d)}
            den = x[:, ix[250.]] - x[:, ix[0.]]
            if (den <= 0).any() or not np.isfinite(den).all():
                stages.append(np.nan); ae50.append(np.nan); te50.append(np.nan); continue
            passed = []
            for u in CANDIDATES:
                a = np.median((x[:, ix[float(u)]] - x[:, ix[0.]]) / den)
                t = np.median(np.abs(x[:, ix[float(u)]] - x[:, ix[float(.6*u)]]) / den)
                if u == 50: ae50.append(float(a)); te50.append(float(t))
                if a >= .80 and t <= .10: passed.append(float(u))
            stages.append(min(passed) if passed else np.nan)
    st = np.asarray(stages)
    p30, p50 = np.mean(st == 30), np.mean(st == 50)
    pgt50 = np.mean(np.isfinite(st) & (st > 50)); pnp = np.mean(~np.isfinite(st))
    ma, mt = float(np.nanmedian(ae50)), float(np.nanmedian(te50))
    J = (ma-A_STAR)**2 + (mt-T_STAR)**2 + (p30-1/3)**2 + (p50-2/3)**2
    return {"center": center, "median_Ae50": ma, "median_Te50": mt,
            "p30": p30, "p50": p50, "p_gt50": pgt50, "p_no_pass": pnp,
            "J_stage": J, "gates_pass": bool(p30+p50 >= .80 and p50 >= .50 and pgt50 <= .20 and pnp <= .05)}


def choose_center(family: str, ds: dict[str, float]) -> tuple[float | None, pd.DataFrame]:
    tab = pd.DataFrame([stage_summary(family, float(c), ds) for c in CENTERS])
    ok = tab[tab.gates_pass].copy()
    if ok.empty: return None, tab
    ok["tie_distance"] = np.abs(ok.center - math.sqrt(50))
    row = ok.sort_values(["J_stage", "tie_distance", "center"], kind="stable").iloc[0]
    return float(row.center), tab


def choose_seps(family: str, center: float, targets: dict[str, float]) -> dict[str, float]:
    raw = {s: [(float(d), obs_sep(family, center, float(d), s)) for d in DELTAS] for s in SEPS}
    best = None
    # Joint exhaustive monotone choice, deterministic lexicographic lower-delta tie break.
    for dw, vw in raw["weak"]:
        for dm, vm in raw["moderate"]:
            if dm <= dw: continue
            for ds, vs in raw["strong"]:
                if ds <= dm: continue
                loss = abs(vw-targets["weak"]) + abs(vm-targets["moderate"]) + abs(vs-targets["strong"])
                z = (loss, dw, dm, ds)
                if best is None or z < best: best = z
    return dict(zip(SEPS, best[1:]))


def main() -> None:
    CAL.mkdir(parents=True, exist_ok=True)
    hill_ds = dict(zip(SEPS, (2., 3., 4.)))
    hill_center, hill_tab = choose_center("HILL", hill_ds)
    hill_tab.to_csv(CAL / "hill_center_search.csv", index=False, float_format="%.17g")
    if hill_center is None:
        (CAL / "HARD_STOP.json").write_text(json.dumps({"STOP_CODE":"STOP_E1_HILL_STAGE_CALIBRATION_FAIL"}, indent=2))
        raise SystemExit("STOP_E1_HILL_STAGE_CALIBRATION_FAIL")
    targets = {s: obs_sep("HILL", hill_center, hill_ds[s], s) for s in SEPS}
    locks = {"HILL": {"center": hill_center, "deltas": hill_ds}}
    audits, summaries = [], []
    for fam in ("EMAX", "LOGISTIC", "WEIBULL", "NP_PCHIP"):
        center = hill_center; ds = dict(hill_ds); converged = False
        rows = []
        for it in range(1, 6):
            new_ds = choose_seps(fam, center, targets)
            new_center, tab = choose_center(fam, new_ds)
            if new_center is None:
                rows.append({"iteration":it,"status":"NO_STAGE_CENTER", "center_before":center, **{f"delta_{s}":new_ds[s] for s in SEPS}})
                break
            rows.append({"iteration":it,"status":"PASS", "center_before":center,"center_after":new_center, **{f"delta_{s}":new_ds[s] for s in SEPS}})
            if np.isclose(new_center, center, rtol=0, atol=0) and new_ds == ds:
                converged = True; center, ds = new_center, new_ds; break
            center, ds = new_center, new_ds
        pd.DataFrame(rows).to_csv(CAL / f"{fam.lower()}_coordinate_descent_audit.csv", index=False, float_format="%.17g")
        if not converged:
            (CAL / "HARD_STOP.json").write_text(json.dumps({"STOP_CODE":"STOP_E1_FAMILY_CALIBRATION_NONCONVERGENCE","family":fam}, indent=2))
            raise SystemExit(f"STOP_E1_FAMILY_CALIBRATION_NONCONVERGENCE:{fam}")
        diffs = {s: obs_sep(fam, center, ds[s], s) for s in SEPS}
        if any(abs(diffs[s]-targets[s])/targets[s] > .05 for s in SEPS):
            (CAL / "HARD_STOP.json").write_text(json.dumps({"STOP_CODE":"STOP_E1_OBSERVABLE_DIFFICULTY_MISMATCH","family":fam,"observed":diffs,"targets":targets}, indent=2))
            raise SystemExit(f"STOP_E1_OBSERVABLE_DIFFICULTY_MISMATCH:{fam}")
        locks[fam] = {"center":center,"deltas":ds}
    for fam, spec in locks.items():
        st = stage_summary(fam, spec["center"], spec["deltas"])
        for s in SEPS:
            summaries.append({"family":fam,"separation":s,"center":spec["center"],"delta":spec["deltas"][s],"median_D_obs":obs_sep(fam,spec["center"],spec["deltas"][s],s),"hill_target_D_obs":targets[s],**st})
    lock = {"version":"v1.0","master_seed":MASTER_SEED,"namespace":NS,"B_CAL":200,"dose_grid":DOSES.tolist(),"families":locks,"hill_D_obs_targets":targets,"status":"PASS"}
    (CAL / "generator_lock_v1.0.json").write_text(json.dumps(lock, indent=2)+"\n")
    pd.DataFrame(summaries).to_csv(CAL / "generator_calibration_summary.csv", index=False, float_format="%.17g")
    pd.DataFrame([{"separation":s,"hill_target_D_obs":targets[s]} for s in SEPS]).to_csv(CAL / "observable_difficulty_targets.csv",index=False,float_format="%.17g")
    print(json.dumps(lock, indent=2), flush=True)

if __name__ == "__main__": main()
