from __future__ import annotations

import hashlib
import math
import sys

import numpy as np

from .config_frozen import CLASSIFICATION_TOL, CLASSIFIER_STARTS, MASTER_SEED, PYDEPS

sys.path.append(str(PYDEPS))
from scipy.cluster.hierarchy import fcluster, linkage


GROUP_ORDER = {"L": 0, "I": 1, "H": 2}


def child_seed(context: str, method: str, start_id: int) -> int:
    token = "|".join([str(MASTER_SEED), "layer1_v1.1_classifier", context, method, str(start_id)])
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big", signed=False)


def stable_order(values, profiles):
    return np.asarray(sorted(range(len(values)), key=lambda i: (float(values[i]), str(profiles[i]))), int)


def classify_threshold(values, tau1, tau2):
    values = np.asarray(values, float)
    labels = np.full(len(values), "I", object)
    labels[values <= tau1] = "L"
    labels[values >= tau2] = "H"
    return labels.astype(str)


def project_native(values, native, profiles):
    x = np.asarray(values, float)
    native = np.asarray(native)
    unique = list(dict.fromkeys(native.tolist()))
    if len(unique) != 3 or any(np.sum(native == c) == 0 for c in unique):
        return {"valid": False, "failure_code": "THREE_NONEMPTY_GROUPS_REQUIRED"}
    means = {c: float(np.mean(x[native == c])) for c in unique}
    ordered = sorted(unique, key=lambda c: (means[c], str(c)))
    labels = np.asarray(["L" if c == ordered[0] else "I" if c == ordered[1] else "H" for c in native], str)
    order = stable_order(x, profiles)
    if [GROUP_ORDER[v] for v in labels[order]] != sorted(GROUP_ORDER[v] for v in labels[order]):
        return {"valid": False, "failure_code": "NONCONTIGUOUS_NATIVE_GROUPS"}
    low, middle, high = x[labels == "L"], x[labels == "I"], x[labels == "H"]
    if not (np.max(low) < np.min(middle) and np.max(middle) < np.min(high)):
        return {"valid": False, "failure_code": "EXACT_TIE_OR_OVERLAP_PREVENTS_CUTOFF"}
    tau1 = float((np.max(low) + np.min(middle)) / 2.0)
    tau2 = float((np.max(middle) + np.min(high)) / 2.0)
    reproduced = classify_threshold(x, tau1, tau2)
    if not np.array_equal(reproduced, labels):
        return {"valid": False, "failure_code": "THRESHOLD_PROJECTION_INVALID"}
    counts = [int(np.sum(labels == g)) for g in ["L", "I", "H"]]
    return {"valid": True, "failure_code": "", "labels": labels, "tau1": tau1, "tau2": tau2, "group_n_L": counts[0], "group_n_I": counts[1], "group_n_H": counts[2], "minimum_group_size": min(counts), "singleton_count": sum(c == 1 for c in counts), "no_exact_tie_split": True, "cutoffs_reproduce_labels": True}


def kmeans_start(values, seed):
    x = np.asarray(values, float)
    unique = np.unique(x)
    if len(unique) < 3:
        return {"valid": False, "failure_code": "UNIQUE_SCORE_LT_3"}
    rng = np.random.Generator(np.random.PCG64(seed))
    initial = rng.choice(unique, 3, replace=False).astype(float)
    centers = initial.copy()
    for iteration in range(1, 1001):
        labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
        counts = np.bincount(labels, minlength=3)
        if np.any(counts == 0):
            return {"valid": False, "failure_code": "EMPTY_GROUP", "initial_centers": initial, "iterations": iteration}
        new = np.asarray([np.mean(x[labels == j]) for j in range(3)], float)
        shift = float(np.max(np.abs(new - centers)))
        centers = new
        if shift <= 1e-8:
            inertia = float(np.sum((x - centers[labels]) ** 2))
            return {"valid": True, "failure_code": "", "initial_centers": initial, "centers": centers, "native": labels, "iterations": iteration, "inertia": inertia, "raw_sizes": counts}
    return {"valid": False, "failure_code": "MAX_ITER", "initial_centers": initial, "iterations": 1000}


def fit_kmeans(values, profiles, context):
    starts, candidates = [], []
    for start_id in range(1, CLASSIFIER_STARTS + 1):
        seed = child_seed(context, "KMEANS", start_id)
        start = kmeans_start(values, seed)
        projection = project_native(values, start["native"], profiles) if start.get("valid") else {"valid": False, "failure_code": start.get("failure_code", "START_FAILURE")}
        valid = bool(start.get("valid") and projection.get("valid"))
        starts.append({"start_id": start_id, "seed": seed, "initial_centers": "|".join(format(v, ".17g") for v in start.get("initial_centers", [])), "converged": bool(start.get("valid", False)), "iterations": start.get("iterations", 0), "inertia": start.get("inertia", np.nan), "raw_group_sizes": "|".join(str(int(v)) for v in start.get("raw_sizes", [])), "projection_valid": bool(projection.get("valid")), "valid_start": valid, "failure_code": "" if valid else projection.get("failure_code", "START_FAILURE"), "selected": False})
        if valid:
            candidates.append((start["inertia"], start_id, projection))
    if not candidates:
        return {"valid": False, "failure_code": "ALL_STARTS_FAILED"}, starts
    minimum = min(v[0] for v in candidates)
    eligible = [v for v in candidates if abs(v[0] - minimum) <= CLASSIFICATION_TOL * max(1.0, abs(v[0]), abs(minimum))]
    _, selected, result = min(eligible, key=lambda v: v[1])
    starts[selected - 1]["selected"] = True
    return {"valid": True, "selected_start_id": selected, **result}, starts


def fit_ward(values, profiles):
    try:
        matrix = linkage(np.asarray(values, float).reshape(-1, 1), method="ward", metric="euclidean", optimal_ordering=True)
        native = fcluster(matrix, t=3, criterion="maxclust")
        result = project_native(values, native, profiles)
        return {"linkage_rows": len(matrix), "clusters_returned": len(np.unique(native)), "implementation": "scipy.cluster.hierarchy.linkage(method=ward,metric=euclidean,optimal_ordering=True)", **result}
    except Exception as exc:
        return {"valid": False, "failure_code": f"WARD_EXCEPTION:{type(exc).__name__}"}


def logsumexp(matrix):
    maximum = np.max(matrix, axis=1, keepdims=True)
    return maximum[:, 0] + np.log(np.sum(np.exp(matrix - maximum), axis=1))


def gmm_start(values, seed):
    x = np.asarray(values, float)
    init = kmeans_start(x, seed)
    if not init.get("valid"):
        return {"valid": False, "failure_code": "KMEANS_INITIALIZATION_FAILED"}
    means = init["centers"].copy()
    assigned = init["native"]
    weights = np.bincount(assigned, minlength=3).astype(float) / len(x)
    variance = max(float(np.mean((x - means[assigned]) ** 2)), 1e-6)
    previous = -np.inf
    for iteration in range(1, 1001):
        logp = np.log(np.maximum(weights, 1e-300))[None, :] - 0.5 * (math.log(2 * math.pi * variance) + (x[:, None] - means[None, :]) ** 2 / variance)
        rowll = logsumexp(logp)
        total = float(rowll.sum())
        resp = np.exp(logp - rowll[:, None])
        nk = resp.sum(axis=0)
        if np.any(nk <= 1e-12):
            return {"valid": False, "failure_code": "EMPTY_COMPONENT", "iterations": iteration}
        weights = nk / len(x)
        means = (resp * x[:, None]).sum(axis=0) / nk
        variance = max(float(np.sum(resp * (x[:, None] - means[None, :]) ** 2) / len(x)), 1e-6)
        if np.isfinite(previous) and abs(total - previous) <= 1e-6:
            logp = np.log(np.maximum(weights, 1e-300))[None, :] - 0.5 * (math.log(2 * math.pi * variance) + (x[:, None] - means[None, :]) ** 2 / variance)
            rowll = logsumexp(logp)
            assigned = np.argmax(logp, axis=1)
            counts = np.bincount(assigned, minlength=3)
            return {"valid": bool(np.all(counts > 0)), "failure_code": "" if np.all(counts > 0) else "EMPTY_ASSIGNED_COMPONENT", "iterations": iteration, "total_log_likelihood": float(rowll.sum()), "native": assigned, "means": means, "weights": weights, "variance": variance, "raw_sizes": counts}
        previous = total
    return {"valid": False, "failure_code": "MAX_ITER", "iterations": 1000}


def fit_gmm(values, profiles, context):
    starts, candidates = [], []
    for start_id in range(1, CLASSIFIER_STARTS + 1):
        seed = child_seed(context, "TIED_GMM", start_id)
        start = gmm_start(values, seed)
        projection = project_native(values, start["native"], profiles) if start.get("valid") else {"valid": False, "failure_code": start.get("failure_code", "START_FAILURE")}
        valid = bool(start.get("valid") and projection.get("valid"))
        starts.append({"start_id": start_id, "seed": seed, "converged": bool(start.get("valid")), "iterations": start.get("iterations", 0), "total_log_likelihood": start.get("total_log_likelihood", np.nan), "component_means": "|".join(format(v, ".17g") for v in start.get("means", [])), "shared_variance": start.get("variance", np.nan), "mixing_proportions": "|".join(format(v, ".17g") for v in start.get("weights", [])), "raw_group_sizes": "|".join(str(int(v)) for v in start.get("raw_sizes", [])), "projection_valid": bool(projection.get("valid")), "valid_start": valid, "failure_code": "" if valid else projection.get("failure_code", "START_FAILURE"), "selected": False})
        if valid:
            candidates.append((-start["total_log_likelihood"], start_id, projection))
    if not candidates:
        return {"valid": False, "failure_code": "ALL_STARTS_FAILED"}, starts
    _, selected, result = min(candidates, key=lambda v: (v[0], v[1]))
    starts[selected - 1]["selected"] = True
    return {"valid": True, "selected_start_id": selected, **result}, starts


def fit_onesd(values, profiles):
    x = np.asarray(values, float)
    mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    tau1, tau2 = mu - sd, mu + sd
    if not np.isfinite([tau1, tau2]).all() or not tau1 < tau2:
        return {"valid": False, "failure_code": "NONFINITE_OR_UNORDERED_CUTOFF", "mu": mu, "sample_sd_ddof1": sd}
    labels = classify_threshold(x, tau1, tau2)
    counts = [int(np.sum(labels == g)) for g in ["L", "I", "H"]]
    if min(counts) == 0:
        return {"valid": False, "failure_code": "THREE_NONEMPTY_GROUPS_REQUIRED", "mu": mu, "sample_sd_ddof1": sd}
    return {"valid": True, "failure_code": "", "labels": labels, "tau1": tau1, "tau2": tau2, "group_n_L": counts[0], "group_n_I": counts[1], "group_n_H": counts[2], "minimum_group_size": min(counts), "singleton_count": sum(c == 1 for c in counts), "no_exact_tie_split": True, "cutoffs_reproduce_labels": bool(np.array_equal(labels, classify_threshold(x, tau1, tau2))), "mu": mu, "sample_sd_ddof1": sd}


def fit_tertiles(values, profiles):
    x = np.asarray(values, float)
    n = len(x)
    order = stable_order(x, profiles)
    sx = x[order]
    k1, k2 = int(math.floor(n / 3)), int(math.floor(2 * n / 3))
    if k1 <= 0 or k2 <= k1 or k2 >= n:
        return {"valid": False, "failure_code": "TERTILE_TARGET_INVALID", "target_k1": k1, "target_k2": k2}
    if not sx[k1 - 1] < sx[k1] or not sx[k2 - 1] < sx[k2]:
        return {"valid": False, "failure_code": "TERTILE_TARGET_EXACT_TIE", "target_k1": k1, "target_k2": k2}
    native = np.empty(n, int); native[order[:k1]] = 0; native[order[k1:k2]] = 1; native[order[k2:]] = 2
    return {"target_k1": k1, "target_k2": k2, "boundary_shift_used": False, **project_native(x, native, profiles)}


def fit_jenks(values, profiles):
    x = np.asarray(values, float)
    n = len(x)
    order = stable_order(x, profiles)
    sx = x[order]
    candidates = []
    for k1 in range(1, n - 1):
        if not sx[k1 - 1] < sx[k1]:
            continue
        for k2 in range(k1 + 1, n):
            if not sx[k2 - 1] < sx[k2]:
                continue
            groups = [sx[:k1], sx[k1:k2], sx[k2:]]
            objective = float(sum(np.sum((g - np.mean(g)) ** 2) for g in groups))
            candidates.append((objective, -min(k1, k2-k1, n-k2), k1, k2))
    if not candidates:
        return {"valid": False, "failure_code": "NO_LEGAL_JENKS_PARTITION", "candidate_count": 0}
    minimum = min(v[0] for v in candidates)
    eligible = [v for v in candidates if abs(v[0] - minimum) <= CLASSIFICATION_TOL * max(1.0, abs(v[0]), abs(minimum))]
    objective, _, k1, k2 = min(eligible, key=lambda v: (v[1], v[2], v[3]))
    native = np.empty(n, int); native[order[:k1]] = 0; native[order[k1:k2]] = 1; native[order[k2:]] = 2
    return {"candidate_count": len(candidates), "selected_k1": k1, "selected_k2": k2, "objective_SSE": objective, **project_native(x, native, profiles)}


def quality_metrics(values, labels):
    x = np.asarray(values, float); labels = np.asarray(labels, str)
    counts = np.asarray([np.sum(labels == g) for g in ["L", "I", "H"]], float)
    proportions = counts / len(x)
    bbal = float(1.0 - 1.5 * np.sum((proportions - 1/3) ** 2))
    grand = float(np.mean(x))
    within = float(sum(np.sum((x[labels == g] - np.mean(x[labels == g])) ** 2) for g in ["L", "I", "H"]))
    between = float(sum(np.sum(labels == g) * (np.mean(x[labels == g]) - grand) ** 2 for g in ["L", "I", "H"]))
    total = within + between
    sbw = np.nan if total <= 0 else between / total
    return {"B_bal": bbal, "S_BW": float(sbw)}


def fit_classifier(values, profiles, method, context):
    if method == "KMEANS":
        return fit_kmeans(values, profiles, context)
    if method == "WARD":
        return fit_ward(values, profiles), []
    if method == "TIED_GMM":
        return fit_gmm(values, profiles, context)
    if method == "ONESD":
        return fit_onesd(values, profiles), []
    if method == "TERTILES":
        return fit_tertiles(values, profiles), []
    if method == "JENKS":
        return fit_jenks(values, profiles), []
    raise KeyError(method)

