from __future__ import annotations

import math

import numpy as np
from scipy.special import expit


def positive(value: float) -> float:
    return float(np.exp(np.clip(value, -700.0, 700.0)))


def raw_emax(d, theta):
    e0, a, k = float(theta[0]), positive(theta[1]), positive(theta[2])
    x = np.asarray(d, float)
    return e0 + a * x / (k + x)


def raw_hill(d, theta):
    e0, a, k, h = float(theta[0]), positive(theta[1]), positive(theta[2]), positive(theta[3])
    x = np.asarray(d, float)
    out = np.zeros_like(x)
    mask = x > 0
    out[mask] = expit(h * (np.log(x[mask]) - math.log(k)))
    return e0 + a * out


def raw_logistic(d, theta):
    low, a, c, s = float(theta[0]), positive(theta[1]), float(theta[2]), positive(theta[3])
    x = np.asarray(d, float)
    return low + a * expit((x - c) / s)


def raw_weibull(d, theta):
    e0, a, lam, p = float(theta[0]), positive(theta[1]), positive(theta[2]), positive(theta[3])
    x = np.asarray(d, float)
    effect = np.zeros_like(x)
    mask = x > 0
    log_power = p * (np.log(x[mask]) - math.log(lam))
    transformed = np.where(log_power > 40.0, 1.0, -np.expm1(-np.exp(np.clip(log_power, -745.0, 40.0))))
    effect[mask] = transformed
    return e0 + a * effect


MODEL_FUNCTIONS = {"RAW_EMAX": raw_emax, "RAW_HILL": raw_hill, "RAW_LOGISTIC": raw_logistic, "RAW_WEIBULL": raw_weibull}
PARAMETER_NAMES = {
    "RAW_EMAX": ["E0", "log_A", "log_K"],
    "RAW_HILL": ["E0", "log_A", "log_K", "log_h"],
    "RAW_LOGISTIC": ["L", "log_A", "C", "log_S"],
    "RAW_WEIBULL": ["E0", "log_A", "log_lambda", "log_p"],
}
PARAMETRIC_KEFF = {"RAW_EMAX": 3.0, "RAW_HILL": 4.0, "RAW_LOGISTIC": 4.0, "RAW_WEIBULL": 4.0}


def central_initialization(model: str, dose: np.ndarray, response: np.ndarray) -> np.ndarray:
    zero = response[dose == 0]
    baseline = float(np.median(zero)) if len(zero) else float(np.median(response[dose == np.min(dose)]))
    amplitude = max(float(np.max(response) - np.min(response)), float(np.std(response, ddof=1)) if len(response) > 1 else 0.1, 0.05)
    positive_dose = dose[dose > 0]
    scale = float(np.median(positive_dose)) if len(positive_dose) else 1.0
    spread = max(float(np.max(dose) - np.min(dose)), scale, 1e-6)
    if model == "RAW_EMAX":
        return np.asarray([baseline, math.log(amplitude), math.log(scale)])
    if model == "RAW_HILL":
        return np.asarray([baseline, math.log(amplitude), math.log(scale), 0.0])
    if model == "RAW_LOGISTIC":
        return np.asarray([baseline, math.log(amplitude), scale, math.log(spread / 4.0)])
    if model == "RAW_WEIBULL":
        return np.asarray([baseline, math.log(amplitude), math.log(scale), 0.0])
    raise KeyError(model)

