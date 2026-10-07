import numpy as np

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
