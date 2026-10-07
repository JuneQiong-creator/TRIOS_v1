from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pygam.penalties import derivative
from pygam.utils import b_spline_basis
from scipy.optimize import lsq_linear


def _basis(z, edge_knots, n_splines=12, spline_order=3):
    return np.asarray(
        b_spline_basis(
            np.asarray(z, float),
            edge_knots=np.asarray(edge_knots, float),
            n_splines=n_splines,
            spline_order=spline_order,
            sparse=False,
            periodic=False,
            verbose=False,
        ),
        float,
    )


def _monotone_map(n_splines: int) -> np.ndarray:
    # c[0] is absorbed into the free intercept. Each remaining coefficient is
    # represented by cumulative non-negative first differences.
    mapping = np.zeros((n_splines, n_splines - 1), float)
    for row in range(1, n_splines):
        mapping[row, :row] = 1.0
    return mapping


@dataclass
class ConstrainedPSpline:
    intercept_: float
    differences_: np.ndarray
    coefficients_: np.ndarray
    edge_knots_: tuple[float, float]
    n_splines: int = 12
    spline_order: int = 3
    selected_lambda: float = 0.0
    effective_degrees_of_freedom: float = 0.0
    GCV: float = np.nan
    replicate_level_SSE: float = np.nan
    solver_status: str = ""
    solver_iterations: int = 0

    @property
    def coef_(self):
        # Same presentation order used by the v1.1 PyGAM object: spline
        # coefficients followed by the free intercept.
        return np.r_[self.coefficients_, self.intercept_]

    def predict_z(self, z) -> np.ndarray:
        basis = _basis(z, self.edge_knots_, self.n_splines, self.spline_order)
        return np.asarray(self.intercept_ + basis @ self.coefficients_, float)

    def predict(self, dose) -> np.ndarray:
        dose = np.asarray(dose, float)
        return self.predict_z(np.log1p(dose))


def fit_constrained_pspline(z, y, lam: float, edge_knots, n_splines=12, spline_order=3) -> ConstrainedPSpline:
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    basis = _basis(z, edge_knots, n_splines, spline_order)
    mapping = _monotone_map(n_splines)
    transformed = basis @ mapping
    design = np.column_stack([np.ones(len(z)), transformed])

    penalty = derivative(n_splines, np.zeros(n_splines), derivative=2, periodic=False).toarray()
    difference_penalty = mapping.T @ penalty @ mapping
    eigenvalues, eigenvectors = np.linalg.eigh((difference_penalty + difference_penalty.T) / 2.0)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    penalty_root = np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    augmented_design = np.vstack([
        design,
        np.column_stack([np.zeros(penalty_root.shape[0]), np.sqrt(lam) * penalty_root]),
    ])
    augmented_response = np.r_[y, np.zeros(penalty_root.shape[0])]
    lower = np.r_[-np.inf, np.zeros(n_splines - 1)]
    upper = np.full(n_splines, np.inf)
    solution = lsq_linear(
        augmented_design,
        augmented_response,
        bounds=(lower, upper),
        method="trf",
        tol=1e-12,
        lsq_solver="exact",
        lsmr_tol=None,
        max_iter=1000,
        verbose=0,
    )
    beta = np.asarray(solution.x, float)
    intercept = float(beta[0])
    differences = np.asarray(beta[1:], float)
    coefficients = mapping @ differences
    fitted = intercept + basis @ coefficients
    sse = float(np.sum((y - fitted) ** 2))

    # Active-set influence trace for the exact constrained solution. The GCV
    # scaling gamma=1.4 is the frozen PyGAM default used in v1.1.
    scale = max(1.0, float(np.max(np.abs(coefficients))))
    active = differences > 1e-9 * scale
    active_columns = np.r_[True, active]
    active_design = design[:, active_columns]
    full_penalty = np.zeros((n_splines, n_splines), float)
    full_penalty[1:, 1:] = difference_penalty
    active_penalty = full_penalty[np.ix_(active_columns, active_columns)]
    normal = active_design.T @ active_design + lam * active_penalty
    influence = active_design @ np.linalg.pinv(normal, rcond=1e-12) @ active_design.T
    edof = float(np.trace(influence))
    denominator = len(y) - 1.4 * edof
    gcv = float(len(y) * sse / denominator**2) if abs(denominator) > 1e-12 else np.inf
    return ConstrainedPSpline(
        intercept_=intercept,
        differences_=differences,
        coefficients_=coefficients,
        edge_knots_=(float(edge_knots[0]), float(edge_knots[1])),
        n_splines=n_splines,
        spline_order=spline_order,
        selected_lambda=float(lam),
        effective_degrees_of_freedom=edof,
        GCV=gcv,
        replicate_level_SSE=sse,
        solver_status=str(solution.message),
        solver_iterations=int(solution.nit),
    )
