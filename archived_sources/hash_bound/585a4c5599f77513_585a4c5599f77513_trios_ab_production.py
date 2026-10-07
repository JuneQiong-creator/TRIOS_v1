"""Current AB objective selector: binary64 fast path, exact dyadic fallback.

Only the ordering of already-admissible candidate cuts is resolved here.
Candidate generation, m_AB, scoring and labels remain at their entrypoints.
"""
from fractions import Fraction
import numpy as np


def _exact_sse(sorted_scores, a, b):
    x = [Fraction.from_float(float(v)) for v in sorted_scores]
    def within(z):
        return sum(v*v for v in z) - sum(z)**2 / len(z)
    return within(x[:a]) + within(x[a:b]) + within(x[b:])


def select_ab_candidate(sorted_scores, candidates):
    """Return one existing (float_objective, cut1, cut2) candidate."""
    if not candidates:
        return None
    first = min(candidates, key=lambda z: (z[0], z[1], z[2]))
    x = np.asarray(sorted_scores, dtype=np.float64)
    energy = float(np.sum(np.square(x, dtype=np.float64)))
    guard = 128*np.finfo(float).eps*max(1., abs(first[0]), energy)
    possible = [z for z in candidates if z[0]-first[0] <= guard]
    if len(possible) == 1:
        return first
    # The guard only selects cases to recompute; it never declares a tie.
    return min(possible, key=lambda z: (_exact_sse(x,z[1],z[2]),z[1],z[2]))
