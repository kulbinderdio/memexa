"""
Lightweight PCA-to-2D projection using power iteration.
No scikit-learn dependency — only numpy.
"""

from __future__ import annotations

import numpy as np


def pca_2d(vectors: list[list[float]]) -> list[tuple[float, float]]:
    """Project a list of embedding vectors down to 2D via PCA.

    Uses power iteration (60 iterations) to find the top two principal
    components without a full SVD decomposition.

    Returns a list of (x, y) tuples normalised to [0, 1].
    Edge cases:
      - n < 2 vectors → returns [(0.5, 0.5)] * n
      - zero-norm vectors are handled safely (treated as zero vectors)
    """
    n = len(vectors)
    if n == 0:
        return []
    if n == 1:
        return [(0.5, 0.5)]

    X = np.array(vectors, dtype=np.float64)  # (n, d)

    # Handle zero-norm rows safely by replacing them with zeros
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    X = X / safe_norms  # unit-normalise each row

    # Mean-centre the matrix
    mean = X.mean(axis=0)
    X = X - mean  # (n, d)

    d = X.shape[1]

    if d < 2:
        # Degenerate: pad with zeros and fall through
        X = np.hstack([X, np.zeros((n, 2 - d))])
        d = 2

    # Covariance matrix C = X^T X / (n-1)  — power iteration on C
    C = X.T @ X / max(n - 1, 1)  # (d, d)

    components: list[np.ndarray] = []

    for _ in range(2):
        # Random initialisation
        rng = np.random.default_rng(seed=len(components))
        v = rng.standard_normal(d)

        # Deflate previous components from initial vector
        for pc in components:
            v = v - (v @ pc) * pc

        v /= np.linalg.norm(v) + 1e-12

        # Power iteration
        for _ in range(60):
            v = C @ v
            # Deflate already-found components
            for pc in components:
                v = v - (v @ pc) * pc
            norm = np.linalg.norm(v)
            if norm < 1e-12:
                # Degenerate direction — pick an orthogonal fallback
                fallback = np.zeros(d)
                fallback[len(components) % d] = 1.0
                for pc in components:
                    fallback = fallback - (fallback @ pc) * pc
                f_norm = np.linalg.norm(fallback)
                v = fallback / (f_norm + 1e-12)
                break
            v = v / norm

        components.append(v)

    pc1 = components[0]  # (d,)
    pc2 = components[1]  # (d,)

    # Project
    proj1 = X @ pc1  # (n,)
    proj2 = X @ pc2  # (n,)

    # Normalise each dimension to [0, 1]
    def normalise(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-12:
            return np.full_like(arr, 0.5)
        return (arr - lo) / (hi - lo)

    x_norm = normalise(proj1)
    y_norm = normalise(proj2)

    return [(float(x), float(y)) for x, y in zip(x_norm, y_norm)]
