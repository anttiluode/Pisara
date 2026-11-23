"""Forecast scores, baselines and Shaw's stored information."""
import numpy as np


def ridge_fit(X, y, lam=1e-3):
    """Ridge on standardised columns; lam is relative to the mean eigenvalue."""
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    A = Z.T @ Z
    A += lam * np.trace(A) / len(A) * np.eye(len(A))
    w = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return (w, mu, sd, y.mean())


def ridge_pred(model, X):
    w, mu, sd, ym = model
    return ((X - mu) / sd) @ w + ym


def nmse(pred, y):
    """Error relative to the variance of the target: 1.0 = no better than the mean."""
    return float(np.mean((pred - y) ** 2) / np.var(y))


def split(n, train_frac=0.6, val_frac=0.0):
    a = int(n * train_frac)
    return np.arange(a), np.arange(a, n)


def lagged(T, k, quad=False):
    """Rows n: [T_n, T_{n-1}, ..., T_{n-k+1}] (and pairwise products if quad).
    Target is T_{n+1}. Row n is usable for n >= k-1 and n+1 < len(T)."""
    idx = np.arange(k - 1, len(T) - 1)
    X = np.stack([T[idx - j] for j in range(k)], 1)
    if quad:
        P = [X[:, i] * X[:, j] for i in range(k) for j in range(i, k)]
        X = np.c_[X, np.stack(P, 1)]
    return X, T[idx + 1], idx


def stored_information(T, bins=15, lag=1):
    """Shaw's I(T_{n+lag}; T_n) in bits from an equiprobable-bin histogram,
    minus the same estimate on a shuffled copy (finite-sample bias)."""
    def mi(a, b):
        qa = np.quantile(a, np.linspace(0, 1, bins + 1)[1:-1])
        qb = np.quantile(b, np.linspace(0, 1, bins + 1)[1:-1])
        ia, ib = np.searchsorted(qa, a), np.searchsorted(qb, b)
        P = np.zeros((bins, bins))
        np.add.at(P, (ia, ib), 1)
        P /= P.sum()
        pa, pb = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
        nz = P > 0
        return float(np.sum(P[nz] * np.log2(P[nz] / (pa @ pb)[nz])))
    a, b = T[:-lag], T[lag:]
    rng = np.random.default_rng(12345)
    bias = np.mean([mi(a, rng.permutation(b)) for _ in range(5)])
    return max(mi(a, b) - bias, 0.0)


def interval_baselines(T, train_idx_rows, test_idx_rows, k=4):
    """Forecast T_{n+1} from past intervals only. Rows are drop indices n."""
    out = {}
    y_all = T[1:]
    tr, te = train_idx_rows, test_idx_rows
    out["clock"] = nmse(np.full(len(te), T[tr + 1].mean()), T[te + 1])
    out["last_interval"] = nmse(T[te], T[te + 1])
    for name, quad in [("history_linear", False), ("history_quadratic", True)]:
        X, y, idx = lagged(T, k, quad)
        row = {n: i for i, n in enumerate(idx)}
        a = [row[n] for n in tr if n in row]
        b = [row[n] for n in te if n in row]
        w = ridge_fit(X[a], y[a])
        out[name] = nmse(ridge_pred(w, X[b]), y[b])
    return out
