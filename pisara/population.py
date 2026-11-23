"""A population of tuft loops that compete for delays.

K loops hear the same drops. Each runs the coincidence ("myelin") rule from
loop.py on its own delay. Competition is by claiming, like lateral
inhibition at the moment of a drop: when loop k looks for the returning event
that coincides with the drop, any event at a delay already held by an earlier
loop (within +-excl ms) is not available to it. So each loop has to find a
coincidence nobody else owns.

    D_k <- D_k + eta * ((t - D_k) - t_rise_k)     t_rise_k: steepest rise in loop
                                                  k's return, unclaimed delays only

(An optional soft repulsion between delays, rho, is kept for comparison; it
is off by default because it only packs the delays into a tight cluster.)

The question for Gate 3: do the delays the loops settle on make a good
Takens embedding of the drip world, better than delays chosen by hand?
"""
import math
import numpy as np

from .loop import njit


@njit(cache=True)
def _run_pop(s, D0, beta, gamma, theta, eta, rho, sigma, Dmin, Dmax, slope_ms, window, learn_until, excl):
    # Dmin, Dmax: one range per loop (arrays)
    K = D0.shape[0]
    n = s.shape[0]
    x = np.zeros((K, n))
    D = D0.copy()
    a = 1.0 / theta
    for t in range(1, n):
        for k in range(K):
            pos = t - 1 - D[k]
            echo = 0.0
            if pos >= 1.0:
                i = int(pos)
                f = pos - i
                echo = (1 - f) * x[k, i] + f * x[k, i + 1]
            x[k, t] = x[k, t - 1] + a * (-x[k, t - 1] + math.tanh(beta * echo + gamma * s[t - 1]))
        onset = s[t - 1] > 0.5 and (t < 2 or s[t - 2] < 0.5)
        if onset and t < learn_until:
            newD = D.copy()
            for k in range(K):
                p = int(t - 1 - D[k])
                lo = p - window
                hi = p + window
                # a loop only hears returns from delays inside its own range
                if lo < t - 1 - Dmax[k]:
                    lo = int(t - 1 - Dmax[k])
                if hi > t - 1 - Dmin[k]:
                    hi = int(t - 1 - Dmin[k])
                if hi > t - 2:
                    hi = t - 2
                if lo > slope_ms + 1 and hi > lo:
                    best = -1e9
                    ib = p
                    for i in range(lo, hi + 1):
                        # claimed delays: a coincidence at a delay an earlier
                        # loop already holds does not count for this loop
                        taken = False
                        for j in range(k):
                            if abs((t - 1 - i) - D[j]) < excl:
                                taken = True
                                break
                        if taken:
                            continue
                        sl = x[k, i] - x[k, i - slope_ms]
                        if sl > best:
                            best = sl
                            ib = i
                    if best > 0.05:
                        newD[k] += eta * ((t - 1 - D[k]) - (ib - slope_ms + 1))
                for j in range(K):
                    if j != k:
                        d = D[k] - D[j]
                        sg = 1.0 if d > 0 else (-1.0 if d < 0 else (1.0 if k > j else -1.0))
                        newD[k] += rho * sg * math.exp(-d * d / (2 * sigma * sigma))
            for k in range(K):
                D[k] = min(max(newD[k], Dmin[k]), Dmax[k])
    return x, D


def run_population(s, D0, beta=0.9, gamma=3.0, theta=5.0, eta=0.3, rho=0.0, sigma=40.0,
                   Dmin=60.0, Dmax=1500.0, slope_ms=3, window=150, learn_until=None, excl=40.0):
    """Returns node traces (K x n) and the final delays (learning stops at learn_until)."""
    s = np.asarray(s, dtype=np.float64)
    lu = len(s) if learn_until is None else int(learn_until)
    K = len(D0)
    Dmin = np.broadcast_to(np.asarray(Dmin, dtype=np.float64), (K,)).copy()
    Dmax = np.broadcast_to(np.asarray(Dmax, dtype=np.float64), (K,)).copy()
    return _run_pop(s, np.asarray(D0, dtype=np.float64), float(beta), float(gamma), float(theta),
                    float(eta), float(rho), float(sigma), Dmin, Dmax,
                    int(slope_ms), int(window), lu, float(excl))


def leaky_trace(s, theta=100.0):
    """The observable that gets embedded: drops integrated by a slow leak,
    so its value says how long ago, and how densely, drops came."""
    y = np.zeros(len(s))
    a = 1.0 / theta
    for t in range(1, len(s)):
        y[t] = y[t - 1] + a * (-y[t - 1] + s[t - 1])
    return y


def mutual_information_curve(y, lags, bins=16, stride=5):
    """I(y(t); y(t - lag)) in bits for each lag (Fraser & Swinney 1986)."""
    out = []
    z = y[::stride]
    q = np.quantile(z, np.linspace(0, 1, bins + 1)[1:-1])
    b = np.searchsorted(q, z)
    for L in lags:
        m = int(round(L / stride))
        u, v = b[m:], b[:-m]
        P = np.zeros((bins, bins))
        np.add.at(P, (u, v), 1)
        P /= P.sum()
        pu, pv = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
        nz = P > 0
        out.append(float(np.sum(P[nz] * np.log2(P[nz] / (pu @ pv)[nz]))))
    return np.array(out)


def first_minimum(curve, lags):
    for i in range(1, len(curve) - 1):
        if curve[i] < curve[i - 1] and curve[i] <= curve[i + 1]:
            return lags[i]
    return lags[int(np.argmin(curve))]


def embed(y, times, delays):
    """Takens vector [y(t), y(t - d1), ...] at each time."""
    times = np.asarray(times)
    cols = [y[times]] + [y[np.clip(times - int(round(d)), 0, None)] for d in delays]
    return np.stack(cols, 1)


def knn_forecast(Xtr, ytr, Xte, k=10):
    """Analogue forecasting (Lorenz 1969; Farmer & Sidorowich 1987): average
    what followed the k most similar past states."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    d = (B ** 2).sum(1)[:, None] + (A ** 2).sum(1)[None, :] - 2 * B @ A.T
    idx = np.argpartition(d, k, axis=1)[:, :k]
    return ytr[idx].mean(1)


@njit(cache=True)
def _decorrelate(y, D0, eta, lam, learn_until, Dmin, Dmax):
    n = y.shape[0]
    D = D0
    m = 0.0
    v = 1e-6
    c = 0.0
    g = 0.0
    trace = np.zeros(n)
    for t in range(1, n):
        m += lam * (y[t] - m)
        v += lam * ((y[t] - m) ** 2 - v)
        pos = t - D
        if pos > 2 and t < learn_until:
            i = int(pos)
            warm = t > 3.0 / lam
            f = pos - i
            back = (1 - f) * y[i] + f * y[i + 1]
            slope = y[i + 1] - y[i]                       # d/dt of the returned signal
            yc = (y[t] - m) / math.sqrt(v)
            bc = (back - m) / math.sqrt(v)
            c += lam * (yc * bc - c)                      # running correlation now vs returned
            g += lam * (yc * slope / math.sqrt(v) - g)    # dc/dD = -<y_c(t) y'(t-D)>
            if warm:
                D += eta * c * g                          # gradient descent on c^2
            if D < Dmin:
                D = Dmin
            if D > Dmax:
                D = Dmax
        trace[t] = D
    return D, trace


def decorrelation_delay(y, D0=20.0, eta=0.2, lam=1e-4, learn_until=None, Dmin=5.0, Dmax=600.0):
    """The opposite of the coincidence rule: one loop moves its delay until
    what returns is uncorrelated with what is arriving now (c -> 0, by
    gradient descent on c^2 with running averages). A local, online version
    of the classic first-zero-of-autocorrelation delay choice."""
    lu = len(y) if learn_until is None else int(learn_until)
    return _decorrelate(np.asarray(y, dtype=np.float64), float(D0), float(eta), float(lam), lu,
                        float(Dmin), float(Dmax))
