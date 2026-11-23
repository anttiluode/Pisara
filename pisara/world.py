"""Drip worlds.

Every world is a sequence of drop intervals T_n (milliseconds). The drops are
then laid out as events on a 1 ms clock, because the machine lives in
continuous time, not in "drop number".

Regimes follow Robert Shaw, "The Dripping Faucet as a Model Chaotic System"
(1984): a water clock, a period-two faucet, a chaotic faucet modelled by his
noisy one-dimensional map, plus Antti's draining coffee filter. Two controls
have the same interval distribution as a structured world but no structure.
"""
import numpy as np

REGIMES = ["periodic", "period2", "chaotic", "henon", "draining", "shuffled", "iid"]


def intervals(regime, n, rng, base=300.0):
    """Return n drop intervals in ms for one regime."""
    if regime == "periodic":          # water clock, 1% jitter
        return base + rng.normal(0, 3.0, n)
    if regime == "period2":           # long-short pairs (Shaw Fig. 3, 28)
        T = np.where(np.arange(n) % 2 == 0, base - 60, base + 60).astype(float)
        return T + rng.normal(0, 3.0, n)
    if regime == "chaotic":           # Shaw Fig. 4: x' = 3.6 x (1 - x) + 0.02 noise
        x, out = 0.4, np.empty(n)
        for i in range(n):
            x = 3.6 * x * (1 - x) + 0.02 * rng.normal()
            x = min(max(x, 0.01), 0.99)
            out[i] = 200.0 + 200.0 * x
        return out
    if regime == "henon":             # two-dimensional chaos: the next interval needs the last TWO
        # Shaw's Fig. 5 uses a Henon map in his own scaling; the printed
        # constants diverge from a generic start, so the standard map is used
        x, y, out = 0.1, 0.0, np.empty(n)
        for i in range(n):
            x, y = y + 1 - 1.4 * x * x, 0.3 * x
            out[i] = base + 75.0 * x + rng.normal(0, 2.0)
        return out
    if regime == "draining":          # coffee filter: rate ~ sqrt(level), refilled when slow
        out, h = np.empty(n), 1.0
        for i in range(n):
            T = 0.8 * base / np.sqrt(h)
            out[i] = T * (1 + 0.02 * rng.normal())
            h -= 0.04
            if T > 2.0 * base or h <= 0.05:
                h = 1.0
        return out
    if regime == "shuffled":          # chaotic intervals, order destroyed
        return rng.permutation(intervals("chaotic", n, rng, base))
    if regime == "iid":               # same range as chaotic, independent draws
        return rng.uniform(200.0, 400.0, n)
    raise ValueError(regime)


def drip_train(T, pulse_ms=10, lead_ms=500):
    """Lay intervals out on a 1 ms clock. Returns (s, drop_times)."""
    t = lead_ms + np.concatenate([[0.0], np.cumsum(T)])
    t = np.round(t).astype(np.int64)
    s = np.zeros(int(t[-1]) + pulse_ms + 1000, dtype=np.float64)
    for ti in t:
        s[ti:ti + pulse_ms] = 1.0
    return s, t


def schedule(parts, rng, base=300.0):
    """Concatenate regimes: parts = [(regime, n_drops, base_ms), ...].
    Returns intervals and the drop index where each part starts."""
    T, starts = [], []
    for regime, n, b in parts:
        starts.append(sum(len(x) for x in T))
        T.append(intervals(regime, n, rng, b))
    return np.concatenate(T), starts
