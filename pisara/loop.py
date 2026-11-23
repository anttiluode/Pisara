"""The tuft loop: one nonlinear node whose own output returns after a delay D.

    dx/dt = ( -x + tanh( beta * x(t - D) + gamma * s(t) ) ) / theta

s(t) is the drip train, x(t - D) is the node's own past coming back (the
"tuft"), beta is how strongly the echo drives the node. With beta = 0 the
loop is cut and the node is a plain leaky unit (the no-loop control).

The node history over the last D ms is the delay line. Sampled at N points it
gives N "virtual nodes" (Appeltant et al., Nat. Commun. 2011).

Learning the delay (the "myelin" rule)
--------------------------------------
When a drop arrives, the loop looks at what it is returning around that
moment: the stretch x(t - D - W) ... x(t - D + W) of its own past (W = 150 ms).
It finds the steepest rise in that stretch, the onset of the strongest
returning event, and moves the delay part of the way toward making that onset
coincide with the drop:

    D <- D + eta * ( (t - D) - t_rise )

So the delay is shaped only by the coincidence of a drop with the loop's own
returning past. There is no error signal about intervals and no readout. The
window is like a plasticity window: events returning far from the drop do
not count.
"""
import math
import numpy as np

try:
    from numba import njit
except ImportError:                      # runs without numba, just slower
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f


@njit(cache=True)
def _run(s, D0, beta, gamma, theta, eta, learn, Dmin, Dmax, slope_ms, window):
    n = s.shape[0]
    x = np.zeros(n)
    Dtrace = np.zeros(n)
    D = D0
    a = 1.0 / theta
    for t in range(1, n):
        # echo by linear interpolation at fractional delay
        pos = t - 1 - D
        echo = 0.0
        if pos >= 1.0:
            i = int(pos)
            f = pos - i
            echo = (1 - f) * x[i] + f * x[i + 1]
        x[t] = x[t - 1] + a * (-x[t - 1] + math.tanh(beta * echo + gamma * s[t - 1]))
        # drop onset: rising edge of the drip pulse
        if learn and s[t - 1] > 0.5 and (t < 2 or s[t - 2] < 0.5):
            p = int(t - 1 - D)
            lo = p - window
            hi = p + window
            if hi > t - 2:
                hi = t - 2
            if lo > slope_ms + 1 and hi > lo:
                best = -1e9
                ib = p
                for i in range(lo, hi + 1):
                    sl = x[i] - x[i - slope_ms]
                    if sl > best:
                        best = sl
                        ib = i
                if best > 0.05:
                    D = D + eta * ((t - 1 - D) - (ib - slope_ms + 1))
                if D < Dmin:
                    D = Dmin
                if D > Dmax:
                    D = Dmax
        Dtrace[t] = D
    return x, Dtrace


def run(s, D, beta=0.9, gamma=3.0, theta=5.0, eta=0.0, Dmin=30.0, Dmax=1500.0,
        slope_ms=3, window=150):
    """Simulate the loop on drip train s. eta > 0 learns the delay."""
    return _run(np.asarray(s, dtype=np.float64), float(D), float(beta), float(gamma),
                float(theta), float(eta), eta > 0, float(Dmin), float(Dmax), int(slope_ms),
                int(window))


def virtual_nodes(x, times, D, N=50):
    """State at each time: the delay line x(t), x(t - D/N), ..., x(t - (N-1) D/N).
    D may be a scalar or one delay per time."""
    times = np.asarray(times)
    Ds = np.broadcast_to(np.asarray(D, dtype=float), times.shape)
    out = np.zeros((len(times), N))
    for j, (t, d) in enumerate(zip(times, Ds)):
        idx = np.round(t - np.arange(N) * d / N).astype(int)
        idx = np.clip(idx, 0, len(x) - 1)
        out[j] = x[idx]
    return out


def echo_forecast(x, drop_times, D, pulse_ms=10, refractory=30, thr=0.25):
    """Physical forecast with no readout: after drop n, the next drop is
    predicted for the first moment the RETURNING signal e(t) = x(t - D) rises
    above thr.

    No leak: e(t) only reads x up to t - D. The search stops at
    t_n + pulse_ms + D, so every value used was already in the loop when drop
    n's pulse ended. (Reading x itself would leak: the next drop's own pulse is
    a crossing.) Returns predicted next-drop times (nan if nothing rises).
    D may be a scalar or one delay per drop."""
    Ds = np.broadcast_to(np.asarray(D, dtype=float), np.shape(drop_times))
    pred = np.full(len(drop_times), np.nan)
    for k, (t, d) in enumerate(zip(drop_times, Ds)):
        d = int(round(d))
        lo, hi = t + refractory, min(t + pulse_ms + d, len(x) - 1 + d)
        if hi <= lo + 1 or lo - d < 0:
            continue
        e = x[lo - d:hi - d]
        above = e > thr
        # an echo that is still high when the search opens belongs to the
        # present drop; wait until it has passed, then take the next rise
        rise = np.nonzero(above[1:] & ~above[:-1])[0]
        if len(rise):
            pred[k] = lo + rise[0] + 1
    return pred
