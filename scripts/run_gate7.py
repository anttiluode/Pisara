"""Gate 7: Neon's second audit.

A. Gate 5B rematch with matched detectors. The echo forecast read the RAW
   jittered return with a first-crossing trigger and a MEAN error; omission
   read the soma-smoothed trace, nearest crossing, MEDIAN. Here the echo
   forecast is scored three ways at each jitter:
     raw_first    as in Gate 5B (raw return, first crossing), mean and median
     smooth_first return smoothed by the same 5 ms leak as the soma, first
                  crossing, mean and median
     early_rate   fraction of forecasts that come more than 100 ms early
   Same seeds as Gate 5 (51-53), so the numbers line up with Gate 5B.

B. Gate 6A with mean error as well as median (median hides how earlier
   omissions corrupt the clocks' history). Same seeds as Gate 6 (61-63).

C. A bank of leaks instead of a hand-matched one (Gate 5A showed a 100 ms
   leak starves the embedding when drops are 600 ms apart). The observable
   is three traces with leaks 50, 150 and 450 ms; each learns its own
   decorrelating tau and contributes its value at t, t - tau, t - 2 tau,
   t - 3 tau (12 coordinates). Compared with the single 100 ms leak, tau..6 tau
   (7 coordinates), on Henon at base 150, 300 and 600 ms, with nearest
   neighbours and with fixed quadratic weights. Fresh seeds 71-73.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pisara import loop, measures, population as P, world  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gate2 import omission_score  # noqa: E402

SIGMAS = [0, 2, 5, 10, 20]


def smooth(e, theta=5.0):
    out = np.zeros_like(e)
    a = 1.0 / theta
    for t in range(1, len(e)):
        out[t] = out[t - 1] + a * (e[t - 1] - out[t - 1])
    return out


def part_a(seed):
    rng = np.random.default_rng(seed)
    T = world.intervals("period2", 800, rng)
    s, tt = world.drip_train(T)
    keep = rng.random(len(tt)) > 0.1
    keep[:20] = True
    s2 = np.zeros_like(s)
    for t in tt[keep]:
        s2[t:t + 10] = 1.0
    rows = np.nonzero(keep[:-1] & keep[1:])[0]
    rows = rows[rows > len(T) // 3]
    out = []
    for sg in SIGMAS:
        x, _, e = loop.run(s2, 600.0, jitter=float(sg), seed=seed, return_echo=True)
        r = {"sigma": sg}
        for name, sig in [("raw_first", e), ("smooth_first", smooth(e))]:
            p = loop.echo_forecast(x, tt[:-1], 600.0, received=sig)[rows] - tt[:-1][rows]
            p = np.where(np.isfinite(p), p, 300.0)
            err = p - T[rows]
            r[name + "_mean"] = float(np.mean(np.abs(err)))
            r[name + "_median"] = float(np.median(np.abs(err)))
            r[name + "_early_rate"] = float(np.mean(err < -100))
        out.append(r)
        print(f"A seed {seed} jitter {sg:2d}: raw mean/median {r['raw_first_mean']:6.1f}/"
              f"{r['raw_first_median']:4.1f} early {100 * r['raw_first_early_rate']:3.0f}% | smoothed "
              f"mean/median {r['smooth_first_mean']:6.1f}/{r['smooth_first_median']:4.1f} early "
              f"{100 * r['smooth_first_early_rate']:3.0f}%", flush=True)
    return out


def part_b(seed):
    rng = np.random.default_rng(seed)
    T = world.intervals("period2", 800, rng)
    s, tt = world.drip_train(T)
    keep = rng.random(len(tt)) > 0.1
    keep[:20] = True
    s2 = np.zeros_like(s)
    for t in tt[keep]:
        s2[t:t + 10] = 1.0
    x, _ = loop.run(s2, 600.0)
    miss = np.nonzero(~keep)[0]
    miss = miss[tt[miss] > tt[len(tt) // 3]]
    a = x / (np.percentile(x, 95) + 1e-12)
    loop_err = []
    for m in miss:
        seg = a[tt[m] - 150:tt[m] + 150]
        up = np.nonzero((seg[1:] > 0.25) & (seg[:-1] <= 0.25))[0]
        loop_err.append(abs(up[np.argmin(np.abs(up + 1 - 150))] + 1 - 150) if len(up) else 150.0)
    seen = tt[keep]
    cm, c2, consec = [], [], []
    for m in miss:
        prev = seen[seen < tt[m]]
        iv = np.diff(prev)
        cm.append(abs(prev[-1] + np.median(iv[-20:]) - tt[m]))
        c2.append(abs(prev[-1] + iv[-2] - tt[m]))
        consec.append(bool(not keep[m - 1] or not keep[m - 2] or not keep[m - 3]))
    consec = np.array(consec)
    res = {"seed": seed, "n": int(len(miss)), "n_after_recent_omission": int(consec.sum())}
    for k, v in [("loop", loop_err), ("clock_mean", cm), ("clock_lag2", c2)]:
        v = np.array(v, float)
        res[k + "_mean"], res[k + "_median"] = float(v.mean()), float(np.median(v))
        res[k + "_mean_after_recent_omission"] = float(v[consec].mean()) if consec.any() else None
    print(f"B seed {seed} ({res['n']} omissions, {res['n_after_recent_omission']} soon after another): "
          + " | ".join(f"{k} mean {res[k + '_mean']:.1f} median {res[k + '_median']:.1f} "
                       f"after-recent {res[k + '_mean_after_recent_omission']}" for k in
                       ["loop", "clock_mean", "clock_lag2"]), flush=True)
    return res


def part_c(seed):
    out = []
    for base in [150.0, 300.0, 600.0]:
        rng = np.random.default_rng(seed)
        T = world.intervals("henon", 2000, rng, base)
        s, tt = world.drip_train(T)
        rows = np.arange(10, len(T))
        a = len(rows) // 2
        tr, te = rows[:a], rows[a:]
        lu = int(tt[tr[-1]])
        on = tt[:-1]

        def feats(thetas, mult):
            cols = []
            for th in thetas:
                y = P.leaky_trace(s, th)
                tau, _ = P.decorrelation_delay(y, learn_until=lu, Dmax=1500.0)
                cols += [y[np.clip(on - int(round(k * tau)), 0, None)] for k in range(mult + 1)]
            return np.stack(cols, 1)

        r = {"seed": seed, "base": base}
        for name, X in [("single_leak_100", feats([100.0], 6)), ("leak_bank", feats([50.0, 150.0, 450.0], 3))]:
            r[name + "_knn"] = measures.nmse(P.knn_forecast(X[tr], T[tr], X[te]), T[te])
            Xc = X - X[tr].mean(0)
            k = Xc.shape[1]
            Q = np.c_[Xc, np.stack([Xc[:, i] * Xc[:, j] for i in range(k) for j in range(i, k)], 1)]
            m = measures.ridge_fit(Q[tr], T[tr])
            r[name + "_quadratic"] = measures.nmse(measures.ridge_pred(m, Q[te]), T[te])
        out.append(r)
        print(f"C seed {seed} base {base:.0f}: single leak knn {r['single_leak_100_knn']:.3f} quad "
              f"{r['single_leak_100_quadratic']:.3f} | leak bank knn {r['leak_bank_knn']:.3f} quad "
              f"{r['leak_bank_quadratic']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gate7.json")
    args = ap.parse_args()
    res = {"A": {sd: part_a(sd) for sd in [51, 52, 53]},
           "B": [part_b(sd) for sd in [61, 62, 63]],
           "C": sum([part_c(sd) for sd in [71, 72, 73]], [])}
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
