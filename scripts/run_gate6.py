"""Gate 6: Sol's two remaining questions for Pisara.

A. Can the coincidence loop beat an ADAPTIVE CLOCK at omission timing, not
   just a damped resonator? Period-two, 10% of drops removed. For each
   missing drop, when does each system expect it?
     loop             fixed D = 600 ms (the good basin), its own rise
     clock_mean       last seen drop + median of the last 20 seen intervals
     clock_lag2       last seen drop + the seen interval two back (an explicit
                      "period-two clock": the algorithm the loop embodies)
   Seen intervals are those between drops that actually arrived, so earlier
   omissions corrupt the clocks' history, as they would corrupt any listener.

B. Does the describing loop need its ARCHIVE? Gate 3-5 forecast with nearest
   neighbours: a store of past states, outside the loop. Replace the archive
   with fixed weights learned once (memory in synapses, not in a store):
     knn              nearest analogues (the archive)
     ridge_linear     one weight per embedding coordinate
     ridge_quadratic  plus all pairwise products of the coordinates (centred)
   on the Henon world, learned tau, tau..6 tau.

Fresh seeds 61-63.
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


def part_a(seed):
    rng = np.random.default_rng(seed)
    T = world.intervals("period2", 800, rng)
    s, tt = world.drip_train(T)
    keep = rng.random(len(tt)) > 0.1
    keep[:20] = True
    s2 = np.zeros_like(s)
    for t in tt[keep]:
        s2[t:t + 10] = 1.0
    x, _ = loop.run(s2, 600.0)
    miss_idx = np.nonzero(~keep)[0]
    miss_idx = miss_idx[tt[miss_idx] > tt[len(tt) // 3]]
    frac, loop_err = omission_score(x, tt[miss_idx])
    seen = tt[keep]
    errs = {"clock_mean": [], "clock_lag2": []}
    for m in miss_idx:
        prev = seen[seen < tt[m]]
        iv = np.diff(prev)
        last = prev[-1]
        errs["clock_mean"].append(abs(last + np.median(iv[-20:]) - tt[m]))
        errs["clock_lag2"].append(abs(last + iv[-2] - tt[m]))
    res = {"seed": seed, "loop_rise": frac, "loop_median_err": loop_err}
    for k, v in errs.items():
        res[k + "_median_err"] = float(np.median(v))
    print(f"A seed {seed}: loop rise {100 * frac:.0f}% median error {loop_err} ms | "
          f"clock_mean {res['clock_mean_median_err']:.1f} ms | clock_lag2 {res['clock_lag2_median_err']:.1f} ms",
          flush=True)
    return res


def part_b(seed):
    rng = np.random.default_rng(seed)
    T = world.intervals("henon", 2000, rng)
    s, tt = world.drip_train(T)
    y = P.leaky_trace(s)
    rows = np.arange(10, len(T))
    a = len(rows) // 2
    tr, te = rows[:a], rows[a:]
    tau, _ = P.decorrelation_delay(y, learn_until=int(tt[tr[-1]]))
    X = P.embed(y, tt[:-1], tau * np.arange(1, 7))
    res = {"seed": seed, "tau": float(tau)}
    res["knn"] = measures.nmse(P.knn_forecast(X[tr], T[tr], X[te]), T[te])
    m = measures.ridge_fit(X[tr], T[tr])
    res["ridge_linear"] = measures.nmse(measures.ridge_pred(m, X[te]), T[te])
    Xc = X - X[tr].mean(0)
    k = Xc.shape[1]
    Q = np.c_[Xc, np.stack([Xc[:, i] * Xc[:, j] for i in range(k) for j in range(i, k)], 1)]
    m = measures.ridge_fit(Q[tr], T[tr])
    res["ridge_quadratic"] = measures.nmse(measures.ridge_pred(m, Q[te]), T[te])
    print(f"B seed {seed}: knn (archive) {res['knn']:.3f} | ridge linear {res['ridge_linear']:.3f} | "
          f"ridge quadratic {res['ridge_quadratic']:.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[61, 62, 63])
    ap.add_argument("--out", default="results/gate6.json")
    args = ap.parse_args()
    res = {"A": [part_a(sd) for sd in args.seeds], "B": [part_b(sd) for sd in args.seeds]}
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
