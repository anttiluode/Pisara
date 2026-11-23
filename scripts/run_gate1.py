"""Gate 1: does the tuft loop forecast the next drop, and does the loop matter?

For each Shaw regime and frozen seed, forecast the next interval T_{n+1} at
the moment drop n begins, using only what has happened so far.

  explicit (written-down) forecasters: clock, last interval, linear and
      quadratic regression on the last 4 intervals
  no-loop node (beta = 0): the same leaky node, readout from its last W ms
  tuft loop (beta = 0.9): readout from its delay line (50 virtual nodes)
  echo forecast: no readout at all, the next drop is predicted to come when
      the loop's own echo rises

W and D are chosen on a validation block and scored once on a test block.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pisara import loop, measures, world  # noqa: E402

DELAYS = list(range(150, 961, 30))


def readout_scores(x, drop_times, T, D, rows_tr, rows_va, rows_te, N=50):
    V = loop.virtual_nodes(x, drop_times[:-1], D, N)      # state at onset of drop n
    y = T                                                  # T[n] = t_{n+1} - t_n
    m = measures.ridge_fit(V[rows_tr], y[rows_tr])
    return (measures.nmse(measures.ridge_pred(m, V[rows_va]), y[rows_va]),
            measures.nmse(measures.ridge_pred(m, V[rows_te]), y[rows_te]))


def echo_score(pred_all, drop_times, T, rows, fallback):
    """Mean absolute timing error (ms) of the echo forecast. Where no echo
    rises, the forecast falls back to the training-mean interval."""
    pred = pred_all[rows] - drop_times[:-1][rows]
    hit = np.isfinite(pred)
    pred = np.where(hit, pred, fallback)
    return float(np.mean(np.abs(pred - T[rows]))), float(hit.mean())


def run(regime, seed, n_drops):
    rng = np.random.default_rng(seed)
    T = world.intervals(regime, n_drops, rng)
    s, tt = world.drip_train(T)
    # drop n begins at tt[n]; the interval to forecast is T[n] = tt[n+1] - tt[n]
    # rows start at 8 so every forecaster has history
    rows = np.arange(8, n_drops)
    a, b = int(0.5 * len(rows)), int(0.7 * len(rows))
    tr, va, te = rows[:a], rows[a:b], rows[b:]

    res = {"regime": regime, "seed": seed,
           "stored_information_bits": measures.stored_information(T)}
    # explicit forecasters: row n uses T[n-1], T[n-2], ... to predict T[n]
    Tm = np.r_[np.nan, T]                       # Tm[n+1] = T[n]; Tm[n] = last interval at drop n
    base = measures.interval_baselines(Tm[1:], tr - 1, te - 1)
    res.update({k: v for k, v in base.items()})

    for name, beta in [("no_loop", 0.0), ("loop", 0.9)]:
        curve = []
        for D in DELAYS:
            x, _ = loop.run(s, D, beta=beta, theta=5.0)
            v, t_ = readout_scores(x, tt, T, D, tr, va, te)
            row = {"D": D, "val": v, "test": t_}
            if beta > 0:
                pa = loop.echo_forecast(x, tt[:-1], D)
                row["echo_val_mae"], _ = echo_score(pa, tt, T, va, T[tr].mean())
                row["echo_test_mae"], row["echo_hit"] = echo_score(pa, tt, T, te, T[tr].mean())
            curve.append(row)
        best = min(curve, key=lambda r: r["val"])
        res[name] = best["test"]
        res[name + "_D"] = best["D"]
        res[name + "_curve"] = curve
    be = min(res["loop_curve"], key=lambda r: r["echo_val_mae"])
    res["echo_mae"], res["echo_D"], res["echo_hit"] = be["echo_test_mae"], be["D"], be["echo_hit"]
    res["clock_mae"] = float(np.mean(np.abs(T[tr].mean() - T[te])))
    res["quad_mae"] = None  # filled below
    X, y, idx = measures.lagged(T, 4, True)        # row j predicts T[idx[j]+1] from T[idx[j]], ...
    row = {n + 1: j for j, n in enumerate(idx)}     # keyed by the drop whose interval is forecast
    a_ = [row[n] for n in tr if n in row]; b_ = [row[n] for n in te if n in row]
    m = measures.ridge_fit(X[a_], y[a_])
    res["quad_mae"] = float(np.mean(np.abs(measures.ridge_pred(m, X[b_]) - y[b_])))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    ap.add_argument("--drops", type=int, default=1500)
    ap.add_argument("--regimes", nargs="+", default=world.REGIMES)
    ap.add_argument("--out", default="results/gate1.json")
    args = ap.parse_args()
    out, t0 = [], time.time()
    for regime in args.regimes:
        for seed in args.seeds:
            r = run(regime, seed, args.drops)
            out.append(r)
            print(f"{regime:9s} seed {seed}  I={r['stored_information_bits']:.2f} bits  "
                  f"clock {r['clock']:.3f}  last {r['last_interval']:.3f}  "
                  f"lin {r['history_linear']:.3f}  quad {r['history_quadratic']:.3f}  "
                  f"no-loop {r['no_loop']:.3f} (W={r['no_loop_D']})  "
                  f"loop {r['loop']:.3f} (D={r['loop_D']})  "
                  f"| MAE ms: clock {r['clock_mae']:.1f} quad {r['quad_mae']:.1f} echo {r['echo_mae']:.1f} (D={r['echo_D']}, hit {r['echo_hit']:.2f})  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
