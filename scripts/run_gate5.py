"""Gate 5: Neon's two checks.

A. Timescale. Does the decorrelating loop's delay follow the WORLD or the
   LEAK of the observable? Henon intervals at base 300 and 600 ms, leak
   theta = 50, 100, 200 ms. Report learned tau, the offline autocorrelation
   zero, and the embedding forecast (tau..6 tau, nearest analogues).

B. Myelin jitter lesion. Make conduction timing unreliable (Gaussian jitter,
   SD sigma ms) and ask which breaks first:
     being in step   period-two, loop at D = 600 ms (fixed, so the basin
                     question is out of the way): echo forecast error and
                     omission timing
     describing      Henon, decorrelating tau, each embedding sample read
                     sigma ms early or late at random
   Prediction (Neon, stated before the run): the echo and omission collapse
   within a few ms of jitter; the embedding degrades gracefully.
   Loop jitter is an offset wandering with ~20 ms memory (a returning pulse
   is shifted as a whole). A smoke run on seed 9 with independent jitter every
   ms shredded the echo into spurious threshold crossings at 1 ms, a detector
   artifact, which is why the jitter was made coherent before the frozen run.

Fresh seeds 51-53.
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

SIGMAS = [0, 1, 2, 5, 10, 20, 40]


def acf_zero(y, lu, maxlag=900):
    yc = y[:lu] - y[:lu].mean()
    ac = np.array([np.mean(yc[L:] * yc[:-L]) for L in range(1, maxlag)]) / yc.var()
    return int(np.argmax(ac < 0)) + 1


def part_a(seed):
    out = []
    for base, theta in [(300, 50), (300, 100), (300, 200), (600, 100), (600, 200)]:
        rng = np.random.default_rng(seed)
        T = world.intervals("henon", 1500, rng, float(base))
        s, tt = world.drip_train(T)
        y = P.leaky_trace(s, theta)
        rows = np.arange(10, len(T))
        a = len(rows) // 2
        tr, te = rows[:a], rows[a:]
        lu = int(tt[tr[-1]])
        tau, _ = P.decorrelation_delay(y, learn_until=lu, Dmax=900.0)

        def sc(d):
            X = P.embed(y, tt[:-1], d)
            return measures.nmse(P.knn_forecast(X[tr], T[tr], X[te]), T[te])
        r = {"seed": seed, "base": base, "theta": theta, "tau_learned": float(tau),
             "acf_zero": acf_zero(y, lu), "embed_learned": sc(tau * np.arange(1, 7)),
             "embed_short_ladder": sc(50.0 * np.arange(1, 7))}
        out.append(r)
        print(f"A seed {seed} base {base} theta {theta}: tau {tau:.0f} (acf zero {r['acf_zero']}) | "
              f"embedding learned {r['embed_learned']:.3f} | ladder 50..300 {r['embed_short_ladder']:.3f}",
              flush=True)
    return out


def part_b(seed):
    out = {"seed": seed, "in_step": [], "describing": []}
    # being in step: period-two, fixed D = 600, 10% omitted
    rng = np.random.default_rng(seed)
    T = world.intervals("period2", 800, rng)
    s, tt = world.drip_train(T)
    keep = rng.random(len(tt)) > 0.1
    keep[:20] = True
    s2 = np.zeros_like(s)
    for t in tt[keep]:
        s2[t:t + 10] = 1.0
    omitted = tt[~keep]
    omitted = omitted[omitted > tt[len(tt) // 3]]
    kept_rows = np.nonzero(keep[:-1] & keep[1:])[0]
    kept_rows = kept_rows[kept_rows > len(T) // 3]
    for sg in SIGMAS:
        x, _, e = loop.run(s2, 600.0, jitter=float(sg), seed=seed, return_echo=True)
        pred = loop.echo_forecast(x, tt[:-1], 600.0, received=e) - tt[:-1]
        p = pred[kept_rows]
        hit = np.isfinite(p)
        mae = float(np.mean(np.abs(np.where(hit, p, 300.0) - T[kept_rows])))
        frac, err = omission_score(x, omitted)
        out["in_step"].append({"sigma": sg, "echo_mae": mae, "echo_hit": float(hit.mean()),
                               "omission_rise": frac, "omission_onset_err": err})
    # describing: Henon, decorrelating tau, jittered reads
    rng = np.random.default_rng(seed + 1000)
    T = world.intervals("henon", 2000, rng)
    s, tt = world.drip_train(T)
    y = P.leaky_trace(s)
    rows = np.arange(10, len(T))
    a = len(rows) // 2
    tr, te = rows[:a], rows[a:]
    tau, _ = P.decorrelation_delay(y, learn_until=int(tt[tr[-1]]))
    jr = np.random.default_rng(seed + 7)
    for sg in SIGMAS:
        d = tau * np.arange(1, 7)
        on = tt[:-1]
        cols = [y[on]] + [y[np.clip(np.round(on - dk + sg * jr.normal(size=len(on))).astype(int), 0, None)]
                          for dk in d]
        X = np.stack(cols, 1)
        out["describing"].append({"sigma": sg, "embed_nmse": measures.nmse(
            P.knn_forecast(X[tr], T[tr], X[te]), T[te])})
    for a_, b_ in zip(out["in_step"], out["describing"]):
        print(f"B seed {seed} jitter {a_['sigma']:2d} ms: echo MAE {a_['echo_mae']:6.1f} ms "
              f"(hit {a_['echo_hit']:.2f}) | omission rise {100 * a_['omission_rise']:.0f}% err "
              f"{a_['omission_onset_err']} ms | Henon embedding NMSE {b_['embed_nmse']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[51, 52, 53])
    ap.add_argument("--out", default="results/gate5.json")
    args = ap.parse_args()
    res = {"A": [], "B": []}
    for sd in args.seeds:
        res["A"] += part_a(sd)
    for sd in args.seeds:
        res["B"].append(part_b(sd))
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
