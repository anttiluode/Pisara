"""Gate 3: do competing tuft loops grow their own Takens embedding?

Six loops, each with its own delay range (150-350, 350-550, ..., 1150-1350
ms: axons of different length classes, myelin fine-tunes within the class),
learn their delays on the first half of the drops with the coincidence rule.
Their delays are then used as Takens delays for one slow observable y(t)
(the drops through a 100 ms leak), and the next interval is forecast by
nearest-neighbour analogues (k = 10) in that embedding.

Same embedding, same forecaster, these delay sets:
  learned_coincidence  the competing loops' delays (coincidence rule)
  learned_decorrelation  tau, 2 tau, ..., 6 tau, where tau is learned online by
                   ONE loop with the opposite rule: move the delay until what
                   returns is uncorrelated with what arrives now
  acf_zero         the same, with tau = first zero of the autocorrelation
                   computed offline (the hand-made answer the rule should find)
  range_centres    250, 450, ..., 1250: same ranges, no learning
  fraser_swinney   multiples of the first minimum of I(y(t); y(t - tau))
  random           uniform in 150-1350 ms (mean of 5 draws)
  *_on_shuffle     delays learned by each rule on the SAME world with its
                   intervals shuffled: if these do as well, the rule found
                   nothing specific to the world's order

Fresh seeds 31-33, not looked at before this run.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pisara import measures, population as P, world  # noqa: E402

EDGES = np.arange(150, 1351, 200)
LO, HI = EDGES[:-1].astype(float), EDGES[1:].astype(float)
CENTRES = (LO + HI) / 2
LAGS = np.arange(20, 801, 10)


def learn_delays(s, learn_until):
    _, D = P.run_population(s, CENTRES, Dmin=LO, Dmax=HI, excl=0.0, learn_until=learn_until)
    return np.sort(D)


def score(y, tt, T, delays, tr, te, k=10):
    X = P.embed(y, tt[:-1], delays)
    pred = P.knn_forecast(X[tr], T[tr], X[te], k)
    return measures.nmse(pred, T[te])


def run(regime, seed, n):
    rng = np.random.default_rng(seed)
    T = world.intervals(regime, n, rng)
    s, tt = world.drip_train(T)
    rows = np.arange(10, n)
    a = int(0.5 * len(rows))
    tr, te = rows[:a], rows[a:]
    learn_until = int(tt[tr[-1]])
    y = P.leaky_trace(s)

    res = {"regime": regime, "seed": seed}
    sets = {}
    sets["learned_coincidence"] = learn_delays(s, learn_until)
    tau_l, _ = P.decorrelation_delay(y, learn_until=learn_until)
    sets["learned_decorrelation"] = tau_l * np.arange(1, 7)
    yc = y[:learn_until] - y[:learn_until].mean()
    ac = np.array([np.mean(yc[L:] * yc[:-L]) for L in range(1, 400)]) / yc.var()
    tau_a = int(np.argmax(ac < 0)) + 1
    sets["acf_zero"] = tau_a * np.arange(1, 7)
    sets["range_centres"] = CENTRES
    mi = P.mutual_information_curve(y[:learn_until], LAGS)
    tau = P.first_minimum(mi, LAGS)
    sets["fraser_swinney"] = tau * np.arange(1, 7)
    Ts = rng.permutation(T)
    ss, _ = world.drip_train(Ts)
    sets["coincidence_on_shuffle"] = learn_delays(ss, learn_until)
    tau_s, _ = P.decorrelation_delay(P.leaky_trace(ss), learn_until=learn_until)
    sets["decorrelation_on_shuffle"] = tau_s * np.arange(1, 7)
    for name, d in sets.items():
        res[name] = score(y, tt, T, d, tr, te)
        res[name + "_delays"] = [round(float(v)) for v in d]
    res["random"] = float(np.mean([score(y, tt, T, np.sort(rng.uniform(150, 1350, 6)), tr, te)
                                   for _ in range(5)]))
    res["fraser_swinney_tau"] = int(tau)
    res["tau_learned"], res["tau_acf"] = float(tau_l), tau_a
    res["clock"] = 1.0
    base = measures.interval_baselines(T, tr - 1, te - 1)
    res["history_quadratic"] = base["history_quadratic"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[31, 32, 33])
    ap.add_argument("--drops", type=int, default=2000)
    ap.add_argument("--regimes", nargs="+",
                    default=["periodic", "period2", "chaotic", "draining", "shuffled"])
    ap.add_argument("--out", default="results/gate3.json")
    args = ap.parse_args()
    out = []
    for regime in args.regimes:
        for seed in args.seeds:
            r = run(regime, seed, args.drops)
            out.append(r)
            print(f"{regime:9s} seed {seed}: coincidence {r['learned_coincidence']:.3f} "
                  f"{r['learned_coincidence_delays']} | decorrelation {r['learned_decorrelation']:.3f} "
                  f"(tau {r['tau_learned']:.0f}) | acf-zero {r['acf_zero']:.3f} (tau {r['tau_acf']}) | "
                  f"FS {r['fraser_swinney']:.3f} (tau {r['fraser_swinney_tau']}) | "
                  f"centres {r['range_centres']:.3f} | random {r['random']:.3f} | "
                  f"coinc-on-shuffle {r['coincidence_on_shuffle']:.3f} | "
                  f"decor-on-shuffle {r['decorrelation_on_shuffle']:.3f} | "
                  f"quad history {r['history_quadratic']:.3f}", flush=True)
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
