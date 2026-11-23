"""Gate 4: one neuron, both rules.

A unit with two loops on the same drops:

  in-step loop     coincidence (Hebbian) delay, from Gate 2. Its echo
                   predicts the next drop with no readout.
  describing loop  decorrelating (anti-Hebbian) delay tau, from Gate 3. The
                   slow trace y(t) sampled at t, t - tau, ..., t - 6 tau is the
                   unit's description of the world's state; the next interval
                   is forecast by nearest analogues among states it has
                   already lived through (online: only past drops are used).

  both_feature     the describing loop's analogue search, with the in-step
                   loop's echo forecast added as one more coordinate of the
                   state (the description is told what the echo expects)
  both             a gate that trusts whichever loop has been right lately:
                   recent absolute error of each, as a running average
                   (alpha = 0.2), and the next forecast comes from the one
                   with the smaller recent error. "When in step, trust the
                   echo; when slipping, use the description."

Everything is online, one pass, no train/test split: every forecast uses only
what happened before it. The world switches:

  clock 300 -> clock 420 -> period-two 300/420 -> chaotic -> draining -> clock 300

Measured overall and in the first 15 drops after each switch (where a loop
that re-locks fast could matter most). Stated before the frozen run, after
one smoke run on seed 9 showed the describing loop dominating overall.

Kill condition: if neither pairing beats the better single loop, overall or
just after the switches, the pairing adds nothing.

Fresh seeds 41-43.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pisara import loop, population as P, world  # noqa: E402

SEGMENTS = [("periodic", 200, 300.0), ("periodic", 200, 420.0), ("period2", 200, 360.0),
            ("chaotic", 300, 300.0), ("draining", 200, 300.0), ("periodic", 200, 300.0)]
K_EMB = 6


def online_knn(X, T, k=10, min_rows=20):
    """Forecast T[n] from rows m < n only (T[m] is known once drop m+1 has come)."""
    pred = np.full(len(T), np.nan)
    for n in range(min_rows, len(T)):
        A = X[:n]
        mu, sd = A.mean(0), A.std(0) + 1e-9
        d = (((A - mu) / sd - (X[n] - mu) / sd) ** 2).sum(1)
        kk = min(k, n)
        idx = np.argpartition(d, kk - 1)[:kk]
        pred[n] = T[idx].mean()
    return pred


def run(seed):
    rng = np.random.default_rng(seed)
    T, starts = world.schedule(SEGMENTS, rng)
    s, tt = world.drip_train(T)
    onsets = tt[:-1]

    # running clock: mean of the last 20 intervals
    clock = np.array([T[max(0, n - 20):n].mean() if n else 300.0 for n in range(len(T))])

    # in-step loop
    x, Dt = loop.run(s, 300.0, eta=0.3)
    Dper = Dt[onsets + 1]
    echo = loop.echo_forecast(x, onsets, Dper) - onsets
    echo = np.where(np.isfinite(echo), echo, clock)

    # describing loop
    y = P.leaky_trace(s)
    _, tau_trace = P.decorrelation_delay(y)
    tau = tau_trace[onsets]
    tau = np.where(tau > 0, tau, 60.0)
    X = np.stack([y[np.clip(onsets - np.round(j * tau).astype(int), 0, None)] for j in range(K_EMB + 1)], 1)
    desc = online_knn(X, T)
    desc = np.where(np.isfinite(desc), desc, clock)

    # both_feature: the echo's expectation as one more coordinate
    Xf = np.c_[X, echo]
    bothf = online_knn(Xf, T)
    bothf = np.where(np.isfinite(bothf), bothf, clock)

    # both: trust whoever has been right lately
    both = np.empty(len(T))
    ec = ek = 50.0
    use_echo = np.zeros(len(T), bool)
    for n in range(len(T)):
        use_echo[n] = ec <= ek
        both[n] = echo[n] if use_echo[n] else desc[n]
        ec += 0.2 * (abs(echo[n] - T[n]) - ec)
        ek += 0.2 * (abs(desc[n] - T[n]) - ek)

    res = {"seed": seed, "segments": []}
    ends = starts[1:] + [len(T)]
    allrows = np.arange(30, len(T))
    for (regime, nseg, base), a, e in zip(SEGMENTS, starts, ends):
        r = np.arange(max(a, 30), e)
        seg = {"regime": regime, "base": base,
               "clock": float(np.mean(np.abs(clock[r] - T[r]))),
               "in_step": float(np.mean(np.abs(echo[r] - T[r]))),
               "describing": float(np.mean(np.abs(desc[r] - T[r]))),
               "both": float(np.mean(np.abs(both[r] - T[r]))),
               "both_feature": float(np.mean(np.abs(bothf[r] - T[r]))),
               "share_echo": float(use_echo[r].mean()),
               "delay_end": float(Dper[e - 1]), "tau_end": float(tau[e - 1])}
        res["segments"].append(seg)
    post = np.concatenate([np.arange(a, min(a + 15, len(T))) for a in starts[1:]])
    for name, p in [("clock", clock), ("in_step", echo), ("describing", desc), ("both", both),
                    ("both_feature", bothf)]:
        res[name + "_overall"] = float(np.mean(np.abs(p[allrows] - T[allrows])))
        res[name + "_post_switch"] = float(np.mean(np.abs(p[post] - T[post])))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    ap.add_argument("--out", default="results/gate4.json")
    args = ap.parse_args()
    out = []
    for sd in args.seeds:
        r = run(sd)
        out.append(r)
        for g in r["segments"]:
            print(f"seed {sd} {g['regime']:9s} {g['base']:.0f}: clock {g['clock']:6.1f} | in-step "
                  f"{g['in_step']:6.1f} | describing {g['describing']:6.1f} | both {g['both']:6.1f} "
                  f"| both-feature {g['both_feature']:6.1f} "
                  f"(echo used {100 * g['share_echo']:.0f}%)  delay {g['delay_end']:.0f} tau {g['tau_end']:.0f}")
        for tag in ["overall", "post_switch"]:
            print(f"seed {sd} {tag.upper()}: " + " | ".join(
                f"{nm} {r[nm + '_' + tag]:.1f}" for nm in
                ["clock", "in_step", "describing", "both", "both_feature"]) + " ms", flush=True)
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
