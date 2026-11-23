"""Gate 2: the myelin arm. Can the loop learn its own delay?

Part A (basins): for each stationary regime, start the delay at 150..900 ms
and let the coincidence rule run for 600 drops. Where does it settle, and how
well does the echo then forecast the next drop (no readout)?

Part B (switching world): water clock at 300 ms -> water clock at 420 ms ->
period-two (300/420) -> draining filter -> water clock at 300 ms. One run,
delay starting at 300 ms. Compare, per segment:
  learned delay echo forecast
  best single fixed delay for the whole run (chosen with hindsight: an
      upper bound on what any fixed loop can do)
  clock that knows the running mean of the last 20 intervals

Part C (omission): in period-two, drop 10% of drops at random. Does the
loop produce a rise at the time the missing drop should have come?
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pisara import loop, world  # noqa: E402

INITS = list(range(150, 901, 50))
ETA = 0.3


def echo_mae(x, tt, T, Dper, rows):
    pred = loop.echo_forecast(x, tt[:-1], Dper) - tt[:-1]
    p = pred[rows]
    hit = np.isfinite(p)
    err = np.abs(np.where(hit, p, np.nanmean(T)) - T[rows])
    return float(err.mean()), float(hit.mean())


def running_clock_mae(T, rows, k=20):
    pred = np.array([T[max(0, n - k):n].mean() for n in rows])
    return float(np.mean(np.abs(pred - T[rows])))


def part_a(seed, n=600):
    out = []
    for regime in ["periodic", "period2", "chaotic", "draining", "shuffled"]:
        rng = np.random.default_rng(seed)
        T = world.intervals(regime, n, rng)
        s, tt = world.drip_train(T)
        rows = np.arange(int(0.7 * n), n)
        res = {"regime": regime, "seed": seed, "settle": [],
               "clock_mae": running_clock_mae(T, rows)}
        for D0 in INITS:
            x, Dt = loop.run(s, D0, eta=ETA)
            Dper = Dt[tt[:-1] + 1]
            mae, hit = echo_mae(x, tt, T, Dper, rows)
            res["settle"].append({"D0": D0, "D_final": float(Dt[tt[-1]]), "echo_mae": mae, "hit": hit})
        fixed = []
        for D in range(150, 961, 10):
            x, _ = loop.run(s, D)
            fixed.append((echo_mae(x, tt, T, D, rows)[0], D))
        res["best_fixed_mae"], res["best_fixed_D"] = min(fixed)
        out.append(res)
        finals = sorted({round(r["D_final"]) for r in res["settle"]})
        maes = [r["echo_mae"] for r in res["settle"]]
        print(f"A {regime:9s} seed {seed}: settles at {finals}  learned echo MAE "
              f"{min(maes):.1f}-{max(maes):.1f} ms | best fixed {res['best_fixed_mae']:.1f} ms "
              f"(D={res['best_fixed_D']}) | running clock {res['clock_mae']:.1f} ms", flush=True)
    return out


SEGMENTS = [("periodic", 200, 300.0), ("periodic", 200, 420.0), ("period2", 200, 360.0),
            ("draining", 200, 300.0), ("periodic", 200, 300.0)]


def part_b(seed):
    rng = np.random.default_rng(seed)
    T, starts = world.schedule(SEGMENTS, rng)
    s, tt = world.drip_train(T)
    x, Dt = loop.run(s, 300.0, eta=ETA)
    Dper = Dt[tt[:-1] + 1]
    ends = starts[1:] + [len(T)]
    fixed = {}
    for D in range(150, 1201, 10):
        xf, _ = loop.run(s, D)
        fixed[D] = xf
    res = {"seed": seed, "segments": [], "D_trace_at_drops": Dper.tolist()}
    tot_l, tot_f = [], []
    all_rows = np.arange(5, len(T))
    best_D = min(fixed, key=lambda D: echo_mae(fixed[D], tt, T, D, all_rows)[0])
    for (regime, n, b), a, e in zip(SEGMENTS, starts, ends):
        rows = np.arange(a + 20, e)          # skip the first 20 drops: time to adapt
        l, _ = echo_mae(x, tt, T, Dper, rows)
        f, _ = echo_mae(fixed[best_D], tt, T, best_D, rows)
        c = running_clock_mae(T, rows)
        seg = {"regime": regime, "base": b, "learned_mae": l, "fixed_mae": f, "clock_mae": c,
               "D_end": float(Dper[e - 1])}
        res["segments"].append(seg)
        tot_l.append(l); tot_f.append(f)
        print(f"B seed {seed} {regime:9s} {b:.0f} ms: learned {l:6.1f} | fixed D={best_D} {f:6.1f} "
              f"| running clock {c:6.1f} ms   delay at end {seg['D_end']:.0f}", flush=True)
    res["best_fixed_D"] = best_D
    res["learned_mean_mae"], res["fixed_mean_mae"] = float(np.mean(tot_l)), float(np.mean(tot_f))
    return res


def resonator(s, period, zeta=0.08, gain=0.5):
    """Control for the omission test (Neon): a damped oscillator kicked by
    the drops. It also keeps ringing when a drop is missing."""
    w = 2 * np.pi / period
    v = np.zeros(len(s)); u = 0.0
    for t in range(1, len(s)):
        u += -w * w * v[t - 1] - 2 * zeta * w * u + gain * s[t - 1]
        v[t] = v[t - 1] + u
    return v


def omission_score(act, omitted, thr=0.25):
    """Normalise by the unit's typical drop response (95th percentile); a
    'rise' is a crossing of thr within +-100 ms of the missing drop."""
    a = act / (np.percentile(act, 95) + 1e-12)
    errs, rises = [], 0
    for t in omitted:
        seg = a[t - 100:t + 100]
        up = np.nonzero((seg[1:] > thr) & (seg[:-1] <= thr))[0]
        if len(up):
            rises += 1
            errs.append(abs(up[np.argmin(np.abs(up + 1 - 100))] + 1 - 100))
    return rises / max(len(omitted), 1), (float(np.median(errs)) if errs else None)


def part_c(seed, n=600, p_omit=0.1):
    rng = np.random.default_rng(seed)
    T = world.intervals("period2", n, rng)
    s, tt = world.drip_train(T)
    keep = rng.random(len(tt)) > p_omit
    keep[:20] = True
    omitted = tt[~keep]
    s2 = np.zeros_like(s)
    for t in tt[keep]:
        s2[t:t + 10] = 1.0
    out = {"seed": seed, "n_omitted": int(len(omitted))}
    late = omitted[omitted > tt[len(tt) // 3]]
    half = len(late) // 2
    # Neon's basin point: start the learned delay everywhere, not only at 520
    sweep = []
    for D0 in INITS:
        x, Dt = loop.run(s2, float(D0), eta=ETA)
        frac, err = omission_score(x, late)
        sweep.append({"D0": D0, "D_final": float(Dt[-1]), "rise_fraction": frac, "median_onset_error": err})
    out["learned_from_every_start"] = sweep
    good = [r for r in sweep if r["rise_fraction"] > 0.9 and (r["median_onset_error"] or 99) < 15]
    out["fraction_of_starts_that_work"] = len(good) / len(sweep)
    # resonator control: pick the period on the first half of omissions, score the second
    best = None
    for P in range(200, 801, 10):
        v = resonator(s2, P)
        f, e = omission_score(v, late[:half])
        sc = (-f, e if e is not None else 99)
        if best is None or sc < best[0]:
            best = (sc, P)
    v = resonator(s2, best[1])
    f, e = omission_score(v, late[half:])
    out["resonator"] = {"period": best[1], "rise_fraction": f, "median_onset_error": e}
    print(f"C seed {seed}: learned loop works from {100 * out['fraction_of_starts_that_work']:.0f}% of starts "
          f"(finals {sorted({round(r['D_final']) for r in sweep})}); resonator P={best[1]}: rise "
          f"{100 * f:.0f}%, median onset error {e} ms", flush=True)
    for name, beta, eta, D0 in [("loop_learned", 0.9, ETA, 520.0), ("no_loop", 0.0, 0.0, 600.0)]:
        x, Dt = loop.run(s2, D0, beta=beta, eta=eta)
        late = omitted[omitted > tt[len(tt) // 3]]
        errs, peaks = [], []
        for t in late:
            seg = x[t - 100:t + 100]
            peaks.append(float(seg.max()))
            if seg.max() > 0.25:
                errs.append(float(np.argmax(np.diff((seg > 0.25).astype(int)) > 0) + 1 - 100))
        out[name] = {"mean_peak_at_omission": float(np.mean(peaks)),
                     "fraction_with_rise": float(np.mean(np.array(peaks) > 0.25)),
                     "rise_time_error_ms": (float(np.mean(np.abs(errs))) if errs else None)}
        print(f"C seed {seed} {name:12s}: rise at {100 * out[name]['fraction_with_rise']:.0f}% of "
              f"omitted drops, mean |onset error| {out[name]['rise_time_error_ms']} ms", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[21, 22, 23])
    ap.add_argument("--out", default="results/gate2.json")
    args = ap.parse_args()
    res = {"A": [], "B": [], "C": []}
    for sd in args.seeds:
        res["A"] += part_a(sd)
    for sd in args.seeds:
        res["B"].append(part_b(sd))
    for sd in args.seeds:
        res["C"].append(part_c(sd))
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
