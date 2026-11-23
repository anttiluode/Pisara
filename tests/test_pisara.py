import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pisara import loop, measures, world  # noqa: E402


def test_echo_forecast_does_not_read_the_future():
    rng = np.random.default_rng(0)
    T = world.intervals("period2", 80, rng)
    s, tt = world.drip_train(T)
    x, _ = loop.run(s, 600.0)
    n = 50
    p = loop.echo_forecast(x, tt[:-1], 600.0)[n]
    # wipe everything after drop n's pulse and rerun: the forecast for n must not move
    s2 = s.copy()
    s2[tt[n] + 10:] = 0.0
    x2, _ = loop.run(s2, 600.0)
    assert loop.echo_forecast(x2, tt[:-1], 600.0)[n] == p


def test_period_two_echo_forecast():
    rng = np.random.default_rng(1)
    T = world.intervals("period2", 120, rng)
    s, tt = world.drip_train(T)
    x, _ = loop.run(s, 600.0)
    pred = loop.echo_forecast(x, tt[:-1], 600.0) - tt[:-1]
    assert np.nanmean(np.abs(pred[40:] - T[40:])) < 8.0


def test_stored_information_orders_regimes():
    rng = np.random.default_rng(2)
    assert measures.stored_information(world.intervals("period2", 2000, rng)) > 0.5
    assert measures.stored_information(world.intervals("chaotic", 2000, rng)) > 1.5
    assert measures.stored_information(world.intervals("shuffled", 2000, rng)) < 0.1


def test_learned_delay_finds_the_clock():
    rng = np.random.default_rng(3)
    T = world.intervals("periodic", 300, rng)
    s, tt = world.drip_train(T)
    _, Dt = loop.run(s, 250.0, eta=0.3)
    assert abs(Dt[tt[-1]] - 300.0) < 10.0


def test_decorrelation_rule_finds_autocorrelation_zero():
    from pisara import population as P
    rng = np.random.default_rng(4)
    T = world.intervals("chaotic", 600, rng)
    s, _ = world.drip_train(T)
    y = P.leaky_trace(s)
    tau, _ = P.decorrelation_delay(y)
    yc = y - y.mean()
    ac = np.array([np.mean(yc[L:] * yc[:-L]) for L in range(1, 300)]) / yc.var()
    assert abs(tau - (int(np.argmax(ac < 0)) + 1)) < 5
