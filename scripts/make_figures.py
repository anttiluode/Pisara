"""Figures for the README."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pisara import loop, world  # noqa: E402

Path("fig").mkdir(exist_ok=True)

# 1. period-two faucet: drops and the loop's returning signal at D = a + b
rng = np.random.default_rng(21)
T = world.intervals("period2", 60, rng)
s, tt = world.drip_train(T)
x, _ = loop.run(s, 600.0)
t0, t1 = tt[40], tt[48]
tax = np.arange(t0, t1)
fig, ax = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True)
ax[0].plot(tax - t0, s[t0:t1], color="#1f5f99", lw=1)
ax[0].set_ylabel("drops")
ax[0].set_yticks([])
ax[1].plot(tax - t0, x[t0 - 600:t1 - 600], color="#c0582b", lw=1.2, label="returning signal x(t - D), D = 600 ms")
ax[1].plot(tax - t0, x[t0:t1], color="#888", lw=0.8, alpha=0.7, label="node now x(t)")
for t in tt[40:48]:
    ax[1].axvline(t - t0, color="#1f5f99", lw=0.6, alpha=0.5)
ax[1].legend(loc="upper right", fontsize=8)
ax[1].set_xlabel("ms")
fig.suptitle("Period-two faucet (240 / 360 ms). The loop's echo of the drop two back arrives when the next drop does.", fontsize=9)
fig.tight_layout()
fig.savefig("fig/period2_echo.png", dpi=120)

# 2. switching world: learned delay vs true interval
g2 = json.loads(Path("results/gate2.json").read_text())
b = g2["B"][0]
rng = np.random.default_rng(b["seed"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gate2 import SEGMENTS  # noqa: E402

Ts, starts = world.schedule(SEGMENTS, rng)
D = np.array(b["D_trace_at_drops"])
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(Ts, ".", ms=2.5, color="#1f5f99", label="drop interval T_n (the world)")
ax.plot(D, color="#c0582b", lw=1.4, label="learned loop delay D")
for a, (reg, n, base) in zip(starts, SEGMENTS):
    ax.axvline(a, color="k", lw=0.5, alpha=0.4)
    ax.text(a + 3, 1430, f"{reg} {base:.0f}", fontsize=7)
ax.set_xlabel("drop number")
ax.set_ylabel("ms")
ax.set_ylim(0, 1550)
ax.legend(loc="center left", fontsize=8)
ax.set_title("The delay follows the clock 300 -> 420 ms, sits at the mean on period-two (not the 720 ms repeat), "
             "and ratchets up harmonics on the draining filter", fontsize=9)
fig.tight_layout()
fig.savefig("fig/learned_delay_switch.png", dpi=120)
print("figures written")
