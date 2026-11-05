import json, os, sys, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import *
CALC = HERE; OUT = os.path.join(HERE, "fig")
lp = json.load(open(os.path.join(CALC, "e38lp.json")))
E = {(r["sc"], r["M"], r["seed"]): r for r in map(json.loads, open(os.path.join(CALC, "e40.jsonl")))}
def save(fig, n): fig.savefig(f"{OUT}/{n}.png"); fig.savefig(f"{OUT}/{n}.pdf"); plt.close(fig)
plt.rcParams.update({"font.size": 9})
W = 8.2 * CM
# fig1: hit rate vs M for two hard scenarios, 1x and 5x
agg = collections.defaultdict(lambda: [0, 0, 0, 0, [], [], 0, 0])
for x in lp:
    o = round(x["opt"]); a = agg[(x["sc"], x["M"])]; r = E[(x["sc"], x["M"], x["seed"])]
    b1 = [g["b150"] for g in r["ga"]]; b2 = [g["b750"] for g in r["ga"]]
    a[0] += sum(abs(v - o) < 1e-6 for v in b1); a[1] += 5; a[4].append((o - np.mean(b1)) / 2)
    a[2] += sum(abs(v - o) < 1e-6 for v in b2); a[3] += 5; a[5].append((o - np.mean(b2)) / 2)
    a[6] += abs(r["ls"] - o) < 1e-6; a[7] += 1
fig, axs = plt.subplots(2, 1, figsize=(W, 10.2 * CM), sharex=True)
for sc, lab, mk in (("w20_19", "20 m rooms, 19 dB", "s"), ("w10_10", "10 m rooms, 10 dB", "D")):
    Ms = sorted(M for (s, M) in agg if s == sc)
    for j, ls, tag in ((0, "-", "6 000 eval."), (2, "--", "30 000 eval.")):
        axs[0].plot(Ms, [100 * agg[(sc, M)][j] / agg[(sc, M)][j + 1] for M in Ms], ls=ls, marker=mk, mfc="white" if j == 0 else "black", color="black", ms=4, label=f"{lab}, {tag}")
        axs[1].plot(Ms, [np.mean(agg[(sc, M)][4 if j == 0 else 5]) for M in Ms], ls=ls, marker=mk, mfc="white" if j == 0 else "black", color="black", ms=4)
    axs[0].plot(Ms, [100 * agg[(sc, M)][6] / agg[(sc, M)][7] for M in Ms], ls=":", marker=mk, mfc="0.6", color="0.45", ms=4, label=f"{lab}, greedy + LS")
axs[0].set_ylabel("Runs reaching\nthe optimum, %"); fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=6.4, loc="lower center", ncol=2, columnspacing=0.8, handlelength=2.2)
axs[1].set_ylabel("Mean shortfall, p.p."); axs[1].set_xlabel("Number of nodes $M$")
for a, t in zip(axs, "ab"): a.set_title(f"({t})", loc="left", fontsize=9, pad=2)
axs[1].set_ylim(-0.1, 3.2); axs[1].set_yticks([0, 1, 2, 3])
fig.tight_layout(rect=(0, 0.12, 1, 1)); save(fig, "fig1")
# fig2: flowchart
fig, ax = scheme(8.2, 10.8)
Y = [0.9, 0.645, 0.37, 0.095]
q = ["Corner optimum:\nfinite candidate set?", "Linear model on\na candidate grid?", "Relaxation bound\navailable for the\nnonlinear model?", "Metaheuristic without\na quality guarantee"]
a = ["enumerate\ncandidates", "branch and bound\n(MILP)", "metaheuristic +\ncertificate by bound", ""]
for i, (y, t) in enumerate(zip(Y, q)):
    box(ax, 0.25, y, 0.46, 0.15 if i != 2 else 0.18, t, fs=7.6, fc="0.93")
    if a[i]:
        box(ax, 0.8, y, 0.36, 0.13, a[i], fs=7.4); arrow(ax, (0.48, y), (0.62, y), "yes", fs=7)
    if i < 3: arrow(ax, (0.25, y - (0.075 if i != 2 else 0.09)), (0.25, Y[i + 1] + (0.075 if i + 1 != 2 else 0.09)), "no", fs=7, off=(0.05, 0))
save(fig, "fig2")
# fig3: розрив лінійної релаксації z_LP - z* за кількістю вузлів (сценарії зі стінами 19 дБ і 10 дБ)
fig, ax = plt.subplots(figsize=(W, 5.6 * CM))
for k, (sc, lab, mk, dx) in enumerate((("w20_19", "20 m rooms, 19 dB", "s", -0.08), ("w10_10", "10 m rooms, 10 dB", "D", 0.08))):
    Ms = sorted({x["M"] for x in lp if x["sc"] == sc})
    for M in Ms:
        g = [max(0.0, x["lp"] - x["opt"]) for x in lp if x["sc"] == sc and x["M"] == M]
        ax.plot([M + dx] * len(g), g, mk, mfc="white" if k else "0.35", mec="black", ms=4, ls="none", label=lab if M == Ms[0] else None)
ax.set_xlabel("Number of nodes $M$"); ax.set_ylabel("$z_{\\mathrm{LP}}-z^{*}$, users")
ax.set_xticks(range(3, 9)); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); save(fig, "fig3")
