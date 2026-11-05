import sys, os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
from common import *
from e37 import adm, SITES, LMAX
from run_exp import BASE
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
d = json.load(open(os.path.join(H, 'layout_0.json'))); U = np.array(d['U'])
p = dict(BASE); p.update(W=20.0, Lw=19.0); A = adm(p, 0)
def covered(S):
    N = A.shape[0]; m = len(S); src, snk = 0, 1 + N + m; ii, jj, cc = [], [], []
    for i in range(N): ii.append(src); jj.append(1 + i); cc.append(1)
    sub = A[:, S]
    for i, k in zip(*np.nonzero(sub)): ii.append(1 + i); jj.append(1 + N + k); cc.append(1)
    for k in range(m): ii.append(1 + N + k); jj.append(snk); cc.append(LMAX)
    G = csr_matrix((np.array(cc, dtype=np.int32), (ii, jj)), shape=(snk + 1, snk + 1))
    F = maximum_flow(G, src, snk).flow.toarray()
    return F[0, 1:1 + N] > 0, F[0].sum()
k = 2; vga, Sga = d['ga'][k]
fig, axs = plt.subplots(1, 2, figsize=(16 * CM, 7.6 * CM))
for ax, S, ttl in ((axs[0], d['S'], f"(a) branch and bound: {int(d['opt'])} covered"), (axs[1], Sga, f"(b) GA run: {int(vga)} covered")):
    cov, tot = covered(list(S))
    for x in range(0, 101, 20): ax.plot([x, x], [0, 100], color='0.55', lw=0.8); ax.plot([0, 100], [x, x], color='0.55', lw=0.8)
    ax.plot(U[cov, 0], U[cov, 1], '.', color='0.55', ms=3.2)
    ax.plot(U[~cov, 0], U[~cov, 1], 'x', color='black', ms=4.5, mew=1.0)
    P = SITES[list(S)]; ax.plot(P[:, 0], P[:, 1], '^', mfc='white', mec='black', ms=8, mew=1.3)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect('equal'); ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(False); ax.set_title(ttl, fontsize=9.5, loc='left'); ax.tick_params(labelsize=8)
    print(ttl, int(tot), int((~cov).sum()))
axs[0].set_ylabel('y, m', fontsize=9); [a.set_xlabel('x, m', fontsize=9) for a in axs]
fig.tight_layout(); fig.savefig(os.path.join(H, 'fig', 'fig4.pdf')); fig.savefig(os.path.join(H, 'fig', 'fig4.png'))
