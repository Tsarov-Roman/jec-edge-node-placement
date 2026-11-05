"""Weak-duality check of the LP bound from the HiGHS dual values for the 64 instances."""
import sys, os, json, time, math, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
from e37 import adm, NS, LMAX, BASE
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix
SC = {"open": {}, "w20_5.3": dict(W=20.0, Lw=5.3), "w20_19": dict(W=20.0, Lw=19.0), "w10_10": dict(W=10.0, Lw=10.0)}
def build(A, M):
    N = A.shape[0]; pairs = np.argwhere(A); P = len(pairs); nv = NS + N + P
    c = np.zeros(nv); c[NS:NS + N] = 1.0                      # max sum z
    Aub = lil_matrix((N + NS + P, nv)); r = 0
    for i in range(N):
        Aub[r, NS + i] = 1
        for k in np.flatnonzero(pairs[:, 0] == i): Aub[r, NS + N + k] = -1
        r += 1
    for j in range(NS):
        for k in np.flatnonzero(pairs[:, 1] == j): Aub[r, NS + N + k] = 1
        Aub[r, j] = -LMAX; r += 1
    for k, (i, j) in enumerate(pairs):
        Aub[r, NS + N + k] = 1; Aub[r, j] = -1; r += 1
    Aeq = np.zeros((1, nv)); Aeq[0, :NS] = 1
    return c, csr_matrix(Aub), np.zeros(N + NS + P), csr_matrix(Aeq), np.array([M], float)
out = []
for x in json.load(open(os.path.join(H, 'e38lp.json'))):
    p = dict(BASE); p.update(SC[x['sc']]); A = adm(p, x['seed']); M = x['M']
    c, Aub, bub, Aeq, beq = build(A, M)
    t = time.time(); res = linprog(-c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=(0, 1), method='highs'); tl = time.time() - t
    mu = np.maximum(0.0, -res.ineqlin.marginals); lam = -res.eqlin.marginals
    red = c - (Aub.T @ mu) - (Aeq.T @ lam).ravel()
    ub = float(beq @ lam + bub @ mu + np.maximum(0.0, red).sum())
    zp = -res.fun
    rec = dict(sc=x['sc'], M=M, seed=x['seed'], opt=x['opt'], lp_primal=zp, lp_old=x['lp'], dual_ub=ub, diff=ub - zp,
               floor_old=math.floor(x['lp'] + 1e-6), floor_dual=math.floor(ub + 1e-9), floor_dual_raw=math.floor(ub), t=tl)
    out.append(rec); print(x['sc'], M, x['seed'], round(zp, 6), round(ub, 9), rec['floor_old'], rec['floor_dual'], rec['floor_dual_raw'], round(x['opt']), flush=True)
json.dump(out, open(os.path.join(H, 'e41.json'), 'w'), indent=1)
