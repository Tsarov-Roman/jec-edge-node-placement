"""Figure 4 data: B&B and GA layouts for one geometry-bound instance (20 m rooms, 19 dB, M = 6, realization 0)."""
import sys, os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import e37
from e37 import adm, cover, ga_cov, SITES, NS, LMAX
from run_exp import BASE, users_for
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
def milp_sol(A, M):
    N = A.shape[0]; pairs = np.argwhere(A); P = len(pairs); nv = NS + N + P
    c = np.zeros(nv); c[NS:NS + N] = -1
    Acon = lil_matrix((1 + N + NS + P, nv)); lo = []; hi = []; r = 0
    Acon[r, :NS] = 1; lo.append(M); hi.append(M); r += 1
    for i in range(N):
        Acon[r, NS + i] = 1
        for k in np.flatnonzero(pairs[:, 0] == i): Acon[r, NS + N + k] = -1
        lo.append(-np.inf); hi.append(0); r += 1
    for j in range(NS):
        for k in np.flatnonzero(pairs[:, 1] == j): Acon[r, NS + N + k] = 1
        Acon[r, j] = -LMAX; lo.append(-np.inf); hi.append(0); r += 1
    for k, (i, j) in enumerate(pairs):
        Acon[r, NS + N + k] = 1; Acon[r, j] = -1; lo.append(-np.inf); hi.append(0); r += 1
    res = milp(c, constraints=LinearConstraint(csr_matrix(Acon), lo, hi), integrality=np.r_[np.ones(NS + N), np.zeros(P)], bounds=Bounds(0, 1), options=dict(time_limit=600))
    return -res.fun, np.flatnonzero(res.x[:NS] > 0.5)
def ga_best(A, M, rng, pop=40, gens=150):
    X = [rng.choice(NS, M, replace=False) for _ in range(pop)]
    f = np.array([cover(A, x) for x in X])
    for _ in range(gens):
        order = np.argsort(-f); new = [X[order[0]], X[order[1]]]
        while len(new) < pop:
            a = rng.choice(pop, 3); b = rng.choice(pop, 3)
            p1 = X[a[np.argmax(f[a])]]; p2 = X[b[np.argmax(f[b])]]
            pool = np.unique(np.r_[p1, p2]); c = rng.choice(pool, M, replace=False)
            if rng.uniform() < 0.3:
                k = rng.integers(M); cand = np.setdiff1d(np.arange(NS), c); c[k] = rng.choice(cand)
            new.append(c)
        X = new; f = np.array([cover(A, x) for x in X])
    i = int(np.argmax(f)); return f[i], X[i]
p = dict(BASE); p.update(W=20.0, Lw=19.0); M = 6; seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
A = adm(p, seed); U = users_for(seed, p)
opt, S = milp_sol(A, M)
runs = [ga_best(A, M, np.random.default_rng(r)) for r in range(5)]
out = dict(seed=seed, M=M, opt=opt, S=S.tolist(), ga=[(float(v), x.tolist()) for v, x in runs], U=U.tolist())
json.dump(out, open(os.path.join(H, f'layout_{seed}.json'), 'w'))
print(opt, [v for v, _ in runs], sorted(S.tolist()), [sorted(x.tolist()) for _, x in runs][:2])
