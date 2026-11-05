"""Linear relaxation of the covering model for the 64 distinct instances, joined with B&B and GA results."""
import json, time, numpy as np
from e37 import adm, NS, LMAX, BASE
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
def build(A, M):
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
    return c, LinearConstraint(csr_matrix(Acon), lo, hi), nv, P
SC = {"open": {}, "w20_5.3": dict(W=20.0, Lw=5.3), "w20_19": dict(W=20.0, Lw=19.0), "w10_10": dict(W=10.0, Lw=10.0)}
if __name__ == "__main__":
    r = json.load(open("e37.json")) + json.load(open("e37b.json"))
    seen = {}; 
    for x in r: seen.setdefault((x["sc"], x["M"], x["seed"]), x)
    out = []
    for (sc, M, seed), x in seen.items():
        p = dict(BASE); p.update(SC[sc]); A = adm(p, seed)
        c, con, nv, P = build(A, M)
        t = time.time(); res = milp(c, constraints=con, integrality=np.zeros(nv), bounds=Bounds(0, 1)); tl = time.time() - t
        rec = dict(sc=sc, M=M, seed=seed, pairs=int(P), lp=-res.fun, opt=x["opt"], t_lp=tl, t_milp=x["t_milp"], ga_hit=x["ga_hit"], ga_mean=x["ga_mean"], t_ga=x["t_ga"])
        out.append(rec); print(sc, M, seed, P, round(-res.fun, 2), round(x["opt"]), round(tl, 2), round(x["t_milp"], 1), flush=True)
    json.dump(out, open("e38lp.json", "w"), indent=1, default=float)
