"""B&B with continuous and with binary assignment variables, and LP relaxation, for the 64 instances."""
import json, time, numpy as np, sys
from e37 import adm, NS, LMAX, BASE
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
SC = {"open": {}, "w20_5.3": dict(W=20.0, Lw=5.3), "w20_19": dict(W=20.0, Lw=19.0), "w10_10": dict(W=10.0, Lw=10.0)}
def build(A, M):
    N = A.shape[0]; pairs = np.argwhere(A); P = len(pairs); nv = NS + N + P
    c = np.zeros(nv); c[NS:NS + N] = -1
    Acon = lil_matrix((1 + N + NS + P, nv)); lo = []; hi = []; r = 0
    Acon[r, :NS] = 1; lo.append(M); hi.append(M); r += 1
    for i in range(N):                                   # sum_j u_ij - z_i = 0
        Acon[r, NS + i] = -1
        for k in np.flatnonzero(pairs[:, 0] == i): Acon[r, NS + N + k] = 1
        lo.append(0); hi.append(0); r += 1
    for j in range(NS):                                  # sum_i u_ij <= L y_j
        for k in np.flatnonzero(pairs[:, 1] == j): Acon[r, NS + N + k] = 1
        Acon[r, j] = -LMAX; lo.append(-np.inf); hi.append(0); r += 1
    for k, (i, j) in enumerate(pairs):                   # u_ij <= a_ij y_j (a_ij = 1 on defined pairs)
        Acon[r, NS + N + k] = 1; Acon[r, j] = -1; lo.append(-np.inf); hi.append(0); r += 1
    return c, LinearConstraint(csr_matrix(Acon), lo, hi), nv, P
def solve(c, con, integ):
    t = time.time(); res = milp(c, constraints=con, integrality=integ, bounds=Bounds(0, 1), options=dict(time_limit=600))
    return -res.fun, time.time() - t, int(res.status), getattr(res, "mip_node_count", None), getattr(res, "mip_gap", None)
if __name__ == "__main__":
    prev = json.load(open("e38lp.json")); out = []
    for x in prev:
        p = dict(BASE); p.update(SC[x["sc"]]); A = adm(p, x["seed"]); c, con, nv, P = build(A, x["M"]); N = A.shape[0]
        z_c, t_c, s_c, n_c, _ = solve(c, con, np.r_[np.ones(NS + N), np.zeros(P)])
        z_b, t_b, s_b, n_b, g_b = solve(c, con, np.ones(nv))
        z_l, t_l, _, _, _ = solve(c, con, np.zeros(nv))
        rec = dict(sc=x["sc"], M=x["M"], seed=x["seed"], pairs=int(P), opt_old=x["opt"], z_cont=z_c, t_cont=t_c, st_cont=s_c, nodes_cont=n_c,
                   z_bin=z_b, t_bin=t_b, st_bin=s_b, nodes_bin=n_b, gap_bin=g_b, lp=z_l, t_lp=t_l, ga_hit=x["ga_hit"], ga_mean=x["ga_mean"], t_ga=x["t_ga"])
        out.append(rec); print(x["sc"], x["M"], x["seed"], round(z_c, 3), round(z_b, 3), round(x["opt"]), round(z_l, 2), f"{t_c:.1f} {t_b:.1f} {t_l:.1f}", n_c, n_b, s_b, flush=True)
        json.dump(out, open("e39.json", "w"), indent=1, default=float)
    print("done")
