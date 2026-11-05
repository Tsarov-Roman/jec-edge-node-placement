"""Series 1: B&B (HiGHS via scipy.optimize.milp) and GA (6 000 evaluations), open / 20 m-5.3 dB / 20 m-19 dB, M = 3, 5, 8."""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import rx_power_w, noise_w
from run_exp import BASE, users_for
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.csgraph import maximum_flow
g = np.arange(5, 100, 10.0); SITES = np.array([(x, y) for x in g for y in g]); NS = len(SITES)
LMAX = 40

def adm(p, seed):
    U = users_for(seed, p); Ps = rx_power_w(U, SITES, p)
    snr = Ps / noise_w(p)
    return (p["B"] / LMAX) * np.log2(1 + snr) >= p["R_min"]          # (N, NS)

def milp_cov(A, M):
    N = A.shape[0]; pairs = np.argwhere(A); P = len(pairs)
    nv = NS + N + P                                            # y, z, x
    c = np.zeros(nv); c[NS:NS + N] = -1
    rows = []; lo = []; hi = []
    Acon = lil_matrix((1 + N + NS + P, nv)); r = 0
    Acon[r, :NS] = 1; lo.append(M); hi.append(M); r += 1       # sum y = M
    for i in range(N):                                          # z_i <= sum_j x_ij
        Acon[r, NS + i] = 1
        for k in np.flatnonzero(pairs[:, 0] == i): Acon[r, NS + N + k] = -1
        lo.append(-np.inf); hi.append(0); r += 1
    for j in range(NS):                                         # sum_i x_ij <= LMAX y_j
        for k in np.flatnonzero(pairs[:, 1] == j): Acon[r, NS + N + k] = 1
        Acon[r, j] = -LMAX; lo.append(-np.inf); hi.append(0); r += 1
    for k, (i, j) in enumerate(pairs):                          # x_ij <= y_j
        Acon[r, NS + N + k] = 1; Acon[r, j] = -1; lo.append(-np.inf); hi.append(0); r += 1
    integ = np.r_[np.ones(NS + N), np.zeros(P)]
    t = time.time()
    res = milp(c, constraints=LinearConstraint(csr_matrix(Acon), lo, hi), integrality=integ,
               bounds=Bounds(0, 1), options=dict(time_limit=600))
    return -res.fun, time.time() - t, res.status, getattr(res, "mip_node_count", None)

def cover(A, S):
    """Максимальна кількість покритих користувачів для множини вузлів S (максимальний потік)."""
    N = A.shape[0]; S = list(S); m = len(S)
    src, snk = 0, 1 + N + m
    ii, jj, cc = [], [], []
    for i in range(N): ii.append(src); jj.append(1 + i); cc.append(1)
    sub = A[:, S]
    for i, k in zip(*np.nonzero(sub)): ii.append(1 + i); jj.append(1 + N + k); cc.append(1)
    for k in range(m): ii.append(1 + N + k); jj.append(snk); cc.append(LMAX)
    G = csr_matrix((np.array(cc, dtype=np.int32), (ii, jj)), shape=(snk + 1, snk + 1))
    return maximum_flow(G, src, snk).flow_value

def ga_cov(A, M, rng, pop=40, gens=150):
    X = [rng.choice(NS, M, replace=False) for _ in range(pop)]
    f = np.array([cover(A, x) for x in X]); ev = pop
    for _ in range(gens):
        order = np.argsort(-f); new = [X[order[0]], X[order[1]]]
        while len(new) < pop:
            a = rng.choice(pop, 3); b = rng.choice(pop, 3)
            p1 = X[a[np.argmax(f[a])]]; p2 = X[b[np.argmax(f[b])]]
            pool = np.unique(np.r_[p1, p2]); c = rng.choice(pool, M, replace=False)
            if rng.uniform() < 0.3:
                k = rng.integers(M); cand = np.setdiff1d(np.arange(NS), c); c[k] = rng.choice(cand)
            new.append(c)
        X = new; f = np.array([cover(A, x) for x in X]); ev += pop
    return f.max(), ev

SC = {"open": {}, "w20_5.3": dict(W=20.0, Lw=5.3), "w20_19": dict(W=20.0, Lw=19.0)}
if __name__ == "__main__":
    out = []
    for sn, ov in SC.items():
        for M in (3, 5, 8):
            for seed in range(3):
                p = dict(BASE); p.update(ov); A = adm(p, seed)
                opt, tm, st, nodes = milp_cov(A, M)
                ga = []
                for r in range(5):
                    t = time.time(); fb, ev = ga_cov(A, M, np.random.default_rng(r)); ga.append((fb, time.time() - t))
                rec = dict(sc=sn, M=M, seed=seed, coverable=int(A.any(1).sum()), opt=opt, t_milp=tm, status=int(st), nodes=nodes,
                           ga_best=max(x[0] for x in ga), ga_mean=float(np.mean([x[0] for x in ga])), ga_hit=sum(abs(x[0] - opt) < 1e-6 for x in ga),
                           t_ga=float(np.mean([x[1] for x in ga])))
                out.append(rec); print(rec, flush=True)
    json.dump(out, open("e37.json", "w"), indent=1)
