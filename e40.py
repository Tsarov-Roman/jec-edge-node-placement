"""Paired GA trajectories (best value after 150 and 750 generations), greedy + single-swap local search, 5 000 random layouts."""
import sys, os, json, time, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
from e37 import adm, cover, NS, LMAX, BASE
from multiprocessing import Pool
SC = {"open": {}, "w20_5.3": dict(W=20.0, Lw=5.3), "w20_19": dict(W=20.0, Lw=19.0), "w10_10": dict(W=10.0, Lw=10.0)}

def ga_traj(A, M, rng, pop=40, gens=750, snap=(150, 750), opt=None):
    X = [rng.choice(NS, M, replace=False) for _ in range(pop)]
    f = np.array([cover(A, x) for x in X]); out = {}
    for g in range(1, gens + 1):
        order = np.argsort(-f); new = [X[order[0]], X[order[1]]]
        while len(new) < pop:
            a = rng.choice(pop, 3); b = rng.choice(pop, 3)
            p1 = X[a[np.argmax(f[a])]]; p2 = X[b[np.argmax(f[b])]]
            pool = np.unique(np.r_[p1, p2]); c = rng.choice(pool, M, replace=False)
            if rng.uniform() < 0.3:
                k = rng.integers(M); cand = np.setdiff1d(np.arange(NS), c); c[k] = rng.choice(cand)
            new.append(c)
        X = new; f = np.array([cover(A, x) for x in X])
        if g in snap: out[g] = float(f.max())
        if opt is not None and abs(f.max() - opt) < 1e-6:     # елітизм: далі найкраще не змінюється
            for h in snap:
                if h >= g: out[h] = float(f.max())
            out['gen_opt'] = g; break
    return out

def greedy_ls(A, M):
    ev = 0; S = []
    for _ in range(M):                                   # жадібне додавання
        best, bj = -1, None
        for j in range(NS):
            if j in S: continue
            v = cover(A, S + [j]); ev += 1
            if v > best: best, bj = v, j
        S.append(bj)
    fg = best; cur = best; improved = True
    while improved:                                      # локальний пошук: перше покращення однією заміною
        improved = False
        for i in range(M):
            for j in range(NS):
                if j in S: continue
                T = S.copy(); T[i] = j; v = cover(A, T); ev += 1
                if v > cur: S, cur, improved = T, v, True; break
            if improved: break
    return float(fg), float(cur), ev, sorted(int(x) for x in S)

def job(key):
    sc, M, seed, opt = key
    p = dict(BASE); p.update(SC[sc]); A = adm(p, seed)
    t = time.time(); g, ls, ev, S = greedy_ls(A, M); t_ls = time.time() - t
    rng = np.random.default_rng(777 + seed)
    rnd = np.array([cover(A, rng.choice(NS, M, replace=False)) for _ in range(5000)])
    runs = []
    for k in range(5):
        t = time.time(); tr = ga_traj(A, M, np.random.default_rng(k), opt=opt); runs.append(dict(b150=tr[150], b750=tr[750], gen_opt=tr.get('gen_opt'), t=time.time() - t))
    rec = dict(sc=sc, M=M, seed=seed, opt=opt, greedy=g, ls=ls, ls_ev=ev, ls_t=t_ls, ls_S=S,
               rnd_opt=int(np.sum(np.abs(rnd - opt) < 1e-6)), rnd_mean=float(rnd.mean()), rnd_max=float(rnd.max()), ga=runs)
    with open(os.path.join(H, 'e40.jsonl'), 'a') as fh: fh.write(json.dumps(rec) + '\n')
    return rec

if __name__ == "__main__":
    lp = json.load(open(os.path.join(H, 'e38lp.json')))
    done = set()
    if os.path.exists(os.path.join(H, 'e40.jsonl')):
        for l in open(os.path.join(H, 'e40.jsonl')): r = json.loads(l); done.add((r['sc'], r['M'], r['seed']))
    keys = [(x['sc'], x['M'], x['seed'], round(x['opt'])) for x in lp if (x['sc'], x['M'], x['seed']) not in done]
    if len(sys.argv) > 1: keys = keys[:int(sys.argv[1])]
    with Pool(2) as P:
        for r in P.imap_unordered(job, keys): print(r['sc'], r['M'], r['seed'], r['opt'], r['ls'], [x['b150'] for x in r['ga']], [x['b750'] for x in r['ga']], round(sum(x['t'] for x in r['ga'])), flush=True)
