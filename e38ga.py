"""Extended GA budget (40 x 750 = 30 000 evaluations) with independent seeds 1000+k, instances where the base GA failed."""
import json, time, numpy as np
from multiprocessing import Pool
from e37 import adm, ga_cov, BASE
SC = {"open": {}, "w20_5.3": dict(W=20.0, Lw=5.3), "w20_19": dict(W=20.0, Lw=19.0), "w10_10": dict(W=10.0, Lw=10.0)}
def job(a):
    sc, M, seed, r = a
    p = dict(BASE); p.update(SC[sc]); A = adm(p, seed)
    t = time.time(); fb, ev = ga_cov(A, M, np.random.default_rng(1000 + r), pop=40, gens=750)
    return dict(sc=sc, M=M, seed=seed, run=r, best=float(fb), evals=int(ev), t=time.time() - t)
if __name__ == "__main__":
    lp = json.load(open("e38lp.json"))
    hard = [(x["sc"], x["M"], x["seed"]) for x in lp if x["ga_hit"] < 5]
    jobs = [h + (r,) for h in hard for r in range(5)]
    print(len(hard), len(jobs), flush=True)
    with Pool(2) as pool:
        res = []
        for i, x in enumerate(pool.imap_unordered(job, jobs)):
            res.append(x)
            if i % 10 == 0: print(i, x, flush=True); json.dump(res, open("e38ga.json", "w"), indent=1)
    json.dump(res, open("e38ga.json", "w"), indent=1); print("done", flush=True)
