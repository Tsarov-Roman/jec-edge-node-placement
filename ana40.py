import json, collections, numpy as np, math
lp = {(x['sc'], x['M'], x['seed']): x for x in json.load(open('e38lp.json'))}
R = [json.loads(l) for l in open('e40.jsonl')]
def cls(x):
    o = round(x['opt']); cap = 40 * x['M']
    return 'cap' if o == cap else ('cov' if o == 200 else 'geo')
C = collections.defaultdict(lambda: dict(n=0, h150=0, h750=0, runs=0, ls=0, greedy=0, rnd=[], worse=0, better=0, inst_fail=0, sf150=[], sf750=[], lsgap=[], lsev=[]))
row = collections.defaultdict(lambda: dict(n=0, h150=0, h750=0, runs=0, ls=0, rnd=[]))
mism = 0
for r in R:
    k = (r['sc'], r['M'], r['seed']); x = lp[k]; o = r['opt']; c = C[cls(x)]; w = row[(r['sc'], r['M'])]
    h150 = sum(abs(g['b150'] - o) < 1e-6 for g in r['ga']); h750 = sum(abs(g['b750'] - o) < 1e-6 for g in r['ga'])
    if h150 != x['ga_hit']: mism += 1
    for d in (c, w): d['n'] += 1; d['h150'] += h150; d['h750'] += h750; d['runs'] += 5; d['ls'] += abs(r['ls'] - o) < 1e-6; d['rnd'].append(r['rnd_opt'])
    c['greedy'] += abs(r['greedy'] - o) < 1e-6; c['lsgap'].append(o - r['ls']); c['lsev'].append(r['ls_ev'])
    c['sf150'] += [(o - g['b150']) / 2 for g in r['ga']]; c['sf750'] += [(o - g['b750']) / 2 for g in r['ga']]
    if h150 < 5: c['inst_fail'] += 1; c['better'] += h750 > h150
print('instances', len(R), 'base-hit mismatches vs e37', mism)
for k, c in C.items():
    print(k, c['n'], 'runs', c['runs'], 'h150', c['h150'], 'h750', c['h750'], 'G', c['greedy'], 'G+LS', c['ls'], 'lsgap max', max(c['lsgap']), 'ev', min(c['lsev']), max(c['lsev']),
          'rnd opt/5000 min-max', min(c['rnd']), max(c['rnd']), 'inst with zero rnd opt', sum(v == 0 for v in c['rnd']),
          'fail-inst', c['inst_fail'], 'improved', c['better'], 'sf150 mean', round(np.mean(c['sf150']), 2), 'sf750', round(np.mean(c['sf750']), 2))
for k in sorted(row): w = row[k]; print(k, w['n'], w['h150'], w['h750'], w['ls'], w['rnd'])
t = [g['t'] for r in R for g in r['ga']]; print('traj time', round(min(t), 1), round(max(t), 1))
