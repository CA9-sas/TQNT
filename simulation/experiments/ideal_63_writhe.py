import sys
sys.path.insert(0, '/home/euk9/projets/physique_nouvelle')
sys.path.insert(0, '/home/euk9/projets/physique_nouvelle/simulation/experiments')
import numpy as np
from simulation.ideal_embedding import plat_seed, sono, squeeze
from repair_curves_braid import alexander
from ukappa_ideal_campaign import measures

TGT = (1, -3, 5, -3, 1)   # 6_3, det 13, amphichiral
hits = []
for sa in (1, -1):
    for sb in (1, -1):
        for sc in (1, -1):
            for a in (1, 2, 3):
                for b in (1, 2, 3):
                    for c in (1, 2, 3):
                        w = [2*sa]*a + [1*sb]*b + [2*sc]*c
                        try:
                            P = plat_seed(w)
                            al = tuple(int(x) for x in alexander(P)[0])
                        except Exception:
                            continue
                        if al == TGT:
                            _, _, _, wr = measures(P)
                            hits.append((sa*a, sb*b, sc*c, wr, P))
                            print('6_3 : s2^%+d s1^%+d s2^%+d Wr=%+.2f' % (sa*a, sb*b, sc*c, wr), flush=True)
if not hits:
    # mots a 4 blocs (CF 13/5 = [2,1,1,2])
    for sa in (1, -1):
        for sb in (1, -1):
            for combo in ([2,1,1,2], [2,2,1,1], [1,2,2,1], [2,1,2,1]):
                w = []
                for i, n in enumerate(combo):
                    g = 2 if i % 2 == 0 else 1
                    sgn = sa if i % 2 == 0 else sb
                    w += [g*sgn]*n
                try:
                    P = plat_seed(w)
                    al = tuple(int(x) for x in alexander(P)[0])
                except Exception:
                    continue
                if al == TGT:
                    _, _, _, wr = measures(P)
                    hits.append((str(combo), sa, sb, wr, P))
                    print('6_3 (4 blocs) : %s sa=%+d sb=%+d Wr=%+.2f' % (combo, sa, sb, wr), flush=True)
if hits:
    P = hits[0][-1]
    Q, info = sono(P, n=280, iters=30000)
    L, TC, LU, wr = measures(Q)
    print('6_3 serre : rho %.2f (publie 54.42=27.21x2) Wr %+.2f' % (L, wr), flush=True)
    # rechauffes
    best, best_rho = Q, L
    rng = np.random.default_rng(63)
    for s in (0.7, 0.8, 0.9):
        for t in range(2):
            Q2, _ = sono(squeeze(best, rng.normal(size=3), s), iters=10000, loosen_start=False)
            try:
                ok = tuple(int(x) for x in alexander(Q2)[0]) == TGT
            except Exception:
                ok = False
            L2, _, _, wr2 = measures(Q2)
            if ok and L2 < best_rho - 1e-3:
                best, best_rho = Q2, L2
                print('  -> %.2f (Wr %+.2f)' % (L2, wr2), flush=True)
    L, TC, LU, wr = measures(best)
    print('RESULT 6_3 : rho=%.2f Wr=%+.2f TC=%.2f LU=%.0f (amphichiral : Wr~0 symetrique, |Wr|>0 brise)' % (L, wr, TC, LU), flush=True)
    np.save('/tmp/ideal_6_3.npy', best)
else:
    print('ECHEC : pas de 6_3 trouve', flush=True)
print('FINI', flush=True)
