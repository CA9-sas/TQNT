import sys
sys.path.insert(0, '/home/euk9/projets/physique_nouvelle')
sys.path.insert(0, '/home/euk9/projets/physique_nouvelle/simulation/experiments')
import numpy as np
from simulation.ideal_embedding import sono, squeeze
from repair_curves_braid import alexander, BRAID_WORDS
from ukappa_ideal_campaign import measures

TGT = tuple(BRAID_WORDS['8_3'][2])
P = np.load('/tmp/ideal_8_3.npy')
best, best_rho = P, measures(P)[0]
rng = np.random.default_rng(83)
for sweep in range(2):
    for s in (0.7, 0.8, 0.9):
        for t in range(2):
            Q = squeeze(best, rng.normal(size=3), s)
            Q2, info = sono(Q, iters=10000, loosen_start=False)
            try:
                ok = tuple(int(x) for x in alexander(Q2)[0]) == TGT
            except Exception:
                ok = False
            L, TC, LU, wr = measures(Q2)
            print('sweep %d s=%.1f essai %d : rho %6.2f Wr %+6.2f %s' % (
                sweep, s, t, L, wr, 'OK' if ok else 'KO'), flush=True)
            if ok and L < best_rho - 1e-3:
                best, best_rho = Q2, L
L, TC, LU, wr = measures(best)
print('FINAL 8_3 anneal : rho %.2f (publie 74.60) Wr %+.2f TC %.2f LU %.0f' % (L, wr, TC, LU), flush=True)
np.save('/tmp/ideal_8_3_annealed.npy', best)
print('FINI', flush=True)
