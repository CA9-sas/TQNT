import sys
sys.path.insert(0, '/home/euk9/projets/physique_nouvelle')
sys.path.insert(0, '/home/euk9/projets/physique_nouvelle/simulation/experiments')
import numpy as np
from simulation.ideal_embedding import sono, squeeze
from repair_curves_braid import alexander, BRAID_WORDS
from ukappa_ideal_campaign import measures, ROPE_PUB

name = sys.argv[1]
TGT = tuple(BRAID_WORDS[name][2])
P0 = np.load('/tmp/seed_%s_plat.npy' % name)
P, info = sono(P0, n=280, iters=30000)
L, TC, LU, wr = measures(P)
print('%s serrage : rho %.2f Wr %+.2f' % (name, L, wr), flush=True)
best, best_rho = P, L
rng = np.random.default_rng(62)
for sweep in range(2):
    for s in (0.7, 0.8, 0.9):
        for t in range(2):
            Q, i2 = sono(squeeze(best, rng.normal(size=3), s), iters=10000, loosen_start=False)
            try:
                ok = tuple(int(x) for x in alexander(Q)[0]) == TGT
            except Exception:
                ok = False
            L2 = measures(Q)[0]
            if ok and L2 < best_rho - 1e-3:
                best, best_rho = Q, L2
                print('  -> nouveau meilleur : %.2f' % L2, flush=True)
L, TC, LU, wr = measures(best)
pub = ROPE_PUB[name]
print('RESULT %s rho=%.2f (%+.0f%% vs pub %.2f) Wr=%+.2f TC=%.2f LU=%.0f' % (
    name, L, (L/pub-1)*100, pub, wr, TC, LU), flush=True)
np.save('/tmp/ideal_%s_tight.npy' % name, best)
print('FINI', flush=True)
