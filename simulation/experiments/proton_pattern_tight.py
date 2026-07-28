import sys
sys.path.insert(0, '/home/euk9/projets/physique_nouvelle')
sys.path.insert(0, '/home/euk9/projets/physique_nouvelle/simulation/experiments')
import numpy as np
from simulation.ideal_embedding import sono_link, anneal, squeeze, linking_number, curve_length
from derive_inductance_factor_test import L_gmd, energy_MeV, TARGET, D_TUBE, FM, MU0
from repair_curves_braid import alexander

def guard(comps):
    lks = (linking_number(comps[0], comps[1]), linking_number(comps[0], comps[2]),
           linking_number(comps[1], comps[2]))
    al = [tuple(int(x) for x in alexander(C)[0]) for C in comps]
    ok = (tuple(abs(l) for l in lks) == (1, 1, 1)
          and al[0] == (1, -1, 1) and al[1] == (3, -5, 3) and al[2] == (3, -5, 3))
    return ok, lks

comps = [np.load(f'/tmp/proton_{t}.npy') for t in ('K', 'S1', 'S2')]
print('Lk initiaux :', guard(comps)[1], flush=True)

# serrage long
comps, info = sono_link(comps, iters=40000, log_every=8000)
ok, lks = guard(comps)
print('serrage 40k : rho = %.2f gap = %.3f Lk = %s gardes %s' % (
    info['ropelength'], info['min_gap'], lks, 'OK' if ok else 'KO'), flush=True)
for h in info['history']:
    print('  it %5d rho %7.2f' % h[:2], flush=True)

# rechauffes graduees s >= 0.7 (regle de surete entrelacs)
best, best_rho = comps, info['ropelength']
rng = np.random.default_rng(17)
for sweep in range(2):
    for s in (0.7, 0.8, 0.9):
        for t in range(2):
            u = rng.normal(size=3)
            sq = [squeeze(C, u, s) for C in best]
            c2, i2 = sono_link(sq, iters=12000, loosen_start=False)
            ok2, lks2 = guard(c2)
            print('  sweep %d s=%.1f essai %d : rho %7.2f %s' % (
                sweep, s, t, i2['ropelength'], 'OK' if ok2 else 'KO ' + str(lks2)), flush=True)
            if ok2 and i2['ropelength'] < best_rho - 1e-3:
                best, best_rho = c2, i2['ropelength']
                print('    -> nouveau meilleur : %.2f' % best_rho, flush=True)

ok, lks = guard(best)
print('FINAL proton : rho = %.2f Lk = %s gardes %s' % (best_rho, lks, 'OK' if ok else 'KO'), flush=True)
for tag, C in (('coeur 3_1 (proton)', best[0]), ('7_2 a', best[1]), ('7_2 b', best[2])):
    X = C * D_TUBE * FM
    L, var = L_gmd(X, D_TUBE * FM)
    E = energy_MeV(L)
    print('%-20s: l = %6.2f fm ; L = %7.3f mu0.fm (var %3.0f%%) ; E = %6.3f MeV (%+6.1f%%)' % (
        tag, curve_length(C) * D_TUBE, L / MU0 / FM, var, E, (E / TARGET - 1) * 100), flush=True)
for i, C in enumerate(best):
    np.save('/tmp/proton_tight_%d.npy' % i, C)
print('FINI', flush=True)
