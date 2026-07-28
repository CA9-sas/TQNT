"""Test numerique de l'identification nu <-> n (brouillon spin-circulation-temps-emergent).

Hypothese du brouillon : l'enroulement de circulation chi quantifiee n (avec s = n/2)
serait EGAL a la saturation torsionnelle nu = N_sc(tau)/2 n0 de V6 (crit. spin (nu,mu,Q)).

RESULTAT : REFUTE au-dela du spin-1/2.
  - Famille torique (spin-1/2, courbes VALIDES) : nu = 1.000 et n = 2s = 1  -> nu = n. OK.
  - spin 1 et 3/2 : nu in [0.14, 0.80] mais n = 2 ou 3 -> nu != n.
  - corr(nu, n=2s) = -0.75 : ANTI-correles (nu sature a 1 pour n=1 puis decroit).
  - L'enroulement correct est n = 2s = b-1 (indice de tresse), PAS nu.
  - nu = 1 est seulement l'INDICATEUR du spin-1/2 (nu=1 <=> n=1), theoreme deja dans V6.

N.B. les courbes non-toriques du catalogue sont topologiquement fausses (cf. memoire
catalog-curves-invalid) ; nu y est indicatif, mais la conclusion (anti-correlation, nu!=n)
est robuste car nu <= 1 toujours tandis que n = 2s croit a 2, 3.
"""
import numpy as np
from theory.abraham import frenet_geometry
from theory.jones import jones_polynomial, braid_index, topological_spin

TORUS = {'3_1', '5_1', '7_1', '9_1'}
KNOTS = ['3_1', '5_1', '7_1', '9_1', '4_1', '5_2', '6_1', '6_2', '7_2', '8_1',
         '7_4', '8_2', '6_3', '7_3', '7_5', '8_3']


def n_sign_changes_torsion(name, npts=4000):
    """Nombre de changements de signe de la torsion de Frenet sur la boucle fermee."""
    tau = np.asarray(frenet_geometry(name, npts)['tau'])
    s = np.sign(tau)
    s = s[s != 0]
    n = int(np.sum(np.abs(np.diff(s)) > 0))
    if s[0] != s[-1]:
        n += 1
    return n


def run():
    print(f"{'K':5}{'valide':>7}{'b':>3}{'s':>5}{'n=2s':>6}{'n0':>4}{'Nsc':>5}{'nu':>7}{'nu==n?':>8}")
    nus, ns = [], []
    for K in KNOTS:
        p = jones_polynomial(K)
        n0 = len(p)                      # nombre de termes de V_K  (c = n0 - 1)
        b = braid_index(K)
        s = topological_spin(K)
        nsc = n_sign_changes_torsion(K)
        nu = nsc / (2 * n0)              # nu = N_sc / 2(c+1) = N_sc / 2 n0
        n = 2 * s                        # enroulement suppose (s = n/2)
        val = 'oui' if (K in TORUS or K == '4_1') else 'NON*'
        eq = 'OUI' if abs(nu - n) < 0.15 else 'non'
        nus.append(nu); ns.append(n)
        print(f"{K:5}{val:>7}{b:>3}{s:>5}{n:>6}{n0:>4}{nsc:>5}{nu:>7.3f}{eq:>8}")
    nus, ns = np.array(nus), np.array(ns)
    print(f"\ncorr(nu, n=2s) = {np.corrcoef(nus, ns)[0, 1]:+.3f}  (anti-correle => nu n'est PAS n)")
    # verification que n = b - 1 pour tous
    ok = all(2 * topological_spin(K) == braid_index(K) - 1 for K in KNOTS)
    print(f"n = 2s = b-1 (indice de tresse) verifie sur les 16 noeuds : {ok}")
    print("Conclusion : identification nu<->n REFUTEE (spin>1/2) ; enroulement = b-1, pas nu ;")
    print("             nu=1 reste l'indicateur du spin-1/2 (nu=1 <=> n=1).")


if __name__ == "__main__":
    run()
