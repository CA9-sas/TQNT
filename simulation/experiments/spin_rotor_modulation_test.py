"""Test rigoureux : la modulation azimutale m=q est-elle l'origine du spin ?

Contexte : un noeud en rotation presente une modulation azimutale a l'ordre de sa symetrie
C_q (le trefle 3_1 a un mode m=3 dominant -- PAS une modulation nulle ; R_Q=||Q||/(|m|R_g) ne
mesurait que le quadrupole m=2, interdit par C_{q>=3}).

Idee testee (option 2 de l'utilisateur) : est-ce l'EXISTENCE d'une periodicite sous rotation
qui quantifie le moment angulaire (rend L discret), la valeur 1/2 venant du double revetement (b) ?

Modele : rotor plan H = L_z^2/2 + V0 cos(q phi), L_z=-i d/dphi, base e^{i m phi}.
  - secteur PERIODIQUE     psi(phi+2pi)=+psi : m in Z      -> L_z entiers
  - secteur ANTIPERIODIQUE psi(phi+2pi)=-psi : m in Z+1/2  (double revetement = b=2) -> demi-entiers
cos(q phi) couple m <-> m +/- q.

RESULTAT :
  - min|L_z| = 0 (periodique) vs 1/2 (antiperiodique), FIXE par le bord, INDEPENDANT de V0.
  - la modulation V0 couple m<->m±q DANS un secteur (bandes) mais ne cree JAMAIS de demi-entier.
  => la modulation N'EST PAS l'origine du spin.
     * la discretisation de L vient de la COMPACITE de phi (tout rotor), pas de la modulation ;
     * le 1/2 vient du secteur ANTIPERIODIQUE = double revetement = indice de tresse b=2.
  La modulation n'ajoute qu'une structure de bandes (+ l'anisotropie du tenseur d'inertie, Q1).
"""
import numpy as np


def rotor(q, V0, half, mmax=24):
    """Spectre du rotor plan avec modulation cos(q phi), secteur (anti)periodique."""
    if half:
        ms = np.arange(-mmax + 0.5, mmax + 0.5, 1.0)   # demi-entiers
    else:
        ms = np.arange(-mmax, mmax + 1, 1.0)           # entiers
    H = np.diag(0.5 * ms ** 2)
    for i in range(len(ms)):
        for j in range(len(ms)):
            if abs(ms[i] - ms[j]) == q:
                H[i, j] += V0 / 2
    return ms, np.sort(np.linalg.eigvalsh(H))


def run():
    print("=== min|L_z| par secteur (independant de la modulation V0) ===")
    for V0 in (0.0, 2.0):
        for half in (False, True):
            ms, _ = rotor(3, V0, half)
            sect = 'ANTIperiodique (double rev., b=2)' if half else 'periodique'
            print(f"  V0={V0}  {sect:34} min|L_z|={min(abs(ms)):.1f}")

    print("\n=== spectre (4 niveaux bas), trefle q=3 ===")
    for half in (False, True):
        lab = 'ANTIperiodique (demi-entier)' if half else 'periodique (entier)'
        _, e0 = rotor(3, 0.0, half)
        _, e2 = rotor(3, 2.0, half)
        print(f"  {lab:30} V0=0:{np.round(e0[:4], 3)}  V0=2:{np.round(e2[:4], 3)}")

    print("\nCONCLUSION : la modulation ne cree pas de demi-entier ; le spin-1/2 vient du")
    print("secteur antiperiodique (double revetement = b=2), pas de la modulation.")
    print("Discretisation de L = compacite de phi ; valeur 1/2 = double revetement. Q2 REFUTEE.")


if __name__ == "__main__":
    run()
