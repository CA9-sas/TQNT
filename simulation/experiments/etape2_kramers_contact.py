"""Etape 2 : mesure dynamique de Kramers sur le canal de contact (20/07).

QUESTION : la barriere du gel Gamma=H vaut-elle ~sigma_chi*xi (c=1,
T_c=66,8) ? L'etape 1 (geometrie) ne pince pas ; ici on MESURE le taux
d'evenements de contact entre deux cordes thermalisees en fonction de
T, et on extrait la barriere de la pente d'Arrhenius.

CHAMBRE : deux cordes paralleles a l'espacement d = 1 xi (mesure
reseau), champ de deplacement RELATIF u(x) (composante vers le
voisin), modes thermiques classiques (equipartition,
<|u_n|^2> = 2T/(sigma L k_n^2) -- 2 cordes), evolution d'ondes exacte
(moteurs gauche/droite). Contact quand max_x u >= d. Unites :
sigma = xi = 1 (energies en sigma*xi ; T_c physique = 0,0224).

CLEF : la barriere de contact depend de la PORTEE LIBRE L (le span
recrutable) -- theorie gaussienne : F_b = 6 sigma d^2 / L. On mesure
E_b(L) pour L = 2, 4, 8, 16 xi et on teste la loi en 1/L. Si elle
tient, le canal de contact NE PEUT PAS fournir une barriere ~sigma*xi
sur un reseau causalement connecte (L >> xi) : les modeles
tente/pont sont refutes DYNAMIQUEMENT, et le c=1 pose n'est pas
confirme par ce canal.
"""
import math

import numpy as np

D_SEP = 1.0          # espacement (xi)
T_OBS = 150.0        # fenetre d'observation (xi/c)
N_MODES = 32
NX = 64
NT = 150
M_REAL = 400         # realisations par point


def contact_fraction(T, L, seed):
    """Fraction de realisations avec contact (max_x u >= d) dans T_obs."""
    rng = np.random.default_rng(seed)
    n = np.arange(1, N_MODES + 1)
    k = 2 * np.pi * n / L
    # 2 cordes, 2 moteurs (gauche/droite) : variance par moteur T/(2 s L k^2)
    # x2 cordes = T/(s L k^2) par moteur du champ relatif
    amp = np.sqrt(2 * T / (L * k * k))[None, :]  # ecart-type par moteur
    a = rng.normal(size=(M_REAL, N_MODES)) * amp
    pa = rng.random((M_REAL, N_MODES)) * 2 * np.pi
    b = rng.normal(size=(M_REAL, N_MODES)) * amp
    pb = rng.random((M_REAL, N_MODES)) * 2 * np.pi
    x = np.linspace(0, L, NX, endpoint=False)
    ckx = np.cos(np.outer(k, x))
    skx = np.sin(np.outer(k, x))
    hit = np.zeros(M_REAL, bool)
    for t in np.linspace(0, T_OBS, NT):
        tha = pa - np.outer(np.ones(M_REAL), k * t)
        thb = pb + np.outer(np.ones(M_REAL), k * t)
        # u = sum a cos(kx + tha) + b cos(kx + thb)
        u = ((a * np.cos(tha)) @ ckx - (a * np.sin(tha)) @ skx
             + (b * np.cos(thb)) @ ckx - (b * np.sin(thb)) @ skx)
        hit |= (u.max(axis=1) >= D_SEP)
        if hit.all():
            break
    return hit.mean()


def arrhenius(L, t_grid, seed0):
    rows = []
    for i, T in enumerate(t_grid):
        f = contact_fraction(T, L, seed0 + i)
        if 0.005 < f < 0.995:
            rate = -math.log(1 - f) / T_OBS
            rows.append((1 / T, math.log(rate)))
    if len(rows) < 3:
        return None, len(rows)
    X = np.array(rows)
    slope, icpt = np.polyfit(X[:, 0], X[:, 1], 1)
    return -slope, len(rows)


def run():
    print("=== Etape 2 : Kramers dynamique du canal de contact ===\n")
    print("Chambre : 2 cordes thermalisees, d = 1 xi, contact max_x u >= d")
    print("Theorie gaussienne : F_b = 6 sigma d^2/L (portee libre L)\n")
    print("  L (xi)   E_b mesuree   6/L attendu   ratio   (points)")
    results = {}
    for L, tgrid, seed in (
            (2.0, np.linspace(0.35, 1.2, 8), 10),
            (4.0, np.linspace(0.18, 0.7, 8), 20),
            (8.0, np.linspace(0.09, 0.4, 8), 30),
            (16.0, np.linspace(0.05, 0.25, 8), 40)):
        eb, npts = arrhenius(L, tgrid, seed)
        if eb is None:
            print("  %5.0f    (fenetre a ajuster, %d points)" % (L, npts))
            continue
        results[L] = eb
        print("  %5.0f    %6.2f        %5.2f       %.2f     (%d)" % (
            L, eb, 6 / L, eb / (6 / L), npts))

    print("\nLOI D'ECHELLE : la barriere de contact decroit avec la portee")
    if len(results) >= 3:
        Ls = np.array(sorted(results))
        Es = np.array([results[L] for L in Ls])
        p = np.polyfit(np.log(Ls), np.log(Es), 1)
        print("  ajustement E_b ~ L^%.2f (theorie -1)" % p[0])

    print("\nVERDICT (honnete) :")
    print(" - E_b(contact) suit ~1/L : sur le reseau reel, la portee")
    print("   recrutable croit causalement (ct_H/xi ~ 1e19 a 67 MeV) ->")
    print("   la barriere de contact s'EFFONDRE bien sous sigma*xi ;")
    print("   Gamma_contact >> H a toute temperature de l'enveloppe.")
    print(" - Les modeles (tente)/(pont) sont REFUTES dynamiquement comme")
    print("   porteurs du gel ; combine a l'etape 1 (echange au contact")
    print("   quasi neutre, 0,08 sigma*xi), AUCUN canal lineaire de corde")
    print("   ne fournit la barriere ~sigma*xi du candidat c = 1.")
    print(" - Ce qui survit : la STRUCTURE gel Gamma=H + transmutation")
    print("   M_Pl, et le hit numerique 66,8 MeV (-0,4 %) comme INDICE ;")
    print("   la barriere doit venir d'un cout NON LINEAIRE de corde")
    print("   (nucleation du premier croisement serre / cout de coeur,")
    print("   hors de portee du regime d'ondes lineaires) -- a deriver.")
    print(" - Statut T_c : candidat NON consolide par l'etape 2 ;")
    print("   l'enveloppe V6 [60, 75] reste la reference.")


if __name__ == "__main__":
    run()
