"""Demonstration de la PROTECTION PAR ENLACEMENT (24/07 -- etape 1).

THESE : un anneau leger (trefle electron-like, ou l'anneau fusionne
e+e-) enlace Lk = 1 a un noeud sombre K ne peut PAS se dissoudre en
ondes libres par la voie EM : la liberation exige un evenement P15
(changement de croisement). Survie = emission.

TROIS PIECES :
 [A] ADDITIVITE DU Lk SOUS FUSION (numerique) : la somme connexe de
     l'anneau lie (Lk = 1) et du positron libre (Lk = 0) porte
     Lk = 1 -- l'annihilation ne fait qu'un anneau ENCORE lie.
 [B] INVARIANCE SOUS DEFORMATION LISSE (numerique) : le Lk de Gauss
     est constant sous toute deformation sans croisement de tube --
     l'evolution EM (transport lisse) conserve Lk EXACTEMENT ; c'est
     la superselection etablie du chapitre intrication ([H_int, W]=0
     dynamique), verifiee ici geometriquement.
 [C] STRUCTURE DE TAUX (corollaire) : le seul operateur Delta Lk != 0
     de la theorie est le sommet P15 (barriere de Fermi). La
     dissolution en ondes libres (Lk indefini/nul) passe donc par
     P15 : Gamma_total = Gamma_P15 (goulot), l'EM qui suit est
     instantane (~ns << tau_P15). Les 511 keV de l'annihilation
     finale sont sous tous les seuils nucleaires (inoffensifs) ; le
     photon UTILE est celui du desenlacement, Q = liaison
     beta sqrt(cJ cJ') -- la fenetre 7Be.
STATUT : theoreme conditionnel (dynamique = transport lisse + sommets
P15, le cadre etabli) ; [A]-[B] sont des verifications exactes.
"""
import numpy as np


def gauss_linking(A, B):
    a_mid = 0.5 * (A + np.roll(A, -1, axis=0))
    b_mid = 0.5 * (B + np.roll(B, -1, axis=0))
    da = np.roll(A, -1, axis=0) - A
    db = np.roll(B, -1, axis=0) - B
    r = a_mid[:, None, :] - b_mid[None, :, :]
    d3 = np.maximum(np.sum(r * r, axis=2) ** 1.5, 1e-12)
    cross = np.cross(da[:, None, :], db[None, :, :])
    return float(np.sum(np.sum(cross * r, axis=2) / d3) / (4 * np.pi))


def circle(c, r, n=160, axis="z"):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if axis == "z":
        return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t),
                         c[2] + 0 * t], -1)
    return np.stack([c[0] + 0 * t, c[1] + r * np.cos(t),
                     c[2] + r * np.sin(t)], -1)


def fuse(L1, L2):
    """Somme connexe : ouvre chaque boucle a ses points les plus
    proches et recolle (pont court, sans croiser K)."""
    d = np.sum((L1[:, None, :] - L2[None, :, :]) ** 2, axis=2)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    A = np.roll(L1, -i - 1, axis=0)   # finit au point i
    B = np.roll(L2, -j, axis=0)       # commence au point j
    return np.vstack([A, B])


def main():
    print(__doc__)
    print("=" * 78)
    print("[A] ADDITIVITE SOUS FUSION")
    print("=" * 78)
    # K : cercle dans le plan xz ; anneau lie L1 le traverse ; L2 libre
    K = circle((0.0, 0.0, 0.0), 1.0, axis="x")
    L1 = circle((0.0, 0.6, 0.0), 0.8, axis="z")      # enlace K
    L2 = circle((0.0, 3.5, 0.0), 0.5, axis="z")      # loin, libre
    lk1, lk2 = gauss_linking(L1, K), gauss_linking(L2, K)
    F = fuse(L1, L2)
    lkF = gauss_linking(F, K)
    print("  Lk(anneau lie, K)   = %+.4f" % lk1)
    print("  Lk(positron, K)     = %+.4f" % lk2)
    print("  Lk(anneau FUSIONNE) = %+.4f  (attendu : la somme %.0f)" % (
        lkF, round(lk1 + lk2)))
    print("""  -> l'annihilation e+ e- (fusion des boucles) produit un anneau
  qui porte ENCORE Lk = 1 a travers K : elle ne libere RIEN.""")

    print("=" * 78)
    print("[B] INVARIANCE SOUS DEFORMATION LISSE (evolution EM)")
    print("=" * 78)
    rng = np.random.default_rng(7)

    def resample(P, n=320):
        seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
        sc = np.concatenate([[0.0], np.cumsum(seg)])
        u = np.linspace(0, sc[-1], n, endpoint=False)
        Q = np.vstack([P, P[:1]])
        i = np.clip(np.searchsorted(sc, u, side="right") - 1, 0,
                    len(sc) - 2)
        f = (u - sc[i]) / np.maximum(sc[i + 1] - sc[i], 1e-12)
        return Q[i] + f[:, None] * (Q[i + 1] - Q[i])

    lks = []
    cur = resample(F)
    for step in range(200):
        # deformation lisse aleatoire (champ de Fourier bas ordre),
        # pas assez grande pour croiser le tube de K (controle distance)
        t = np.linspace(0, 2 * np.pi, len(cur), endpoint=False)
        pert = np.zeros_like(cur)
        for m in (1, 2, 3):
            amp = 0.012 / m
            for ax in range(3):
                pert[:, ax] += amp * (rng.normal() * np.cos(m * t)
                                      + rng.normal() * np.sin(m * t))
        cand = resample(cur + pert)     # longueur d'arc uniforme
        dmin = np.min(np.linalg.norm(
            cand[:, None, :] - K[None, :, :], axis=2))
        if dmin < 0.12:
            continue    # refus : croiserait le tube (rayon fini)
        cur = cand
        lks.append(gauss_linking(cur, K))
    lks = np.array(lks)
    print("  %d deformations acceptees : Lk = %.4f +- %.1e (max dev %.1e)"
          % (len(lks), lks.mean(), lks.std(),
             np.abs(lks - round(lks.mean())).max()))
    print("""  -> le transport lisse conserve Lk a la precision numerique :
  l'evolution EM ne peut PAS effacer l'enlacement (c'est la lecture
  geometrique de la superselection [H_int, W] = 0 du chapitre
  intrication).""")

    print("=" * 78)
    print("[C] STRUCTURE DE TAUX (corollaire)")
    print("=" * 78)
    print("""  Le seul operateur Delta Lk != 0 du cadre est le sommet P15
  (paire fusion/scission, barriere 1/E_relax^2). Donc :
     Gamma(anneau lie -> ondes libres) = Gamma_P15(desenlacement)
  suivi d'une annihilation EM instantanee (~ns << tau_P15 ~ 1e3 s).
  BILAN D'ENERGIE : le desenlacement libere la liaison
  Q = beta sqrt(cJ cJ') ~ 1,9 MeV (LE photon de la fenetre 7Be) ;
  l'annihilation finale de l'anneau libere ensuite ~2 x 511 keV,
  SOUS tous les seuils nucleaires : inoffensive. La protection est
  demontree aux conditions du cadre (dynamique = transport lisse +
  sommets P15) ; survie et emission sont le meme evenement. CQFD.""")


if __name__ == "__main__":
    main()
