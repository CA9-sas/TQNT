"""Le profil du tube comme régularisation — K_33 sans coupure
(27/07/2026, la cible isolée de la Partie VIII).

PRINCIPE : la coupure dure était arbitraire (a=r vs a=D inversait le
signe). La régularisation PHYSIQUE est le profil transverse du
courant : on calcule K_33 avec l'étalement transverse EXACT — le
noyau 1/|r-r'| est moyenné sur les sections des deux éléments de
courant. AUCUNE coupure : le profil EST la régularisation.

LE PROFIL EST DÉSIGNÉ PAR LE CADRE : le courant du sens est la
circulation de FRAMING (l'anneau conducteur du réservoir sombre : la
circulation persistante vit sur le bord du ruban) => courant de
SURFACE (coquille au rayon a du cœur). Le profil volumique (disque
uniforme) est calculé comme variante de contrôle.

ORDRE SUIVANT NOMMÉ (pas encore inclus) : les lignes de courant de
surface suivent le framing — White : Tw = Lk - Wr ; pour le lepton,
Q = Sigma chi + Lk = -1 avec chi = -2 => Lk = +1, d'où
Tw = 1 - 3,4165 = -2,42 tours sur le circuit : inclinaison
hélicoïdale theta = atan(2 pi a |Tw| / L) ~ 25 degrés (facteur
longitudinal cos²theta ~ 0,82, termes azimutaux non estimés).

MISE EN ŒUVRE : trèfle idéal (SONO), repère de transport parallèle
(la moyenne azimutale est indépendante du repère — l'holonomie de
Fuller ne gêne pas), sections échantillonnées (coquille m=12 ;
disque 4 rayons x 8) ; tangentes de la ligne centrale (ordre
dominant ; corrections de Frenet O(a kappa) déclarées).

GARDE-FOUS : discrétisation déclarée (ds = a/7) ; l'inclinaison de
framing et les tangentes déplacées = ordre suivant ; règle du test
nul.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from simulation.ideal_embedding import sono, resample
from theory.omega_ts import _knot_curve

M_E = 0.51099895
ALPHA = 1 / 137.035999
HBARC = 197.32698
MU = 17.715562
M_L = MU ** 2
SQRT_SIGMA = 418.47
WR_IDEAL = 3.4165


def frames(P):
    """repère de transport parallèle (Bishop) le long de la courbe."""
    dl = np.roll(P, -1, 0) - P
    ds = np.linalg.norm(dl, axis=1)
    t = dl / ds[:, None]
    n = len(P)
    u = np.zeros((n, 3))
    v = np.zeros((n, 3))
    a0 = np.array([1.0, 0.0, 0.0])
    if abs(a0 @ t[0]) > 0.9:
        a0 = np.array([0.0, 1.0, 0.0])
    u[0] = a0 - (a0 @ t[0]) * t[0]
    u[0] /= np.linalg.norm(u[0])
    v[0] = np.cross(t[0], u[0])
    for i in range(1, n):
        w = u[i - 1] - (u[i - 1] @ t[i]) * t[i]
        u[i] = w / np.linalg.norm(w)
        v[i] = np.cross(t[i], u[i])
    return t, u, v, ds


def k33_smeared(P, a, sections):
    """K_33/L et N/L avec moyenne transverse exacte sur les sections.

    sections : liste de (rho_frac, poids) x azimuts m — points (rho, phi).
    """
    P = np.asarray(P, float)
    P = P - P.mean(0)
    t, u, v, ds = frames(P)
    n = len(P)
    s = np.concatenate([[0.0], np.cumsum(ds)])[:-1]
    L = ds.sum()
    sig = 2 * np.pi * s / L
    # points de section : X[i, q, :]
    pts = []
    wts = []
    for rho, w_r, m in sections:
        for k in range(m):
            phi = 2 * np.pi * k / m
            pts.append((rho * np.cos(phi), rho * np.sin(phi)))
            wts.append(w_r / m)
    pts = np.array(pts)
    wts = np.array(wts)
    wts = wts / wts.sum()
    q = len(pts)
    X = (P[:, None, :] + a * pts[:, 0][None, :, None] * u[:, None, :]
         + a * pts[:, 1][None, :, None] * v[:, None, :])   # (n, q, 3)
    W2 = wts[:, None] * wts[None, :]                        # (q, q)
    K33 = 0.0
    N = 0.0
    for i in range(n):
        diff = X[i][:, None, None, :] - X[None, :, :, :]     # (q, n, q, 3)
        d = np.linalg.norm(diff, axis=3)                     # (q, n, q)
        d[d < 1e-12] = np.inf                                # coïncidences
        g = (W2[:, None, :] / d).sum(axis=(0, 2))            # (n,)
        tt = t[i] @ t.T
        c3 = np.cos(3 * (sig[i] - sig))
        w_arc = ds[i] * ds
        K33 += (tt * c3 * w_arc * g).sum()
        N += (tt * w_arc * g).sum()
    return K33 / L, N / L, L


def main():
    print(__doc__)
    P0 = np.asarray(_knot_curve("3_1", 200), float)
    P_id, info = sono(P0, iters=20000)
    ell = info["ropelength"]
    P = resample(P_id, 240)
    dl = np.roll(P, -1, 0) - P
    L_geo = np.linalg.norm(dl, axis=1).sum()
    a = L_geo / ell                                   # rayon de tube
    L_fm = M_L / SQRT_SIGMA ** 2 * HBARC
    pref = ALPHA * HBARC / (2 * L_fm)
    print("=" * 78)
    print("trèfle idéal : ropelength = %.3f ; ds/a = %.2f ; préfacteur"
          " = %.4f MeV" % (ell, (L_geo / 240) / a, pref))

    print("=" * 78)
    print("K_33 AVEC LE PROFIL (aucune coupure)")
    print("=" * 78)
    shell = [(1.0, 1.0, 12)]
    disk = [(np.sqrt(0.125), 0.25, 8), (np.sqrt(0.375), 0.25, 8),
            (np.sqrt(0.625), 0.25, 8), (np.sqrt(0.875), 0.25, 8)]
    for lab, sec in [("COQUILLE (framing, canonique)", shell),
                     ("disque uniforme (contrôle)", disk)]:
        K33, NL, _ = k33_smeared(P, a, sec)
        me = 0.5 * pref * K33
        print("  %-30s : K_33/L = %+.4f ; N/L = %.3f" % (lab, K33, NL))
        print("  %-30s   m_e-candidat = %.3f MeV  (%+.1f %%)" % (
            "", me, 100 * (me / M_E - 1)))
    print("""
  RAPPEL des coupures dures : a=r -> +0,449 (-10,5 %) ; a=D -> -0,237
  (signe inversé). Le profil remplace ce choix : le nombre ci-dessus
  est SANS paramètre de régularisation.""")

    print("=" * 78)
    print("LE CALCUL COMPLET : COURANT DE FRAMING HÉLICOÏDAL (White)")
    print("=" * 78)
    tw = 1 - WR_IDEAL
    theta = np.arctan(2 * np.pi * a * abs(tw) / L_geo)
    print("  Tw = Lk - Wr = %.3f tours ; inclinaison ~%.1f deg" % (
        tw, np.degrees(theta)))
    # filaments de framing : fermés PARCE QUE Lk est entier (White)
    t, u, v, ds = frames(P)
    n = len(P)
    s_arc = np.concatenate([[0.0], np.cumsum(ds)])[:-1]
    L = ds.sum()
    sig = 2 * np.pi * s_arc / L
    # angle de raccord du repère de Bishop (holonomie de Fuller)
    w_tr = u[-1] - (u[-1] @ t[0]) * t[0]
    w_tr /= np.linalg.norm(w_tr)
    delta_seam = np.arctan2(w_tr @ v[0], w_tr @ u[0])
    LK = 1                                   # dérivé de Q = Sigma chi + Lk
    m = 12
    from simulation.ideal_embedding import linking_number
    X = None
    for branche in (0, -1, 1, -2, 2):        # ambiguïté mod 2pi du raccord
        rot = 2 * np.pi * LK - (delta_seam + 2 * np.pi * branche)
        ph = (2 * np.pi * np.arange(m)[None, :] / m
              + rot * (s_arc / L)[:, None])
        Xc = (P[:, None, :] + a * np.cos(ph)[:, :, None] * u[:, None, :]
              + a * np.sin(ph)[:, :, None] * v[:, None, :])
        lk_check = linking_number(Xc[:, 0, :], P)
        if lk_check == LK:
            X = Xc
            break
    print("  contrôle d'enlacement : Lk(filament, cœur) = %s (attendu %d ;"
          " branche %+d)" % (lk_check, LK, branche))
    assert X is not None, "aucune branche ne réalise Lk=1"
    dl = np.roll(X, -1, axis=0) - X                            # (n, m, 3)
    Mid = X + 0.5 * dl
    K33h = 0.0
    Nh = 0.0
    for i in range(n):
        diff = Mid[i][:, None, None, :] - Mid[None, :, :, :]   # (m, n, m, 3)
        d = np.linalg.norm(diff, axis=3)
        d[d < 1e-12] = np.inf
        dot = np.einsum('kc,jlc->kjl', dl[i], dl)              # (m, n, m)
        g = (dot / d).sum(axis=(0, 2)) / m ** 2                # (n,)
        c3 = np.cos(3 * (sig[i] - sig))
        K33h += (c3 * g).sum()
        Nh += g.sum()
    K33h /= L
    Nh /= L
    me_h = 0.5 * pref * K33h
    print("  K_33/L (hélicoïdal complet) = %+.4f ; N/L = %.3f" % (K33h, Nh))
    print("  m_e-candidat COMPLET = %.3f MeV  (vs m_e = %.3f ; %+.1f %%)" % (
        me_h, M_E, 100 * (me_h / M_E - 1)))
    print("  CONTRÔLES CROISÉS :")
    print("    E_complet vs alpha·m_l = %.4f MeV : %+.2f %%  (la fraction"
          " EM alpha" % (ALPHA * M_L, 100 * (me_h / (ALPHA * M_L) - 1)))
    print("      du tube — chantier k=135 — RETROUVÉE par la géométrie ;")
    print("      via m_l·L = (3/4)² exact, équivaut à K_33 = 9/4 = 2,25 ;")
    print("      mesuré %.4f (%+.2f %%) — coïncidence géométrique candidate,"
          % (K33h, 100 * (K33h / 2.25 - 1)))
    print("      SANS mécanisme : non retenue (règle du test nul).")
    print("    m_e / E_complet = %.4f  vs delta = 0,2222 : %+.1f %%" % (
        M_E / me_h, 100 * (M_E / me_h / (2.0 / 9.0) - 1)))
    print("      — la coïncidence delta·alpha·m_l revient par la voie")
    print("      géométrique ; toujours sans mécanisme, toujours non")
    print("      retenue ; la cible devient : LA PARTICIPATION DU MODE.")
    print("""
  Ce calcul contient TOUT : positions déplacées, tangentes réelles
  des lignes hélicoïdales (inclinaison de framing incluse), couplages
  azimutaux — sans paramètre de régularisation ni d'amplitude
  (Lk = 1 dérivé de la charge, Tw par White, poids égaux).""")

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""   Le profil physique (coquille = courant de framing) REMPLACE la
   coupure ; le calcul hélicoïdal complet (White, Lk = 1 vérifié par
   enlacement) donne un nombre SANS AUCUN paramètre : E = 2,28 MeV.
   DEUX FAITS : (i) E = alpha·m_l à -0,3 % — le calcul géométrique
   RETROUVE la fraction EM alpha du tube (contrôle croisé non
   trivial avec le chantier k=135) ; (ii) m_e = 0,224·E — le mode
   n=3 à participation PLEINE porte toute la fraction EM, m_e exige
   une participation ~0,22 (numériquement ~delta, +0,8 % — consigné,
   sans mécanisme, non retenu). Le -10,5 % de la coupure dure était
   un artefact. CIBLE FINALE UNIQUE : la participation du mode
   (pourquoi l'électron nodal ne porte-t-il qu'un cinquième de la
   circulation ?). Rien au-delà n'est revendiqué.""")


if __name__ == "__main__":
    main()
