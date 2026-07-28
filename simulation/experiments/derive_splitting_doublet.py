"""Le splitting intra-doublet (1/m_Q) : structure derivee, coefficient pince (22/07).

CIBLE : le dernier nombre « lu » du secteur des bandes P -- le splitting
entre les deux membres d'un doublet j_K=3/2 (J=1/J=2). Quatre doublets
observes : D1/D2*, Ds1/Ds2*, B1/B2*, Bs1/Bs2*.

STRUCTURE (a deriver du socle) :
 (a) DEGENERESCENCE a m_Q -> inf : corollaire du pivot S17 -- le repere
     du lourd est inertiel, l'echelle des bandes est AVEUGLE a S_Q
     (c'est pourquoi le doublet EST un doublet : la bande est commune,
     verifie B/Bs universelle n=19/2) ;
 (b) REPONSE A INERTIE FINIE : le ruban lourd repond au couple du tube
     tournant sur son temps de Compton 1/m_Q -- LA MEME identification
     que le moment magnetique canonique (torsion du tube au rayon
     lambda_C, V6 sec:v6-torsion) => Delta_m = kappa/m_Q, i.e.
     Delta_M^2 ~ 2*kappa*(1 + Lbar/m_Q) : quasi constant en M^2 ;
 (c) ORDRE : l'etat aligne (J=2) au-dessus ;
 (d) DILUTION ETRANGE : le couple est porte par la circulation du bout
     leger (son moment ~ 1/m) ; sur le tube COMMUN du meson, la
     substitution l->s ne porte que la MOITIE (demi-substitution
     appliquee au moment) : kappa_s/kappa_l = (1 + m_l/m_s)/2.
"""
import numpy as np

SQRT_SIGMA = 0.41847
SIGMA = SQRT_SIGMA ** 2                 # 0.175117 GeV^2
PI_SIGMA = np.pi * SIGMA                # quantum V-P (demi-tour)
ML, MS = 0.310, 0.4948
MC, MB = 1.557, 4.952                   # masses de noeuds TQNT

# (nom, J=1 (j_K=3/2), J=2, m_Q, etrange?)   [PDG, GeV]
DOUBLETS = [
    ("D1/D2*",   2.4221, 2.4611, MC, False),
    ("Ds1/Ds2*", 2.53511, 2.5691, MC, True),
    ("B1/B2*",   5.7261, 5.7395, MB, False),
    ("Bs1/Bs2*", 5.8287, 5.83986, MB, True),
]
DEMI = (1 + ML / MS) / 2                # 0.8133


def main():
    print(__doc__)
    print("=" * 78)
    print("[1] DONNEES ET LOI 1/m_Q (masses de noeuds TQNT)")
    print("=" * 78)
    print("%-10s %8s %9s %9s %10s %10s %10s" % (
        "doublet", "dm(MeV)", "dM2(GeV2)", "dn(1/2t)", "k=dm*mQ", "k/sigma", "k_red/s."))
    rows = []
    for nom, m1, m2, mq, strange in DOUBLETS:
        dm = m2 - m1
        dm2 = m2**2 - m1**2
        dn = dm2 / PI_SIGMA
        kappa = dm * mq
        kred = kappa / (DEMI if strange else 1.0)
        rows.append((nom, dm, dm2, dn, kappa, kred, strange))
        print("%-10s %8.1f %9.4f %9.3f %10.4f %10.3f %10.3f" % (
            nom, 1e3 * dm, dm2, dn, kappa, kappa / SIGMA, kred / SIGMA))
    kaps = np.array([r[4] for r in rows]) / SIGMA
    kreds = np.array([r[5] for r in rows]) / SIGMA
    print("""
  -> ORDRE : J=2 au-dessus de J=1 : 4/4 (structure (c) OK).
  -> LOI 1/m_Q : en Delta_m, c->b tombe de x2,9 ; en kappa = dm*m_Q :
     %.3f +- %.3f sigma_chi (dispersion %.0f %%) -- la suppression
     Compton (b) est VERIFIEE (le x2,9 devient +-10 %%).
  -> DILUTION ETRANGE : kappa_s/kappa_l = %.3f (c), %.3f (b) --
     moyenne %.3f vs demi-substitution du moment (1+m_l/m_s)/2 = %.3f
     (+4,8 %%) : structure (d) COMPATIBLE. Correction appliquee :
     la dispersion de kappa_red tombe a %.3f +- %.3f (%.0f %% -> %.0f %%).""" % (
        kaps.mean(), kaps.std(), 100 * kaps.std() / kaps.mean(),
        rows[1][4] / rows[0][4], rows[3][4] / rows[2][4],
        (rows[1][4] / rows[0][4] + rows[3][4] / rows[2][4]) / 2, DEMI,
        kreds.mean(), kreds.std(),
        100 * kaps.std() / kaps.mean(), 100 * kreds.std() / kreds.mean()))

    print("=" * 78)
    print("[2] LE COEFFICIENT : kappa_red = (%.3f +- %.3f) sigma_chi" % (
        kreds.mean(), kreds.std()))
    print("=" * 78)
    mean, err = kreds.mean(), kreds.std() / np.sqrt(3)   # 4 dets, 3 dof
    for nom, val in [("3/8", 0.375), ("1/e", np.exp(-1)),
                     ("s = DWr/pi", 0.359), ("1/3", 1 / 3.0),
                     ("d-1 = rac2-1", np.sqrt(2) - 1), ("1/pi", 1 / np.pi),
                     ("2/(3pi) x pi", 2 / 3.0)]:
        ns = abs(val - mean) / err
        print("  %-18s %7.4f   %5.1f sigma %s" % (
            nom, val, ns, "" if ns > 3 else "(compatible)"))
    print("""
  -> PLUSIEURS formes closes dans la bande (3/8 a -1 %%, 1/e, s...),
  AUCUNE avec mecanisme etabli : PAR LA REGLE DU NULL, le coefficient
  RESTE MESURE. Le residu se retrecit de « 4 splittings lus » a
  « UN coefficient d'ordre 1 : kappa_red = (%.2f +- %.2f) sigma_chi ».""" % (
        kreds.mean(), kreds.std()))

    print("=" * 78)
    print("[3] PREDICTIONS (kappa_red universel + structures (a)-(d))")
    print("=" * 78)
    k_l = kreds.mean() * SIGMA
    print("""  - Tout nouveau doublet j_K=3/2 suit Delta_m = kappa/m_Q avec
    kappa = %.4f GeV^2 (x %.3f si bout etrange) ;
  - doublet j_K=1/2 (couplage <S.j> moitie du 3/2) : B0*/B1' attendus
    splittes de ~%.1f MeV (fourchette centroide 5591-5665, S17) --
    RESERVE : le secteur j_K=1/2 charme (D0*/D1', larges, masses
    disputees) montre ~70 MeV : distorsion de canaux couples connue,
    extension declaree FRAGILE ;
  - bandes P des baryons etranges : meme kappa_red, dilution PLEINE
    (jambes propres, pas de tube commun : facteur m_l/m_s au lieu de
    la demi-substitution) -- discriminant avec la prediction du shift
    s (aperture identite).""" % (k_l, DEMI, 1e3 * k_l / (2 * MB)))

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  STRUCTURE DERIVEE (4 pieces) : (a) degenerescence = corollaire du
  pivot S17 (l'echelle aveugle a S_Q -- pourquoi le doublet existe) ;
  (b) 1/m_Q = reponse Compton du ruban lourd (la meme identification
  que le moment magnetique canonique) -- verifie : kappa constant a
  +-10 %% quand Delta_m varie x2,9 ; (c) ordre 4/4 ; (d) dilution
  etrange = demi-substitution du moment (+4,8 %%, resserre la
  dispersion x2). COEFFICIENT : kappa_red = (0,37 +- 0,02) sigma_chi
  -- UN nombre mesure pour 4 splittings ; formes closes non pincees
  (regle du null). Le dernier « lu » du secteur des bandes est reduit
  a un coefficient d'ordre 1 a structure fermee.""")


if __name__ == "__main__":
    main()
