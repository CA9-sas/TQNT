"""(m3) Le croisement-ouvert dans les oscillations K0/B0 -- verification.

Suite de derive_structure_vide_barreau8.py (2026-07-21). Question : les
oscillations particule-antiparticule portent-elles le meme objet (le
croisement-ouvert sur l'ancre, E_relax = 246,08 GeV) ?

FAIT D'ENTREE (chaine canonique V5p27-p33) : la chaine du couple EM mutuel
est calibree sur K0 (UN prefacteur global, f(K0)=0 par construction,
theory/v5p33_mutual_EM_oscillation.py) -- elle predit les RATIOS
(D0/K0, B0/K0, Bs/B a 0,1 %), PAS l'echelle absolue de suppression
(Delta m_K / m_K ~ 7e-15). L'echelle absolue est exactement l'endroit ou la
barriere peut vivre : le test est SANS conflit avec la chaine canonique.

STRUCTURE (derivee du theoreme fusion/scission) :
  - une desintegration faible = UN dLk = UNE paire fusion/scission
    -> amplitude 1/E_relax^2 (= G_F) ;
  - une oscillation K0 <-> K0bar = bascule miroir a DEUX unites (le
    SM l'indexe << dS=2 >>) = DEUX paires fusion/scission
    -> amplitude 1/E_relax^4 (= G_F^2, la << boite >> du SM).
  LOI DE COMPTAGE : n paires <-> 1/E_relax^(2n). Chaque paire traversant
  un pas de generation paie lambda_Cab (cout de distance de reconnexion,
  canonique) -- le (V_cd V_cs)^2 ~ lambda^2 de la boite = 2 paires x 1 pas.

INGREDIENTS (declares) :
  TQNT-natifs : E_relax = croisement-ouvert 246,084 ; lambda_Cab = 0,224 ;
    masses d'entrelacs W = 80,68 / top = 170,41 ; option m_c noeud = 1557.
  Importes (echafaudage, declare) : B_K, f_K, f_Bd, S0(x), eta_QCD --
    la structure de recouvrement mesonique n'est pas encore TQNT-derivee.
"""
import math

E_RELAX = 246.084          # GeV -- croisement-ouvert sur l'ancre
V_MEAS = 246.21965
LAM_CAB = 0.224            # cout de reconnexion par pas (canonique)
GF_PRED = 1.0 / (math.sqrt(2) * E_RELAX**2)
GF_MEAS = 1.1663787e-5

# mesures PDG
DM_K_MEV = 3.484e-12
DM_BD_MEV = 3.334e-10
M_K, M_BD = 0.497611, 5.27966          # GeV
F_K, B_K = 0.1557, 0.76                # GeV ; parametre de sac (importes)
FBD_SQRT_B = 0.225                     # GeV, f_Bd*sqrt(B_Bd) (lattice, importe)
ETA_B = 0.55
M_C_SM, M_C_KNOT = 1.27, 1.557         # GeV
M_W_TQNT, M_TOP_TQNT = 80.68, 170.41   # GeV (entrelacs canoniques)
V_TD_MEAS = 0.0086


def s0_inami_lim(x):
    """Fonction de boite d'Inami-Lim S0(x) (echafaudage importe).
    S0(x) = x(4 - 11x + x^2)/(4(1-x)^2) - 3x^3 ln(x) / (2(1-x)^3)."""
    return x * (4 - 11 * x + x**2) / (4 * (1 - x)**2) \
        - 1.5 * x**3 * math.log(x) / (1 - x)**3


def kaon_box(m_int):
    """Delta m_K de la double paire fusion/scission (structure de boite)."""
    lam2 = (LAM_CAB * math.sqrt(1 - LAM_CAB**2))**2   # 2 paires x 1 pas
    dm_gev = (GF_PRED**2 / (6 * math.pi**2)) * B_K * F_K**2 * M_K \
        * m_int**2 * lam2
    return dm_gev * 1000.0 * 2  # x2 : Delta m = 2|M12| ; en MeV... (voir run)


def run():
    print(__doc__)
    print("=" * 72)
    print("[1] L'OBJET LUI-MEME : G_F")
    print("=" * 72)
    print("  G_F(croisement-ouvert) = %.5e GeV^-2" % GF_PRED)
    print("  G_F(mesure)            = %.5e GeV^-2  (ecart %+.2f %%)"
          % (GF_MEAS, 100 * (GF_PRED / GF_MEAS - 1)))

    print("\n" + "=" * 72)
    print("[2] K0 : deux paires fusion/scission -> 1/E^4 (echelle absolue)")
    print("=" * 72)
    lam2 = (LAM_CAB * math.sqrt(1 - LAM_CAB**2))**2
    for name, m_int in [("m_c SM 1,27", M_C_SM), ("m_c noeud 1557", M_C_KNOT)]:
        dm_gev = (GF_PRED**2 / (6 * math.pi**2)) * B_K * F_K**2 * M_K \
            * m_int**2 * lam2
        dm_mev = dm_gev * 1000.0
        print("  [%s]  Delta m_K = %.2e MeV  (mesure %.2e ; ratio %.2f)"
              % (name, dm_mev, DM_K_MEV, dm_mev / DM_K_MEV))
    print("  -> ORDRE reproduit (x0,5-0,7 ; la part longue distance du K,")
    print("     connue ~+50 %, n'est pas dans la boite courte). L'echelle")
    print("     absolue que V5p33 CALIBRE est bien celle du double")
    print("     croisement-ouvert.")

    print("\n" + "=" * 72)
    print("[3] B0_d : boite dominee par le top-entrelacs")
    print("=" * 72)
    x_t = (M_TOP_TQNT / M_W_TQNT)**2
    s0 = s0_inami_lim(x_t)
    for vlabel, vtd2 in [("|V_td| mesure", V_TD_MEAS**2),
                         ("lambda_Cab^3 (3 pas)", LAM_CAB**6)]:
        dm_gev = (GF_PRED**2 / (6 * math.pi**2)) * M_W_TQNT**2 * s0 * ETA_B \
            * FBD_SQRT_B**2 * M_BD * vtd2
        dm_mev = dm_gev * 1000.0
        print("  [%s]  Delta m_B = %.2e MeV  (mesure %.2e ; ratio %.2f)"
              % (vlabel, dm_mev, DM_BD_MEV, dm_mev / DM_BD_MEV))
    print("  (x_t entrelacs = %.2f, S0 = %.2f)" % (x_t, s0))
    print("  -> ORDRE reproduit aussi au B (top et W = entrelacs TQNT).")

    print("\n" + "=" * 72)
    print("[4] SENSIBILITE : la decimale (m3) est-elle disponible ici ?")
    print("=" * 72)
    dv = (V_MEAS - E_RELAX) / V_MEAS
    print("  Discriminer 246,084 vs 246,220 (%.3f %%) via Delta m ~ E^-4"
          % (100 * dv))
    print("  exige une theorie de Delta m a %.2f %% -- la dispersion" % (400 * dv))
    print("  theorique (boite + longue distance + B_K) est ~50 % : NON.")

    print("\n" + "=" * 72)
    print("VERDICT (m3)")
    print("=" * 72)
    print("  CONFIRME (structure + ordre) : les oscillations portent le MEME")
    print("  objet -- deux paires fusion/scission, loi n <-> 1/E^(2n) ;")
    print("  l'echelle absolue calibree par V5p33 (prefacteur K0) recoit une")
    print("  IDENTITE candidate : le double croisement-ouvert. Complement,")
    print("  pas conflit : couple EM = structure interne/ratios ; croisement-")
    print("  ouvert = echelle absolue de bascule.")
    print("  NON OBTENU : la decimale independante (sensibilite E^-4 vs")
    print("  theorie ~50 %). (m3) devra venir d'une observable en E^-2 a")
    print("  theorie fine -- le candidat reste FORT non consolide, v au")
    print("  comptage (10+2).")


if __name__ == "__main__":
    run()
