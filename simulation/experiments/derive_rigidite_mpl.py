"""Deriver la rigidite elle-meme (pourquoi M_Pl) ? -- trois mecanismes au banc (19/07).

CHANTIER (brouillon, PAS V6). La rigidite metrique du vide chi
(S_vac = c^4/8piG, i.e. M_Pl) est la primitive de la lecture inversee du
raccord. Peut-on la DERIVER de l'interieur de la TQNT ?

CADRE STRUCTUREL (le pendant exact du verdict m_l, derive_ml_tube_profile) :
la couche fondamentale de la TQNT est SANS ECHELLE -- l'algebre CS
(k = 135) est topologique, le potentiel microscopique V = -Lambda rho/d^2
est invariant d'echelle. Toute quantite dimensionnee est donc une donnee
de TRANSMUTATION : il en faut exactement UNE (l'ancre) ; le contenu
derivable de la theorie = les RAPPORTS sans dimension.

TROIS MECANISMES CANDIDATS pour deriver M_Pl quand meme :
 (a) gravite INDUITE (Sakharov) : 1/G ~ N_dof Lambda_UV^2/(12 pi) --
     les fluctuations du champ chi jusqu'a sa coupure UV induisent la
     rigidite. Testable : N_dof et Lambda_UV sont fixes par la TQNT.
 (b) ENTROPIQUE (Jacobson) : 1/4G = entropie d'intrication par aire du
     vide chi. Testable : l'entropie CS est TOPOLOGIQUE (-ln D,
     sous-extensive, verify_anyonic_counting) -- y a-t-il un terme d'aire ?
 (c) COINCIDENCE DU TUBE : M_Pl =? (hbar c/4 pi D) alpha^-9 (D = diametre
     du tube, 0,0214 fm). A chiffrer ET a auto-annuler (equivalences).
"""
import math

ALPHA = 7.2973525693e-3
K_CS = 135
HBARC_MEVFM = 197.3269804
M_PL_MEV = 1.220890e22        # sqrt(hbar c/G), CODATA
M_BASE = 938.240
M_L = 310.0                   # m_l (trefle, calibree)
SQRT_SIGMA = 418.47
S_OCTET = 2.242006
V_EW = 246.22e3               # MeV (identifiee, non derivee)
D_TUBE_FM = 0.0214            # diametre du tube (mesure, loi de masse)
LAM_C_FM = HBARC_MEVFM / M_BASE


def run():
    print("=== Deriver la rigidite (pourquoi M_Pl) : trois bancs d'essai ===\n")
    a9 = ALPHA ** 9

    # (a) gravite induite (Sakharov)
    print("(a) Gravite induite (Sakharov) : M_Pl^2 ~ N_dof Lambda_UV^2/(12 pi)")
    n_dof = K_CS + 1              # especes primaires SU(2)_k : k+1
    lam_uv = HBARC_MEVFM / D_TUBE_FM   # coupure du tube, MeV
    mpl_pred = math.sqrt(n_dof / (12 * math.pi)) * lam_uv
    print("  N_dof = k+1 = %d ; Lambda_UV = hbar c/D = %.2f GeV" % (
        n_dof, lam_uv / 1000))
    print("  M_Pl predit = %.1f GeV  vs  %.3e GeV" % (
        mpl_pred / 1000, M_PL_MEV / 1000))
    print("  -> DEFICIT %.1f ORDRES. Pour fermer il faudrait Lambda_UV ~" % (
        math.log10(M_PL_MEV / mpl_pred)))
    print("     %.1e GeV -- AUCUNE echelle TQNT ne la fournit. MORT.\n" % (
        M_PL_MEV / math.sqrt(n_dof / (12 * math.pi)) / 1000))

    # (b) entropique (Jacobson)
    print("(b) Entropique (Jacobson) : 1/4G = entropie par aire du vide chi")
    D_tot = math.sqrt((K_CS + 2) / 2) / math.sin(math.pi / (K_CS + 2))
    print("  entropie CS du vide : TOPOLOGIQUE, -ln D = -%.2f par region," % math.log(D_tot))
    print("  SOUS-EXTENSIVE (verify_anyonic_counting) : PAS de terme d'aire")
    print("  intrinseque -- un terme d'aire n'apparait qu'avec une coupure")
    print("  UV epsilon (S ~ c_WZW A/epsilon^2), et epsilon est... une echelle")
    print("  de transmutation. Le mecanisme REFORMULE la question (M_Pl <->")
    print("  epsilon), il ne la resout pas. MORT comme derivation.\n")

    # (c) coincidence du tube
    print("(c) Coincidence du tube : M_Pl =? (hbar c/(4 pi D)) alpha^-9")
    mpl_tube = (HBARC_MEVFM / (4 * math.pi * D_TUBE_FM)) / a9
    ratio = M_PL_MEV / mpl_tube
    print("  (hbar c/4piD) alpha^-9 = %.4e MeV ; M_Pl/pred = %.3f (%+.1f %%)"
          % (mpl_tube, ratio, 100 * (ratio - 1)))
    d_equiv = math.sqrt(12.0 / 7.0) * LAM_C_FM / (4 * math.pi)
    print("  AUTO-ANNULATION : via l'identite de hierarchie, cette forme est")
    print("  EQUIVALENTE a D = sqrt(12/7) lambdabar_C/(4 pi) = %.5f fm" % d_equiv)
    print("  (mesure %.4f fm, %+.1f %%) -- une relation tube<->Compton du" % (
        D_TUBE_FM, 100 * (d_equiv / D_TUBE_FM - 1)))
    print("  nucleon, PAS une derivation de M_Pl. Meme classe que la")
    print("  cellule de phase N = 138,6 (deja defavorisee) : COINCIDENCE.\n")

    # theoreme de fermeture des echelles : etat des rapports
    print("THEOREME DE FERMETURE (pendant du verdict m_l) :")
    print("  couche fondamentale sans echelle (CS topologique + V ~ 1/d^2")
    print("  invariant) -> AUCUNE echelle n'est derivable de l'interieur ;")
    print("  il faut exactement UNE ancre de transmutation. La seule qui")
    print("  preexiste a T_c est la rigidite (M_Pl). Contenu derivable =")
    print("  les RAPPORTS. Etat des lieux :\n")
    rows = [
        ("m_base/M_Pl", M_BASE / M_PL_MEV, "sqrt(12/7) alpha^18 : DERIVE (-0,025 %)"),
        ("sqrt(sigma)/M_Pl", SQRT_SIGMA / M_PL_MEV, "= (m_base/M_Pl)/S_octet : DERIVE"),
        ("m_l/M_Pl", M_L / M_PL_MEV, "0,330 x m_base/M_Pl : NON derive (calibre)"),
        ("v/M_Pl", V_EW / M_PL_MEV, "mur de la valeur de v : NON derive"),
    ]
    for lab, val, statut in rows:
        print("  %-18s = %.3e   %s" % (lab, val, statut))
    print("\n  m_l/m_base = %.4f (aucune forme close etablie ; le null des" % (M_L / M_BASE))
    print("  formes scan_ml_sigma_forms s'applique).")

    print("\nVERDICT : la rigidite N'EST PAS derivable de l'interieur --")
    print("elle est LA constante de transmutation de la TQNT (l'unique")
    print("echelle, pendant exact de Lambda_QCD et du verdict m_l). Le")
    print("programme correct n'est pas << deriver M_Pl >> mais << reduire")
    print("toute echelle a un rapport derive x M_Pl >> : fait pour m_base")
    print("et sigma_chi (lecture inversee), ouvert pour m_l et v.")
    print("Consequence de comptage : ancre M_Pl <-> sqrt(sigma) devient")
    print("derivee -- echange de convention, neutre en nombre (une ancre")
    print("reste une ancre), mais l'ancre devient une constante mesuree")
    print("independamment (G) : le theoreme G est la RELATION d'echange.")


if __name__ == "__main__":
    run()
