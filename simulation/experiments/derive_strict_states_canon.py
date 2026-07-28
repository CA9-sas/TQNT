"""Rederivation des masses des 6 etats sombres STRICTS sous le document principal (15/07).

CONTEXTE : les 6 entrelacs amphichiraux purs (Q=0 strict, population stable
secondaire de la DM) portaient des masses de l'ere Omega_ts (1.55-51.8 GeV),
"sans loi de secteur canonique" (CLAUDE.md). Les raies F3/CTA en heritent.

LOI CANONIQUE APPLIQUEE : la masse statique par noeud M_topo = sqrt(sigma_chi)
* l(K) = (m_l / l(3_1)) * l(K) (loi geometrique consolidee par m_s derive),
valide pour le plateau leger et le reservoir sombre. Pour un entrelacs sombre :
  M(L) = somme M_topo(K_i) + E_enlacement
avec E_enlacement ~ beta_otimes * (Lk-Lk_opt)^2 * <sqrt(cJ cJ)> ~ quelques MeV
(negligeable a l'echelle du GeV). PAS d'habillage EM (etats sombres), PAS
d'ecrantage de confinement (pas de secteur fort) : la somme geometrique est
l'estimation canonique naturelle -- ESTIMATION, pas loi de secteur etablie
(statut declare).

QUESTIONS : ou tombent les 6 etats ? que devient la prediction CTA 16-52 GeV ?
"""

RL = {"3_1": 16.372, "4_1": 20.99, "6_3": 27.21, "8_3": 37.30}
SCALE = 310.0 / RL["3_1"]        # MeV par unite (m_l = 310 MeV calibre)
STRICTS = [
    ("6_3+6_3",       ["6_3", "6_3"],        1.55),
    ("6_3+6_3+6_3",   ["6_3", "6_3", "6_3"], 4.70),
    ("4_1+8_3",       ["4_1", "8_3"],        16.01),
    ("8_3+8_3",       ["8_3", "8_3"],        17.24),
    ("6_3+8_3",       ["6_3", "8_3"],        26.48),
    ("4_1+4_1+8_3",   ["4_1", "4_1", "8_3"], 51.78),
]
# extensions EW-neutres auto-conjuguees (memes contenus + quartet 8_3 double)
EW_NEUTRES = [
    ("4_1+8_3+8_3",   ["4_1", "8_3", "8_3"], 16.0),
    ("6_3+8_3+8_3",   ["6_3", "8_3", "8_3"], 26.5),
]


def m_topo(k):
    return SCALE * RL[k] / 1000.0    # GeV


def run():
    print("=== Etats stricts : masses canoniques (geometrique) vs Omega_ts ===\n")
    print("M_topo par noeud : " + " ; ".join(
        "%s = %.0f MeV" % (k, 1000 * m_topo(k)) for k in ("4_1", "6_3", "8_3")))
    print("(E_enlacement ~ beta_otimes : quelques MeV, negligeable)\n")

    print("%-14s %10s %12s %8s  %s" % ("etat", "document principal(GeV)", "Omega_ts", "ratio", "contient 8_3"))
    document principal = []
    for nom, ks, mold in STRICTS:
        m = sum(m_topo(k) for k in ks)
        document principal.append((nom, m, "8_3" in ks))
        print("%-14s %10.3f %12.2f %8.3f  %s" % (
            nom, m, mold, m / mold, "oui" if "8_3" in ks else "non (raie fine)"))
    print()
    for nom, ks, mold in EW_NEUTRES:
        m = sum(m_topo(k) for k in ks)
        print("%-14s %10.3f %12.2f   (EW-neutre, extension du quartet)" % (nom, m, mold))

    print("\nVERDICT :")
    print(" - EFFONDREMENT du spectre strict : 1.03-1.55 GeV (vs 1.55-51.8 Omega_ts,")
    print("   facteurs 0.66 a 0.03) -- quasi-degenere avec le bord du reservoir")
    print("   (c_min=23 : 1.71 GeV). Les etats a 16-52 GeV N'EXISTENT PLUS aux regles etablies.")
    print(" - CTA 16-52 GeV : la prediction (raies puis continuum structure) est")
    print("   VIDEE -- les etats porteurs sont a 1.1-1.5 GeV, hors bande CTA basse")
    print("   (seuil ~20 GeV). Toute la spectroscopie Class E bascule dans la bande")
    print("   Fermi-LAT (1.0-1.6 GeV).")
    print(" - Raies fines survivantes (sans 8_3, pas d'etalement de writhe) :")
    print("   6_3+6_3 a 1.03 GeV et 6_3+6_3+6_3 a 1.55 GeV. Les 4 etats a 8_3")
    print("   (1.10-1.50 GeV) restent etales par le doublet de writhe (Wr=+-7.2).")
    print(" - La confrontation F3 (line search Fermi 2015) doit etre REFAITE a ces")
    print("   energies : les Delta_c_crit calcules aux energies Omega_ts sont caducs.")
    print(" - STATUT : estimation canonique (somme geometrique + enlacement")
    print("   negligeable ; pas d'habillage EM ni d'ecrantage pour des etats")
    print("   sombres) -- remplace les valeurs Omega_ts (caduques), mais la loi de")
    print("   secteur 'entrelacs sombre' n'est pas etablie independamment.")


if __name__ == "__main__":
    run()
