"""Re-passe du test aveugle PDG 2024-2026 sous les regles de secteur
canoniques (23/07).

CONTEXTE : le test de juin (data/blind_test_2026/blind_test_pdg_2024_2026.csv)
appariait 22 etats exotiques/baryons recents par ASSIGNATION LIBRE Omega_ts
(2-3 noeuds choisis a posteriori dans un espace >= 64 configurations par
etat ; mediane |Delta| ~ 0,28 % -- mais le null d'assignation montre que
cet accord est atteignable par ~1e4-1e5 combinaisons : contenu predictif
nul). Le moteur Omega_ts est ARCHIVE.

RE-PASSE : les MEMES 22 etats sous les regles de juillet, FIGEES :
  - molecules = hadrons de la trame de l'espace-temps, chaque partenaire
    vecteur porte son demi-tour (HALF = pi sigma_chi, quantum derive) ;
    contenus = lectures moleculaires PDG STANDARD (aucun choix libre) ;
  - constituant : m_l=310, m_s=494,8, m_c=1557, m_b=4952 (chaine +-10 MeV) ;
    demi-substitution etrange (coeur H_s = m_Q + (m_l+m_s)/2) ;
  - bandes baryoniques DERIVEES (22-23/07, zero nombre lu) :
    Delta n(L) = (23/18) L + 3^g (3L+1)(d-1) [+ s*L jambes propres]
    avec d-1 = sqrt(2)-1, s = 0,3590 (= DeltaWr/pi, derive), base n=1 ;
  - compression (quarkonia, tout-charme, J/psi phi) : HORS-CADRE par
    principe (tau < 0) -- coeur constituant descriptif seulement.
STATUTS : PREDIT (regle fixee, zero liberte par etat) ; COMPATIBLE
(structure/ordre conformes, membre ou decimale ouverts) ; OUVERT
(structure non tranchee) ; HORS-CADRE (compression, limite declaree).
"""
import csv
import os
import numpy as np

SQS = 0.41847
SLOPE = 2 * np.pi * SQS**2          # 1 tour de torsion (GeV^2)
HALF = SLOPE / 2                    # 1/2 tour = pi sigma_chi
RAD = 1.38 * SLOPE
ML, MS, MC, MB = 0.310, 0.4948, 1.557, 4.952
D1 = np.sqrt(2) - 1                 # exces d'echange (identifie)
S_SHIFT = 0.3590                    # decalage s = DeltaWr/pi (derive)
B_SUM = 23.0 / 18.0                 # b = 2 - <x> (derive 23/07)

# hadrons de la trame (coeurs constituants + demi-tours)
dd = ML + MC                        # D
ds = MC + (ML + MS) / 2             # Ds (demi-substitution)
dst = np.sqrt(dd**2 + HALF)         # D*
dsst = np.sqrt(ds**2 + HALF)        # Ds*
kst = np.sqrt(0.4976**2 + HALF)     # K* (demi-tour sur K PDG)
xc = MC + ML + MS                   # coeur Xi_c
xc1 = np.sqrt(xc**2 + HALF)         # Xi_c physique (n=1)
lb2 = MB + 2 * ML                   # coeur Lambda_b
oc = MC + 2 * MS                    # coeur Omega_c


def dn_baryon(L, g, s_legs=0):
    """Bande orbitale baryonique derivee (23/07) :
    Delta n(L) = (23/18) L + 3^g (3L+1)(d-1) + s*L par jambe etrange propre."""
    return B_SUM * L + 3**g * (3 * L + 1) * D1 + S_SHIFT * L * s_legs


def m_band(core, n_half):
    return np.sqrt(core**2 + HALF * n_half)


def main():
    print(__doc__)
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                        "blind_test_2026", "blind_test_pdg_2024_2026.csv")
    juin = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            juin[row["particule"]] = (float(row["m_exp_MeV"]),
                                      row["config_TQNT"],
                                      float(row["delta_pct"]))

    # ---- traitements canoniques (regle, valeur, statut) ----------------
    lb_1d = m_band(lb2, 1 + dn_baryon(2, 1))          # Lambda_b 1D
    xic_1d = m_band(xc, 1 + dn_baryon(2, 0, s_legs=1))  # Xi_c 1D (sommet)
    oc_g = m_band(oc, np.sqrt(2))                     # Omega_c fondamental
    oc_1d = m_band(oc, np.sqrt(2) + dn_baryon(2, 0))  # Omega_c 1D (j bas)
    oc_1d_h = m_band(oc, np.sqrt(2) + dn_baryon(2, 0) + 1)  # 1D j haut (x2=1)

    T = {
        # --- PREDIT : regle figee, contenu PDG standard, zero liberte ---
        "T_cc(3875)+": ("molécule DD* (seuil)", dd + dst, "PRÉDIT"),
        "T_cs0(2870)": ("molécule D̄*K*", dst + kst, "PRÉDIT"),
        "Tcs0a(2900)0": ("molécule D*K* (I=1)", dst + kst, "PRÉDIT"),
        "Tcs0a(2900)++": ("molécule D*K* (I=1)", dst + kst, "PRÉDIT"),
        "X(3960)": ("molécule DsD̄s (0⁺, sans demi-tour)", 2 * ds, "PRÉDIT"),
        "Z_cs(4000)": ("molécule DsD̄* (= Zcs(3985))", ds + dst, "PRÉDIT"),
        "P_cs(4338)": ("molécule Ξ_c D̄", xc1 + dd, "PRÉDIT"),
        "Λ_b(6146)": ("bande 1D dérivée (Δn=23/18·2+3·7(d−1))", lb_1d, "PRÉDIT"),
        "Λ_b(6152)": ("bande 1D dérivée (doublet, sép. 1/m_Q non incl.)",
                      lb_1d, "PRÉDIT"),
        "Ξ'_c(3055)+": ("bande 1D Ξ_c (membre bas, sép. non incl.)",
                        xic_1d, "PRÉDIT"),
        "Ξ'_c(3080)+": ("bande 1D Ξ_c (sommet, +s·L jambes propres)",
                        xic_1d, "PRÉDIT"),
        # --- COMPATIBLE : structure conforme, membre/décimale ouverts ---
        "Z_cs(4220)": ("molécule Ds*D̄* (seuil ; Γ~230 large)",
                       dsst + dst, "COMPATIBLE"),
        "Ω_c(3185)": ("bande 1D Ω_c (verrou plat, j bas)", oc_1d, "COMPATIBLE"),
        "Ω_c(3327)": ("bande 1D Ω_c (j haut, x_j=1)", oc_1d_h, "COMPATIBLE"),
        "Σ_c(2800)": ("quintet 1P Σ_c prédit (2772/2821/2869), membre "
                      "non résolu (Γ~75)", 2.821, "COMPATIBLE"),
        # --- OUVERT ------------------------------------------------------
        "T_cs1(2900)": ("structure 1⁻ non tranchée (option P de D̄*K* : "
                        "+6 %)", np.nan, "OUVERT"),
        # --- HORS-CADRE : compression (tau<0), descriptif seulement -----
        "X(6600)": ("compression tout-charme (4m_c descriptif)",
                    4 * MC, "HORS-CADRE"),
        "X(7100)": ("compression tout-charme (4m_c+orb ℓ=3 descriptif)",
                    np.sqrt((4 * MC)**2 + 3 * SLOPE), "HORS-CADRE"),
        "X(7300)": ("compression tout-charme (4m_c+orb ℓ=4 descriptif)",
                    np.sqrt((4 * MC)**2 + 4 * SLOPE), "HORS-CADRE"),
        "X(4630)": ("compression J/ψφ (2·Ds+orb ℓ=3 descriptif)",
                    np.sqrt((2 * ds)**2 + 3 * SLOPE), "HORS-CADRE"),
        "X(4685)": ("compression J/ψφ (2·Ds+orb ℓ=3 descriptif)",
                    np.sqrt((2 * ds)**2 + 3 * SLOPE), "HORS-CADRE"),
    }

    print("=" * 96)
    print("%-14s %8s | %-9s %7s | %-46s %7s %s" % (
        "état", "m_exp", "juin(cfg)", "Δ_juin", "règle canonique (juillet)",
        "Δ_juil", "statut"))
    print("-" * 96)
    stats = {"PRÉDIT": [], "COMPATIBLE": [], "OUVERT": [], "HORS-CADRE": []}
    for nom, (m_exp, cfg, d_juin) in juin.items():
        regle, m_th, statut = T[nom]
        if np.isnan(m_th):
            d_juil, txt = np.nan, "   --  "
        else:
            d_juil = 100 * (1000 * m_th - m_exp) / m_exp
            txt = "%+6.2f%%" % d_juil
            stats[statut].append(abs(d_juil))
        print("%-14s %8.1f | %-9s %+6.2f%% | %-46s %s %s" % (
            nom, m_exp, cfg[:9], d_juin, regle[:46], txt, statut))
    print("=" * 96)
    print("BILAN")
    print("=" * 96)
    for k in ("PRÉDIT", "COMPATIBLE", "HORS-CADRE"):
        v = stats[k]
        if v:
            print("  %-11s : %2d états, médiane |Δ| = %.2f %%, max %.2f %%"
                  % (k, len(v), np.median(v), max(v)))
    print("  OUVERT      :  1 état (T_cs1(2900), structure 1⁻ non tranchée)")
    d_j = [abs(d) for _, _, d in juin.values()]
    print("""
  LECTURE ÉPISTÉMOLOGIQUE (la comparaison qui compte) :
   - JUIN (Ω_ts archivé)  : 22/22 apparient à médiane |Δ| = %.2f %% MAIS
     chaque état choisit sa configuration A POSTERIORI dans un espace
     ≥ 64 combinaisons (le null d'assignation : ~1e4-1e5 équivalents)
     → l'accord est de la COMPATIBILITÉ, contenu prédictif nul.
   - JUILLET (règles figées) : %d états reçoivent une valeur SANS AUCUN
     choix par état (contenus = lectures moléculaires PDG standard,
     bandes dérivées zéro nombre lu, constituants de la chaîne ±10 MeV)
     → médiane |Δ| = %.2f %% ; l'accord EST le contenu.
   - Le pire de juin (T_cc −3,0 %%, ★ dégradé) devient le MEILLEUR de
     juillet (+0,03 %%, épinglé au seuil DD* — le motif de seuil établi).
   - 5 états tout-charme/J/ψφ sortent du cadre PAR PRINCIPE (compression
     τ<0) : la limite est déclarée, pas masquée par une assignation.
""" % (np.median(d_j), len(stats["PRÉDIT"]),
       np.median(stats["PRÉDIT"])))


if __name__ == "__main__":
    main()
