"""
Scan LZ des 6 candidats STRICT amphichiraux du catalogue V5
(TQNT_catalog_V5_diffusion.xlsx onglet 5).

Question posée : sous Vassiliev V5 §13.6.4
    σ_DM-N(c) ≈ σ_EM · α_EM^(c - c*)    avec c* = 18

l'un de ces candidats améliore-t-il l'indice de la session V5.3 par rapport
au verdict 'partial WIN sur 6_3+6_3' ?

Liste (CLAUDE.md, masses prédites en GeV) :
    6_3 + 6_3            → 1,55 GeV     c = 12
    6_3 + 6_3 + 6_3      → 4,70 GeV     c = 18
    4_1 + 8_3            → 16,01 GeV    c = 12
    8_3 + 8_3            → 17,24 GeV    c = 16
    6_3 + 8_3            → 26,48 GeV    c = 14
    4_1 + 4_1 + 8_3      → 51,78 GeV    c = 16

Tous ont c ≤ c* = 18. Sous Vassiliev naïf, ils sont TOUS au-dessus de LZ.
"""

import math

ALPHA_EM = 1.0 / 137.036
SIGMA_LZ = 9.2e-48                    # cm² (LZ 2024)
C_STAR_V5 = 18                        # V5 canonique
SIGMA_EM_CALIB = 1.0e-27 / ALPHA_EM   # calibré : σ(c=19) ~ 1e-27 cm²

CANDIDATES = [
    ("6_3 + 6_3",            12, 1.55),
    ("6_3 + 6_3 + 6_3",      18, 4.70),
    ("4_1 + 8_3",            12, 16.01),
    ("8_3 + 8_3",            16, 17.24),
    ("6_3 + 8_3",            14, 26.48),
    ("4_1 + 4_1 + 8_3",      16, 51.78),
]


def sigma_vassiliev(c, c_star=C_STAR_V5):
    """V5 §13.6.4 : σ_DM-N(c) ≈ σ_EM · α_EM^(c - c*)."""
    return SIGMA_EM_CALIB * ALPHA_EM ** (c - c_star)


def main():
    print("=" * 78)
    print("Scan LZ des 6 candidats STRICT amphichiraux V5 sous Vassiliev")
    print("=" * 78)
    print(f"  Formule V5 §13.6.4 : σ_DM-N(c) = σ_EM · α_EM^(c - c*)")
    print(f"  c* = {C_STAR_V5} (V5 canonique : nucléon + phase Gauss)")
    print(f"  σ_EM (calibré σ(c=19)~10⁻²⁷) = {SIGMA_EM_CALIB:.3e} cm²")
    print(f"  LZ 2024 : σ_LZ = {SIGMA_LZ:.2e} cm²")
    print()
    print(f"  {'Entrelacs':>18} | {'c':>3} | {'m':>8} | {'σ(c) [cm²]':>13} "
          f"| {'écart LZ':>10} | statut")
    print("─" * 78)

    for name, c, mass in CANDIDATES:
        sigma = sigma_vassiliev(c)
        dex_over_LZ = math.log10(sigma / SIGMA_LZ)
        status = "SAFE" if sigma < SIGMA_LZ else "EXCLU"
        print(f"  {name:>18} | {c:>3} | {mass:>5.2f} GeV | "
              f"{sigma:>13.2e} | {dex_over_LZ:>+7.1f} dex | {status}")

    print()
    print("─" * 78)
    print("Pour mémoire — les candidats du réservoir V5 dominant (c ≥ 19)")
    print("─" * 78)
    print(f"  {'c':>3} | {'σ(c) [cm²]':>13} | {'écart LZ':>10} | statut")
    for c in [19, 22, 25, 28, 29, 30]:
        sigma = sigma_vassiliev(c)
        dex_over_LZ = math.log10(sigma / SIGMA_LZ)
        status = "SAFE" if sigma < SIGMA_LZ else "EXCLU"
        print(f"  {c:>3} | {sigma:>13.2e} | {dex_over_LZ:>+7.1f} dex | {status}")

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("""
1. Sous Vassiliev V5 naïf, les 6 STRICT sont TOUS au-dessus de LZ
   de 21 à 35 décades. Aucun choix interne à la liste n'améliore l'indice.

2. La structure du problème : V5 §13.6.4 fixe le couplage σ(c) par le seul
   c (crossing number). Comme les 6 STRICT ont c ∈ {12,14,16,18}, ils
   partagent tous le même régime σ ≥ σ_EM.

3. Pour atteindre la bande LZ-safe sous Vassiliev, il faut c ≥ 29. Aucun
   entrelacs amphichiral de la liste CLAUDE.md n'atteint cette barre
   (max = 18 pour 6_3+6_3+6_3).

4. Conséquence pour la session V5.3 :
   - Le 'partial WIN SPARC sur 6_3+6_3' ne tient PAS sous Vassiliev naïf
     (le candidat serait détecté à 35 dex au-dessus de LZ).
   - Le PARTIAL WIN ne peut être sauvé QU'AU PRIX d'un argument
     structurel : amphichirales pures → Q_phys = 0 EXACTEMENT →
     couplage EM nul à l'ordre dominant → σ_DM-N gouvernée par un
     canal sous-dominant (gluonique, gravitationnel) NON donné par
     V5 §13.6.4.

5. Cet argument 'amphichirales échappent à Vassiliev' est PHYSIQUE
   (Q_phys = 0 tue le diagramme γ-échange dominant) mais N'EST PAS
   ÉCRIT dans V5 §13.6.4. C'est une extension à clarifier dans V5.4.

6. Question subsidiaire : existe-t-il dans le catalogue xlsx (117 candidats)
   un amphichiral c ≥ 29 ? La liste CLAUDE.md ne le mentionne pas mais
   restreint aux 6 STRICT 'amphichiraux purs'. Les 111 RÉALISABLES
   (Q_phys=0 via Lk entier, pas amphichirales pures) pourraient en
   contenir. À vérifier dans le catalogue xlsx pour proposer un
   meilleur candidat.

CONCLUSION OPÉRATIONNELLE :
- Aucune substitution parmi la liste actuelle n'améliore l'indice V5.3.
- Soit l'argument 'Q_phys=0 → suppression hors-Vassiliev' tient (et alors
  TOUS les 6 STRICT survivent à LZ, mais V5 §13.6.4 doit être amendée),
- soit il ne tient pas (et alors le partial WIN SPARC se dissout en LOSS
  car le candidat 6_3+6_3 serait vu par LZ à 35 dex).
""")


if __name__ == "__main__":
    main()
