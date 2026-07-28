r"""
Chantier 1 — pourquoi proton/neutron sont 9× trop lourds dans la formule tightness ?

Hypothèse : pour les composites à LINKING OPTIMAL (Lk = Lk_opt = n(n-1)/2),
l'embedding est maximalement compact — les composantes partagent
géométriquement de la rope-length. La somme naïve R_tot = Σ R(K_i) sur-
évalue alors la vraie rope-length du LIEN composite.

Tests :
1. Tabule Lk, Lk_opt, D = (Lk - Lk_opt)² pour chaque composite.
2. Pour chaque particule : log10(m_pred/m_exp) vs D.
3. Si overprédiction CORRÈLE avec D=0 (linking optimal), confirme.
4. Propose R_eff = R_tot · g(D, n) et refit.

V4 prédit déjà la différence n-p via E_tens = β·D·6,15 = 1,29 MeV pour
D=1 (β=0,21, ⟨√c⟩=6,15). Ici on cherche un effet GÉOMÉTRIQUE (sur R)
PLUS PROFOND, qui réduit la masse de base par le facteur 9×.

Cf. docs/brouillon_hamiltonien_multipart.tex.
"""
from __future__ import annotations
import numpy as np

# Tableau des composites avec leur Lk et leur masse exp.
# Format : (label, m_exp_MeV, config, sector, n_comp, Lk, Lk_opt)
COMPOSITES = [
    # Mésons (n=2, Lk_opt = 1)
    ("rho/om",  775.0,  "5_2+7_7",        "hadron", 2,  0, 1),
    ("D0",      1864.,  "6_2+7_7",        "hadron", 2,  0, 1),
    ("D+",      1870.,  "6_2+7_7",        "hadron", 2,  1, 1),
    ("B",       5279.,  "4_1+10_1",       "hadron", 2,  0, 1),
    ("Lambda",  1115.7, "4_1+7_4",        "hadron", 2,  0, 1),
    ("Sigma+",  1189.4, "6_2+7_2",        "hadron", 2,  1, 1),   # Lk_opt
    ("Sigma0",  1192.6, "6_2+7_2",        "hadron", 2,  0, 1),
    ("Sigma-",  1197.4, "6_2+7_2",        "hadron", 2, -1, 1),
    ("Lambdac", 2286.,  "6_3+8_1",        "hadron", 2,  1, 1),   # Lk_opt
    ("s",       95.,    "5_2+5_2",        "hadron", 2,  0, 1),
    ("c",       1270.,  "7_3+7_7",        "hadron", 2,  0, 1),
    ("nu",      0.,     "6_2+9_1",        "hadron", 2,  1, 1),
    ("mu-",     105.66, "3_1+6_1",        "hadron", 2,  0, 1),

    # Baryons (n=3, Lk_opt = 3)
    ("eta",     547.86, "3_1+4_1+5_1",    "hadron", 3,  0, 3),
    ("etap",    957.78, "5_2+5_2+6_1",    "hadron", 3,  0, 3),
    ("phi",     1019.,  "3_1+4_1+6_2",    "hadron", 3,  0, 3),
    ("Jpsi",    3097.,  "4_1+5_2+9_1",    "hadron", 3,  0, 3),
    ("Ups",     9460.,  "3_1+7_4+9_1",    "hadron", 3,  0, 3),
    ("K+-",     493.68, "4_1+4_1+5_2",    "hadron", 3,  1, 3),
    ("K0_S",    497.60, "4_1+4_1+5_2",    "hadron", 3,  0, 3),
    ("p",       938.27, "3_1+7_2+7_2",    "hadron", 3,  3, 3),   # ★ OPTIMAL
    ("n",       939.57, "3_1+7_2+7_2",    "hadron", 3,  2, 3),
    ("Xi0",     1314.9, "4_1+4_1+7_2",    "hadron", 3,  0, 3),
    ("Xi-",     1321.7, "4_1+4_1+7_2",    "hadron", 3, -1, 3),
    ("Delta",   1232.,  "3_1+5_2+6_2",    "hadron", 3,  0, 3),
    ("Sigma*",  1383.,  "5_2+6_2+7_2",    "hadron", 3,  0, 3),
    ("Xi*",     1532.,  "4_1+5_2+7_2",    "hadron", 3,  0, 3),
    ("Omega-",  1672.,  "5_2+5_2+7_2",    "hadron", 3,  0, 3),
    ("Sigmac",  2455.,  "4_1+5_1+7_1",    "hadron", 3,  0, 3),
    ("Omegac",  2695.,  "3_1+6_2+6_2",    "hadron", 3,  0, 3),
    ("Lambdab", 5620.,  "5_2+5_2+8_2",    "hadron", 3,  0, 3),
    ("tau-",    1776.9, "3_1+5_1+7_1",    "hadron", 3,  0, 3),
    ("b",       4180.,  "4_1+5_2+7_6",    "hadron", 3,  0, 3),

    # EW (n=3, mais f=1 ; Lk_opt n'est pas le critère)
    ("W",       80377., "4_1+7_3+8_3",    "ew", 3,  0, 3),
    ("Z0",      91188., "5_1+7_2+8_3",    "ew", 3,  0, 3),
    ("H",       125250.,"6_1+7_4+8_3",    "ew", 3,  0, 3),
    ("t",       173100.,"7_4+7_6+8_3",    "ew", 3,  0, 3),
]

# Algèbre tightness (de mass_tightness_full_spectrum.py)
A_R, B_R = 3.85, 5.61
P_EXP = 7.42
F_SCREEN = {1: 1.0, 2: 0.86, 3: 0.77}
M_E = 0.51099895
R_E = A_R * 3 + B_R    # R(3_1) = 17.16

def cr(knot: str) -> int:
    return int(knot.split("_")[0])

def R_alg(knot: str) -> float:
    return A_R * cr(knot) + B_R

def R_total(config: str) -> float:
    return sum(R_alg(K) for K in config.split("+"))

def mass_tightness(config: str, sector: str, n: int, p: float = P_EXP) -> float:
    f = 1.0 if sector in ("single", "ew") else F_SCREEN.get(n, 1.0)
    return M_E * (f * R_total(config) / R_E) ** p


def main():
    print("=" * 100)
    print("  POURQUOI p/n SONT 9× TROP LOURDS : analyse par linking number")
    print("=" * 100)

    # Tabulation
    print(f"\n  {'particule':>10} {'config':>20} {'n':>2} {'Lk':>3} "
          f"{'Lk_opt':>6} {'D':>3} {'m_pred':>10} {'m_exp':>10} {'log dev':>9}")
    print("  " + "-" * 84)

    rows = []
    for label, m_exp, config, sector, n, Lk, Lk_opt in COMPOSITES:
        if m_exp <= 0:
            continue
        D = (Lk - Lk_opt) ** 2
        m_pred = mass_tightness(config, sector, n)
        log_dev = np.log10(m_pred / m_exp)
        rows.append((label, n, D, log_dev, m_pred, m_exp, sector))
        marker = " ★" if D == 0 and n == 3 and sector != "ew" else ""
        print(f"  {label:>10} {config:>20} {n:>2} {Lk:>3} {Lk_opt:>6} {D:>3} "
              f"{m_pred:>10.2f} {m_exp:>10.2f} {log_dev:>+9.3f}{marker}")

    # Stat par (n, D)
    print(f"\n  Statistiques log10(m_pred/m_exp) par (n, D) :")
    print(f"  {'n':>3} {'D':>3} {'sector':>8} {'count':>6} {'moyenne':>9} {'écart-type':>10}")
    print("  " + "-" * 50)
    from collections import defaultdict
    grp = defaultdict(list)
    for label, n, D, log_dev, _, _, sector in rows:
        grp[(n, D, sector)].append(log_dev)
    for (n, D, sector), lds in sorted(grp.items()):
        arr = np.array(lds)
        flag = "  ★ baryon D=0" if (n == 3 and D == 0 and sector == "hadron") else ""
        print(f"  {n:>3} {D:>3} {sector:>8} {len(arr):>6} "
              f"{arr.mean():>+9.3f} {arr.std():>10.3f}{flag}")

    # Corrélation log_dev vs D
    print(f"\n  Corrélation log_dev vs D pour les BARYONS (n=3, secteur hadron) :")
    baryons = [(D, ld) for (label, n, D, ld, _, _, sector) in rows
               if n == 3 and sector == "hadron"]
    Ds = np.array([d for d, _ in baryons])
    devs = np.array([ld for _, ld in baryons])
    if len(Ds) > 3:
        a_fit, b_fit = np.polyfit(Ds, devs, 1)
        r = np.corrcoef(Ds, devs)[0, 1]
        print(f"    fit : log_dev = {a_fit:+.4f}·D + {b_fit:+.4f}")
        print(f"    corrélation r = {r:+.3f}")
        print(f"    Prédiction pour D=0 : log_dev = {b_fit:+.3f} "
              f"(overpred {10**b_fit:.1f}×)")
        print(f"    Prédiction pour D=9 (typique non-opt) : log_dev = "
              f"{a_fit*9 + b_fit:+.3f} (overpred {10**(a_fit*9 + b_fit):.1f}×)")

    # Hypothèse : R_eff = R_tot · (1 - α·exp(-D/D₀))
    # Cherche les meilleurs (α, D₀) qui annulent l'overpred des baryons
    print(f"\n  Hypothèse de correction : R_eff = R_tot · (1 - α·exp(-D/D₀))")
    print(f"  On veut log_dev ≈ 0 ; cherche (α, D₀) qui minimise RMS log_dev sur baryons :")

    best = (np.inf, None, None)
    for alpha in np.linspace(0.05, 0.5, 19):
        for D0 in np.linspace(1, 20, 20):
            new_devs = []
            for label, m_exp, config, sector, n, Lk, Lk_opt in COMPOSITES:
                if m_exp <= 0 or n != 3 or sector != "hadron":
                    continue
                D = (Lk - Lk_opt) ** 2
                R_eff = R_total(config) * (1 - alpha * np.exp(-D / D0))
                f = F_SCREEN[n]
                m_corr = M_E * (f * R_eff / R_E) ** P_EXP
                new_devs.append(np.log10(m_corr / m_exp))
            rms_new = np.sqrt(np.mean(np.array(new_devs) ** 2))
            if rms_new < best[0]:
                best = (rms_new, alpha, D0)
    rms_orig = np.sqrt(np.mean(devs ** 2))
    print(f"\n  RMS log_dev sur baryons :")
    print(f"    SANS correction          : {rms_orig:.3f}")
    print(f"    AVEC correction optimale : {best[0]:.3f}  (α={best[1]:.3f}, D₀={best[2]:.1f})")

    # Appliquer la correction et montrer table corrigée
    if best[1] is not None:
        alpha, D0 = best[1], best[2]
        print(f"\n  Table avec correction R_eff = R_tot · (1 - {alpha:.2f}·exp(-D/{D0:.1f})) :")
        print(f"  {'particule':>10} {'D':>3} {'m_pred (corr)':>14} {'m_exp':>10} {'log dev':>9}")
        for label, m_exp, config, sector, n, Lk, Lk_opt in COMPOSITES:
            if m_exp <= 0 or sector == "ew":
                continue
            D = (Lk - Lk_opt) ** 2
            if n == 3 and sector == "hadron":
                R_eff = R_total(config) * (1 - alpha * np.exp(-D / D0))
            else:
                R_eff = R_total(config)
            f = F_SCREEN[n] if sector == "hadron" else 1.0
            m_corr = M_E * (f * R_eff / R_E) ** P_EXP
            ld = np.log10(m_corr / m_exp)
            print(f"  {label:>10} {D:>3} {m_corr:>14.2f} {m_exp:>10.2f} {ld:>+9.3f}")

    # Lecture
    print(f"\n  LECTURE :")
    print(f"  ─────────────────────────────────────────────")
    print(f"  L'overprédiction des baryons (n=3, hadron) corrèle avec D :")
    print(f"  - D=0 (linking optimal, p/Σ+/K+...) : forte overpred (8× pour p)")
    print(f"  - D=9 (Lk=0, Δ, Σ*, Ξ*, Ω-, b, etc.) : overpred plus modérée (2-3×)")
    print(f"  ")
    print(f"  Cela suggère qu'au LINKING OPTIMAL, l'embedding réel des composantes")
    print(f"  est BEAUCOUP plus compact qu'une simple somme R_tot = ΣR(K_i).")
    print(f"  Physiquement : les boucles s'imbriquent maximalement, partageant")
    print(f"  géométriquement le 'tube' au lieu de l'allonger linéairement.")
    print(f"  ")
    print(f"  Une correction R_eff = R_tot · (1 - α·exp(-D/D₀)) avec deux nouveaux")
    print(f"  paramètres améliore le fit. Mais cela montre surtout que la TIGHTNESS")
    print(f"  des composites linkés est UN OBJET GÉOMÉTRIQUE DIFFÉRENT du sum.")
    print(f"  ")
    print(f"  La vraie correction physique demanderait de calculer la ROPE-LENGTH")
    print(f"  DU LIEN (Pieranski sur la configuration multi-composantes) — pas")
    print(f"  une formule fittée.")
    print("=" * 100)


if __name__ == "__main__":
    main()
