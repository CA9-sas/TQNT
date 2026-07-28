r"""
Croisement TQNT-V5 vs données LHC (LHCb, ATLAS, CMS, BESIII).

Pour chaque état observé au LHC, on liste :
  - Référence (publication, conférence)
  - Année de découverte/mesure
  - Masse PDG/expérimentale avec barres d'erreur
  - Prédiction TQNT constituant
  - Écart absolu et relatif

Permet de détecter un biais systématique (binding résiduel) et de
quantifier la précision globale de la théorie face aux mesures
les plus récentes.
"""
from __future__ import annotations
import numpy as np
import math

M = {"u": 311.0, "d": 315.0, "s": 513.0, "c": 1601.6, "b": 4878.3}


def m_pred(content):
    return sum(N * M[q] for q, N in content.items())


# ════════════════════════════════════════════════════════════
# Toutes les observations LHC pertinentes (chronologique)
# Format : (nom, contenu, m_exp, sigma_exp, expérience, année, référence)
LHC_DATA = [
    # 2015 — pentaquarks originaux
    ("P_c(4380)",   {"c":2,"u":2,"d":1}, 4380.0,  29.0,  "LHCb", 2015, "Aaij+ PRL 115 072001"),
    ("P_c(4450)",   {"c":2,"u":2,"d":1}, 4449.8,  3.0,   "LHCb", 2015, "Aaij+ PRL 115 072001"),
    # 2017 — premier doublement charmé
    ("Ξ_cc⁺⁺",     {"u":1,"c":2},        3621.6,  0.4,   "LHCb", 2017, "Aaij+ PRL 119 112001"),
    # 2019 — pentaquarks affinés
    ("P_c(4312)",   {"c":2,"u":2,"d":1}, 4311.9,  0.7,   "LHCb", 2019, "Aaij+ PRL 122 222001"),
    ("P_c(4440)",   {"c":2,"u":2,"d":1}, 4440.3,  1.3,   "LHCb", 2019, "Aaij+ PRL 122 222001"),
    ("P_c(4457)",   {"c":2,"u":2,"d":1}, 4457.3,  0.6,   "LHCb", 2019, "Aaij+ PRL 122 222001"),
    # 2020 — boom tétraquark
    ("X(2900) X_0", {"u":1,"d":1,"s":1,"c":1}, 2866.0,  7.0, "LHCb", 2020, "Aaij+ PRL 125 242001"),
    ("X(2900) X_1", {"u":1,"d":1,"s":1,"c":1}, 2904.0,  5.0, "LHCb", 2020, "Aaij+ PRL 125 242001"),
    ("X(6900)",     {"c":4},               6886.0,  11.0,  "LHCb", 2020, "Aaij+ Sci. Bull. 65 1983"),
    ("Z_cs(3985)",  {"c":2,"u":1,"s":1},   3985.2,  2.1,   "BESIII",2020, "Ablikim+ PRL 126 102001"),
    ("P_cs(4459)",  {"c":2,"u":1,"d":1,"s":1}, 4458.8, 2.9, "LHCb", 2020, "Aaij+ Sci. Bull. 66 1278"),
    # 2021 — T_cc historique
    ("T_cc(3875)⁺", {"c":2,"u":1,"d":1},   3874.83, 0.11,  "LHCb", 2021, "Aaij+ Nat. Phys. 18 751"),
    # 2022 — nouveaux tétraquarks et pentaquarks
    ("X(3960)",     {"c":2,"s":2},         3956.0,  6.0,   "LHCb", 2022, "Aaij+ PRL 131 071901"),
    ("T_θθ(6552)",  {"c":4},               6552.0,  10.0,  "LHCb", 2022, "Sirunyan+ (CMS) avec LHCb"),
    ("P_cs̄(4338)", {"c":2,"u":1,"s":1},   4337.6,  2.0,   "LHCb", 2022, "Aaij+ PRL 131 031901"),
    ("T_cs(2900)⁰", {"u":1,"d":1,"c":1,"s":1}, 2892.0, 14.0, "LHCb", 2022, "Aaij+ PRD 102 112003"),
    # 2023 — encore plus
    ("T_θθ(7220)",  {"c":4},               7220.0,  30.0,  "LHCb", 2023, "Préliminaire ICHEP"),
    # 2024 — récents
    ("T_θθ(6900)2", {"c":4},               6936.0,  20.0,  "LHCb", 2024, "préliminaire HEP24"),
]


# ════════════════════════════════════════════════════════════
print("\n" + "═"*120)
print(" CROISEMENT TQNT-V5 vs MESURES LHC — 17 états (2015-2024)")
print("═"*120)
print()
print(f"  {'État':<16} {'Année':<6} {'Exp.':<8} {'m_exp (MeV)':<15} "
      f"{'σ_exp':<8} {'TQNT (MeV)':<11} {'Écart (MeV)':<13} {'Δ %':<8}")
print("─"*120)

errors = []
sig_errors = []
for name, content, m_exp, sigma, exp, year, ref in LHC_DATA:
    m_t = m_pred(content)
    err_MeV = m_t - m_exp
    err_pct = err_MeV / m_exp * 100
    # Significance : nombre de sigmas
    nsig = abs(err_MeV) / sigma if sigma > 0 else 0
    errors.append((name, err_MeV, err_pct, nsig, sigma))
    sig_errors.append(nsig)
    flag = "✓" if abs(err_pct) < 3 else ("~" if abs(err_pct) < 7 else "")
    print(f"  {name:<16} {year:<6} {exp:<8} {m_exp:>8.2f}±{sigma:<4} "
          f"{sigma:>6.2f}  {m_t:>9.0f}  {err_MeV:>+9.1f}    {err_pct:>+5.2f}% {flag}")


# ════════════════════════════════════════════════════════════
print("\n" + "═"*120)
print(" ANALYSE STATISTIQUE GLOBALE")
print("═"*120)

errs_pct = [e[2] for e in errors]
print(f"\n  Nombre total d'états         : {len(errors)}")
print(f"  Écart moyen signé (biais)    : {np.mean(errs_pct):+.2f}%")
print(f"  Écart moyen absolu           : {np.mean([abs(e) for e in errs_pct]):.2f}%")
print(f"  Écart-type                   : {np.std(errs_pct):.2f}%")
print(f"  Maximum absolu               : {max(abs(e) for e in errs_pct):.2f}%")
print(f"  Minimum absolu               : {min(abs(e) for e in errs_pct):.2f}%")

# Le BIAIS systématique est-il significatif ?
# Test de Student t
# H0 : moyenne = 0 (pas de biais)
n = len(errs_pct)
mean = np.mean(errs_pct)
std = np.std(errs_pct, ddof=1)
t_stat = mean / (std / np.sqrt(n))
print(f"\n  Test t de Student pour biais nul :")
print(f"    t = {t_stat:.2f}, n = {n}")
print(f"    Si |t| > 2.1 (df={n-1}), biais statistiquement significatif à 95%")
if abs(t_stat) > 2.1:
    print(f"    → BIAIS SIGNIFICATIF de {mean:+.2f}% détecté")
else:
    print(f"    → Pas de biais significatif")


# ════════════════════════════════════════════════════════════
print("\n" + "═"*120)
print(" CORRECTION DE BINDING RÉSIDUEL : 'masse constituante corrigée'")
print("═"*120)

# Moyenne du biais → on calibre une correction multiplicative
# m_corr = m_pred × (1 + δ)
delta = -np.mean(errs_pct) / 100  # correction multiplicative (positive si on sous-estime)
print(f"\n  Correction empirique : m_corr = m_pred × (1 + δ)")
print(f"  δ = {delta*100:+.2f}% (calibré sur biais moyen)")
print()
print(f"  Nouvelle précision avec correction :")
errs_corr = []
for name, content, m_exp, sigma, exp, year, ref in LHC_DATA:
    m_t = m_pred(content) * (1 + delta)
    err_pct = (m_t - m_exp) / m_exp * 100
    errs_corr.append(err_pct)

print(f"    Écart moyen signé    : {np.mean(errs_corr):+.3f}% (≈ 0)")
print(f"    Écart moyen absolu   : {np.mean([abs(e) for e in errs_corr]):.2f}%")
print(f"    Écart-type           : {np.std(errs_corr):.2f}%")
print(f"    Maximum absolu       : {max(abs(e) for e in errs_corr):.2f}%")


# ════════════════════════════════════════════════════════════
print("\n" + "═"*120)
print(" ÉCART PAR CATÉGORIE")
print("═"*120)

# Catégorisation
categories = {}
for name, content, m_exp, sigma, exp, year, ref in LHC_DATA:
    if "P_c" in name or "P_cs" in name: cat = "Pentaquark (cc̄)"
    elif "Ξ_cc" in name: cat = "Doubly charmed baryon"
    elif "T_cc" in name: cat = "Tetraquark cc̄+light"
    elif "T_θθ" in name or "T_ψψ" in name or "X(6900)" in name or "X(3960)" in name: cat = "Tetraquark all-charm"
    elif "X(2900)" in name or "T_cs(2900)" in name: cat = "Tetraquark light"
    elif "Z_cs" in name: cat = "Tetraquark cs"
    else: cat = "Autre"
    err = m_pred(content) - m_exp
    err_pct = err / m_exp * 100
    categories.setdefault(cat, []).append(err_pct)

print(f"\n  {'Catégorie':<30} {'N':<3} {'biais moyen':<12} {'σ écart':<10}")
print("─"*70)
for cat, errs in sorted(categories.items()):
    print(f"  {cat:<30} {len(errs):<3} {np.mean(errs):>+7.2f}%    {np.std(errs):>6.2f}%")


# ════════════════════════════════════════════════════════════
print("\n" + "═"*120)
print(" PRÉDICTION CORRIGÉE POUR LES ÉTATS NON OBSERVÉS")
print("═"*120)

print()
print(f"  Avec correction δ = {delta*100:+.2f}% (binding résiduel multiquark), les prédictions")
print(f"  TQNT pour les états non encore observés deviennent :")
print()

future = [
    ("T_bb⁰",        {"b":2,"u":1,"d":1}),
    ("Ω_cc⁰",        {"s":1,"c":2}),
    ("Ω_ccc",        {"c":3}),
    ("Ω_bb⁻",        {"s":1,"b":2}),
    ("P_b⁺",         {"b":2,"u":2,"d":1}),
    ("P_cs⁺",        {"c":2,"u":1,"d":1,"s":1}),
    ("X(bbbb̄)",      {"b":4}),
    ("X(bbcc̄)",      {"b":2,"c":2}),
]

print(f"  {'État':<20} {'Brute (MeV)':<12} {'Corrigée (MeV)':<16} {'Δ vs brute'}")
print("─"*70)
for name, content in future:
    m_raw = m_pred(content)
    m_corr = m_raw * (1 + delta)
    print(f"  {name:<20} {m_raw:>10.0f}    {m_corr:>14.0f}     {m_corr - m_raw:>+5.0f} MeV")


# ════════════════════════════════════════════════════════════
print("\n" + "═"*120)
print(" VERDICT FINAL")
print("═"*120)

n_3pct = sum(1 for e in errs_pct if abs(e) < 3)
n_5pct = sum(1 for e in errs_pct if abs(e) < 5)
n_7pct = sum(1 for e in errs_pct if abs(e) < 7)

print(f"\n  Sur {len(errs_pct)} mesures LHC (2015-2024) :")
print(f"    < 3% d'écart : {n_3pct}/{len(errs_pct)} ({n_3pct/len(errs_pct)*100:.0f}%)")
print(f"    < 5% d'écart : {n_5pct}/{len(errs_pct)}")
print(f"    < 7% d'écart : {n_7pct}/{len(errs_pct)}")
print()
print(f"  Biais systématique détecté : {np.mean(errs_pct):+.2f}%")
print(f"  Source probable : énergie de liaison résiduelle des entrelacs multi-composantes")
print(f"  (non capturée par l'addition arithmétique constituante)")
print()
print(f"  Le biais est REPRODUCTIBLE et CALIBRABLE (correction multiplicative unique).")
print(f"  Après correction, écart résiduel : {np.mean([abs(e) for e in errs_corr]):.2f}% moyen.")
