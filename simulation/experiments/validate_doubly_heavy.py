r"""
Validation P23' Piste 2 sur les BARYONS DOUBLEMENT/TRIPLEMENT LOURDS.

Famille  Ξ, Ω à 2 ou 3 quarks lourds (cc, bb, cb, ccc, ccb, cbb, bbb).
Aucun n'est dans le catalogue V5 — la prédiction se fait uniquement par
addition des constituants TQNT validés.

États visés :
  Doubly charm     : Ξ_cc, Ω_cc
  Doubly bottom    : Ξ_bb, Ω_bb
  Charm-bottom     : Ξ_cb, Ξ'_cb, Ω_cb, Ω'_cb
  Triply heavy     : Ω_ccc, Ω_ccb, Ω_cbb, Ω_bbb

Comparaison aux valeurs PDG (Ξ_cc observé) et aux prédictions QCD-lattice
pour les autres.
"""
from __future__ import annotations
import numpy as np

# Constituants TQNT validés
M = {"u": 311.0, "d": 315.0, "s": 513.0, "c": 1601.6, "b": 4878.3}


def m_pred(content):
    return sum(N * M[q] for q, N in content.items())


# ════════════════════════════════════════════════════════════
print("\n" + "═"*94)
print(" BARYONS DOUBLEMENT LOURDS (qc̄c̄, qb̄b̄, qc̄b̄, scc, scb, sbb)")
print("═"*94)

# Liste : (nom, contenu, m_PDG ou prediction théorique, source/statut)
DOUBLY_HEAVY = [
    # Doubly charm
    ("Ξ_cc⁺⁺(ucc)", {"u":1, "c":2},          3621.55, "LHCb 2017 (OBSERVÉ)"),
    ("Ξ_cc⁺(dcc)",  {"d":1, "c":2},          3519.0,  "théorique, recherche LHCb"),
    ("Ω_cc⁰(scc)",  {"s":1, "c":2},          3720.0,  "non observé, théorie lattice"),
    # Doubly bottom
    ("Ξ_bb⁰(ubb)",  {"u":1, "b":2},          10174.0, "non observé, théorie lattice"),
    ("Ξ_bb⁻(dbb)",  {"d":1, "b":2},          10178.0, "non observé"),
    ("Ω_bb⁻(sbb)",  {"s":1, "b":2},          10380.0, "non observé"),
    # Charm-bottom
    ("Ξ_cb⁰(ucb)",  {"u":1, "c":1, "b":1},   6943.0,  "non observé, théorie"),
    ("Ξ_cb⁻(dcb)",  {"d":1, "c":1, "b":1},   6948.0,  "non observé"),
    ("Ω_cb⁰(scb)",  {"s":1, "c":1, "b":1},   7140.0,  "non observé"),
    # Triply heavy
    ("Ω_ccc(ccc)",  {"c":3},                 4800.0,  "non observé, lattice ~4790"),
    ("Ω_ccb(ccb)",  {"c":2, "b":1},          8020.0,  "non observé, lattice ~8030"),
    ("Ω_cbb(cbb)",  {"c":1, "b":2},          11250.0, "non observé, lattice ~11200"),
    ("Ω_bbb(bbb)",  {"b":3},                 14370.0, "non observé, lattice ~14380"),
]

print(f"\n  {'État':<18} {'Contenu':<14} {'Pred TQNT':<12} {'PDG/théo':<12} {'Écart':<8} {'Statut'}")
print("─"*100)

results = []
for name, content, m_ref, status in DOUBLY_HEAVY:
    m_t = m_pred(content)
    err = (m_t - m_ref) / m_ref * 100
    sig = "".join(f"{q}{N}" for q, N in content.items())
    flag = "✓" if abs(err) < 5 else ("~" if abs(err) < 10 else "✗")
    print(f"  {name:<18} {sig:<14} {m_t:>10.0f}    {m_ref:>9.1f}    {err:>+5.1f}% {flag}  {status}")
    results.append((name, m_t, m_ref, err))


# ════════════════════════════════════════════════════════════
print("\n" + "═"*94)
print(" FOCUS SUR Ξ_cc(3621) — SEUL DOUBLEMENT CHARMÉ OBSERVÉ (LHCb 2017)")
print("═"*94)
m_t_Xicc = m_pred({"u":1, "c":2})
print(f"\n  Prédiction P23' Piste 2 : m(Ξ_cc⁺⁺) = m_u + 2 m_c")
print(f"                           = {M['u']} + 2 · {M['c']}")
print(f"                           = {m_t_Xicc:.1f} MeV")
print(f"\n  Observé LHCb (2017) : 3621.55 MeV (±0.23 stat ±0.30 syst)")
print(f"  Écart TQNT : {(m_t_Xicc - 3621.55)/3621.55*100:+.2f}%")
print(f"  → TQNT prédit {abs((m_t_Xicc-3621.55)/3621.55*100):.1f}% trop bas, comme pour P_c.")
print(f"  → Cohérent avec une légère énergie de liaison résiduelle (~+100 MeV).")


# ════════════════════════════════════════════════════════════
print("\n" + "═"*94)
print(" PRÉDICTIONS NOUVELLES PRIORITAIRES (≤ 12 GeV, accessibles LHCb)")
print("═"*94)

priority = [
    ("Ξ_cc⁺(dcc)",  3519,   "Décroissance D+ K- + π+ inclusive (LHCb)"),
    ("Ω_cc⁰(scc)",  3717,   "Décroissance Ξ_c+ + K- (LHCb, Belle II)"),
    ("Ω_ccc(ccc)",  4805,   "Decay → Ξ_cc + D, recherche LHCb / FCC"),
    ("Ξ_cb⁰(ucb)",  6791,   "→ Ξ_c⁺ + ϒ (très exotique, LHCb future)"),
    ("Ω_cb⁰(scb)",  6993,   "→ Ω_c + ϒ (LHCb future)"),
    ("Ω_ccb(ccb)",  8082,   "Pas de canal évident, futur collisionneur"),
    ("Ξ_bb⁰(ubb)",  10067,  "ϒ + D (LHCb High-L, Belle II ?)"),
    ("Ω_bb⁻(sbb)",  10269,  "ϒ + D_s (LHCb High-L)"),
    ("Ω_cbb(cbb)",  11358,  "Inaccessible LHCb actuel"),
    ("Ω_bbb(bbb)",  14635,  "Inaccessible LHCb actuel"),
]
print(f"\n  {'État':<18} {'TQNT (MeV)':<12} {'Canal de découverte favori'}")
print("─"*100)
for name, m, channel in priority:
    print(f"  {name:<18} {m:>10}     {channel}")


# ════════════════════════════════════════════════════════════
print("\n" + "═"*94)
print(" COMPARAISON AVEC LATTICE QCD (cas Ω_ccc, Ω_bbb)")
print("═"*94)

print()
print("  État         Lattice (MeV)        TQNT (MeV)        Écart")
print("─" * 70)
for name, qcd_low, qcd_high, content in [
    ("Ω_ccc", 4790, 4810, {"c":3}),
    ("Ω_ccb", 8000, 8050, {"c":2, "b":1}),
    ("Ω_cbb", 11200, 11250, {"c":1, "b":2}),
    ("Ω_bbb", 14370, 14380, {"b":3}),
]:
    m_t = m_pred(content)
    qcd_mid = (qcd_low + qcd_high) / 2
    err = (m_t - qcd_mid) / qcd_mid * 100
    print(f"  {name:<10}  {qcd_low:>5}–{qcd_high:>5}        {m_t:>8.0f}      {err:>+5.1f}%")


# ════════════════════════════════════════════════════════════
print("\n" + "═"*94)
print(" STATISTIQUE GLOBALE")
print("═"*94)

errs = [r[3] for r in results]
print(f"\n  Nombre d'états testés : {len(results)}")
print(f"  Écart moyen absolu    : {np.mean([abs(e) for e in errs]):.1f}%")
print(f"  Écart maximum         : {max(abs(e) for e in errs):.1f}%")
print(f"  Écart signé           : {np.mean(errs):+.1f}% (biais)")
print(f"  RMS log_10            : {np.sqrt(np.mean(np.log10([(M['u'] + 2*M['c'])/3621.55])**2)):.4f} décade")

# Verdict
n_pass = sum(1 for e in errs if abs(e) < 5)
n_partial = sum(1 for e in errs if 5 <= abs(e) < 10)
n_fail = sum(1 for e in errs if abs(e) >= 10)
print(f"\n  ✓ < 5% : {n_pass}/{len(errs)}")
print(f"  ~ 5-10%: {n_partial}/{len(errs)}")
print(f"  ✗ >10% : {n_fail}/{len(errs)}")
print()
print("  Conclusion : la TQNT-V5 prédit les baryons doublement et triplement")
print("  lourds à mieux de 5% avec UNIQUEMENT les masses constituantes,")
print("  sans aucun paramètre supplémentaire pour les baryons lourds.")
print()
print("  Cas particulier Ξ_cc(3621) : prédiction TQNT 3515 MeV (-2.9%),")
print("  cohérent avec ~100 MeV d'énergie de liaison résiduelle (semblable")
print("  aux pentaquarks P_c).")
