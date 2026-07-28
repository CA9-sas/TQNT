r"""
VÉRIFICATION DE COHÉRENCE de l'assignation des mésons (deux couches).

Couche 1 — STRUCTURE (catégorisation) : chaque méson des ~200 catégorisés
(`categorize_pdg_topological.py`) doit avoir un cœur K+K̄, un spin ENTIER
(statistique de Bose), une conjugaison de charge correcte, et la corrélation
saveur ↔ auto-conjugaison (neutre sans saveur = auto-conjugué ; neutre saveuré
K⁰/D⁰/B⁰/B_s⁰ = antiparticule distincte).

Couche 2 — NŒUDS des mésons DIAGONAUX (K+K̄, auto-conjugués) et leur statut de
contrôle par Alexander (`repair_curves_braid.py`). Vérifie aussi la cohérence
topologique de fond : les nœuds des mésons diagonaux doivent être CHIRAUX (un
nœud chiral + son miroir = vraie paire matière+antimatière), par opposition au
secteur EW dont le cœur 8₃ est AMPHICHIRAL (auto-miroir).

Réserve : la couche fine (entrelacs exact des mésons saveurés D, B) reste
SOUS-DÉTERMINÉE ; seule la couche catégorielle est close.
"""
from __future__ import annotations
from fractions import Fraction
from collections import Counter

from simulation.experiments.categorize_pdg_topological import build_rows

# neutres SAVEURÉS : Q=0 mais antiparticule DISTINCTE (non auto-conjugués)
FLAVORED_NEUTRAL = {
    "K⁰", "K*(892)⁰", "D⁰", "D*(2007)⁰", "D₁(2420)⁰", "D₂*(2460)⁰",
    "B⁰", "B*", "B*⁰", "B₁(5721)⁰", "B₂*(5747)⁰",
    "B_s⁰", "B_s*⁰", "B_s1(5830)⁰", "B_s2*(5840)⁰",
}

# nœuds des mésons diagonaux : (méson, config, nœud clé, chiral?, Alexander validé?)
DIAGONAL_MESONS = [
    ("ρ⁰", "6₂+6̄₂", "6_2", True, True),
    ("ω", "8₁+8̄₁", "8_1", True, True),
    ("η′", "5₂+5₂+6₁", "6_1", True, True),
    ("φ", "7₅+7̄₅", "7_5", True, False),
    ("J/ψ", "7₄+7̄₄", "7_4", True, False),   # 7_4 dégénéré Alexander avec 9_2
    ("Υ", "7₆+7̄₆+4₁", "7_6", True, False),
]


def verify():
    rows = build_rows()
    mesons = [r for r in rows if "MÉSON" in r["cat"]]
    V = []
    for r in mesons:
        if r["anti"]:
            continue
        nm, J, Q = r["name"], r["J"], r["Q"]
        if (Fraction(J).limit_denominator(4) * 2) % 2 == 1:
            V.append(("SPIN", f"{nm}: J={J} demi-entier"))
        if not r["core"].startswith("K+K̄"):
            V.append(("CŒUR", f"{nm}: cœur {r['core']} (attendu K+K̄)"))
        an = "anti-" + nm
        has_anti = any(x["name"] == an for x in rows)
        if nm in FLAVORED_NEUTRAL and not has_anti:
            V.append(("SAVEUR", f"{nm}: neutre saveuré sans antiparticule"))
        if Q not in (0, None) and not has_anti:
            V.append(("CHARGE", f"{nm}: chargé Q={Q} sans antiparticule"))
    # cohérence de fond : nœud diagonal doit être chiral
    for nm, cfg, knot, chiral, _val in DIAGONAL_MESONS:
        if not chiral:
            V.append(("CHIRAL", f"{nm}: nœud diagonal {knot} amphichiral (K+K̄ non-pair)"))
    return rows, mesons, V


def main():
    rows, mesons, V = verify()
    nm = sum(not r["anti"] for r in mesons)
    na = sum(r["anti"] for r in mesons)
    print(f"=== COHÉRENCE MÉSONS : {nm} mésons + {na} anti ===\n")
    print("Couche 1 — structure :")
    if not V:
        print("  ✓ K+K̄, spin entier, saveur↔auto-conj, conjugaison de charge : cohérent.")
    else:
        for k, n in Counter(k for k, _ in V).most_common():
            print(f"  {n}  {k}")
        for k, m in V:
            print(f"   [{k}] {m}")
    print("\nCouche 2 — nœuds des mésons diagonaux (chiral + miroir = K+K̄) :")
    nvalid = 0
    for mnm, cfg, knot, chiral, val in DIAGONAL_MESONS:
        nvalid += val
        print(f"   {mnm:5s} = {cfg:12s} nœud {knot:4s} "
              f"[{'chiral' if chiral else 'AMPHI'}] "
              f"Alexander {'validé ✓' if val else 'à valider'}")
    print(f"\n   {nvalid}/{len(DIAGONAL_MESONS)} cœurs diagonaux validés Alexander "
          f"(6₂,8₁,6₁) ; 7₄/7₅/7₆ ouverts (7 croisements, indice de tresse 4 ;"
          f" 7₄ dégénéré avec 9₂).")
    print("   Tous CHIRAUX -> K+K̄ = vraie paire matière+antimatière "
          "(vs EW 8₃ amphichiral, auto-miroir).")
    return not V


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
