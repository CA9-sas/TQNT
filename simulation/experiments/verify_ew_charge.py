r"""
Vérification de la charge Q = Σχ + Lk des Z et top réassignés (ew_reassignment.py).

Loi de charge (V5.tex §charge, théorème « Q_phys ∈ ℤ ») : Q = Σχ + Lk, avec
χ = σ·b entier — torique (b=2) → ±2 ; amphichiral → 0 ; 7_2 → 0 (σ=0, établi par
le nucléon 3_1+7_2+7_2, Σχ=-2) ; hyperbolique b=3 → ±1.
(NB : il existe AUSSI une charge par nœud FRACTIONNAIRE Q(K)=span de Jones —
3_1=5/3, 10_1=3/5, 6_2=2/3… — schéma antérieur, distinct ; on ne l'utilise PAS ici,
l'utilisateur demande Q=Σχ+Lk.)

RÉSULTAT.

Z⁰ (Q = 0) — le nouveau est STRICTEMENT MEILLEUR :
    Z ancien   7_2+8_3+9_1 : Σχ = +2 (le 9_1 torique) → Lk = -2 pour Q=0.
    Z nouveau  7_2+8_3+6_3 : Σχ =  0 (7_2:0, 8_3:0 amphi, 6_3:0 amphi) → Lk = 0.
  Σχ=0 EXACT (règle, sans ambiguïté de signe) : l'état neutre le plus PROPRE
  possible — aucun déséquilibre de χ, aucun enlacement requis. Cohérent avec
  Z⁰ neutre ET auto-conjugué (deux amphichiraux). Meilleur que l'ancien (Lk=-2).

top (Q_quark = +2/3) — problème PRÉEXISTANT, ni réglé ni aggravé :
    top ancien   8_2+8_3+10_1 : Σχ = 0  → Lk = +2/3 (NON entier ✗).
    top nouveau  8_3+10_1+9_1 : Σχ = +1 → Lk = -1/3 (NON entier ✗).
  Q=Σχ+Lk est ENTIER par théorème pour tout entrelacs 3-nœuds. Le +2/3 du quark
  top est FRACTIONNAIRE → hors d'atteinte de cette loi, pour l'ANCIEN comme pour
  le NOUVEAU. La réassignation ne change donc RIEN à cette tension. Sous la loi
  entière, le nouveau top porte naturellement Q=+1 (Lk=0) ou Q=0 (Lk=-1).
  Le +2/3 relève de la charge de QUARK (saveur), que la loi d'entrelacs entière
  ne reproduit pas — cohérent avec Règle 0 (saveur/charge de quark = comptabilité
  SM, [[flavor-is-SM-artifact]]). Note : les signes de χ(9_1)=±2, χ(10_1)=±1
  sont mesurés sur des courbes fausses (incertains), mais Σχ reste ENTIER dans
  tous les cas → la conclusion (≠ 2/3) est robuste.

BILAN : la réassignation est CHARGE-COHÉRENTE. Z passe proprement (et mieux) ;
top ne dégrade pas la situation de charge (le +2/3 était déjà hors-loi).
"""
from __future__ import annotations
from theory.jones import braid_index

AMPHI = {"4_1", "6_3", "8_3"}
TORUS = {"3_1", "5_1", "7_1", "9_1"}   # T(2,q), b=2, |χ|=2


def chi(k, sign=+1):
    """χ = σ·b (règle discrète V5.tex). sign = chiralité (torique/hyperb.)."""
    if k in AMPHI:
        return 0
    if k == "7_2":
        return 0
    if k in TORUS:
        return sign * 2
    return sign * 1        # hyperbolique b=3 → ±1


def report(label, comp, chis, q_target):
    Schi = sum(chis)
    lk = q_target - Schi
    ok = abs(lk - round(lk)) < 1e-9
    terms = ", ".join(f"{k}:{c:+d}" for k, c in zip(comp, chis))
    print(f"  {label:16s} {'+'.join(comp):16s} Σχ={Schi:+d} ({terms})")
    print(f"                   Q={q_target:+.3f} → Lk={lk:+.3f} "
          f"{'ENTIER ✓' if ok else 'NON entier ✗'}")


def main():
    print("Loi : Q = Σχ + Lk  (χ = σ·b entier)\n")
    print("Z⁰ (Q = 0) :")
    report("Z ancien", ["7_2", "8_3", "9_1"], [0, 0, +2], 0)
    report("Z NOUVEAU", ["7_2", "8_3", "6_3"], [0, 0, 0], 0)
    print("\ntop (Q_quark = +2/3, fractionnaire) :")
    report("top ancien", ["8_2", "8_3", "10_1"], [+1, 0, -1], 2 / 3)
    report("top NOUVEAU", ["8_3", "10_1", "9_1"], [0, -1, +2], 2 / 3)
    print("\ntop sous la loi ENTIÈRE (états naturels du nouveau, Σχ=+1) :")
    report("top NOUVEAU", ["8_3", "10_1", "9_1"], [0, -1, +2], +1)   # Lk=0
    report("top NOUVEAU", ["8_3", "10_1", "9_1"], [0, -1, +2], 0)    # Lk=-1
    print("\n→ Z : neutre à Lk=0 (le plus propre). top : entier par théorème, "
          "+2/3 hors-loi (idem ancien).")


if __name__ == "__main__":
    main()
