r"""
Vérification du SPIN (spin-statistique) des Z, top réassignés — et CORRECTION de top.

Règle TQNT : le spin émerge de l'indice de tresse b. Par nœud s=(b-1)/2, donc
b PAIR → spin demi-entier (FERMION), b IMPAIR → spin entier (BOSON). Pour un
entrelacs : FERMION ssi le nombre de composantes b-pair est IMPAIR (parité de
spin-statistique). Validé sur le nucléon 3_1+7_2+7_2 : b(3_1)=2 (f), b(7_2)=3 (b)
×2 → 1 fermion → FERMION (spin 1/2) ✓.

RÉSULTAT.

Z⁰ (boson, spin 1) — PASSE :
    Z nouveau 7_2+8_3+6_3 : b = 3,4,4 → fermions {8_3,6_3} = 2 (pair) → BOSON ✓.
  (comme l'ancien 7_2+8_3+9_1 : fermions {8_3,9_1}=2 → boson.) Z est validé sur
  les TROIS contraintes : masse (+3,2 %), charge (Σχ=0, Lk=0), spin (boson).

top (fermion, spin 1/2) — le pick masse-seule ÉCHOUE, correction nécessaire :
    top 8_3+10_1+9_1 (masse +2,3 %) : b = 4,3,2 → fermions {8_3,9_1} = 2 (pair)
      → BOSON ✗ (top doit être FERMION). Le 9_1 (b=2) inverse la parité.
  → CORRECTION sous contrainte de spin (fermion) + masse + ordre :
    top 8_3+4_1+10_1 : b = 4,3,3 → 1 fermion {8_3} → FERMION ✓, masse -6,6 %.
    (substitution minimale 8_2→4_1 ; partage 8_3+10_1 avec H.)

TENSION MASSE↔SPIN (honnête) : le meilleur top en MASSE (9_1, +2,3 %) est un
BOSON interdit ; imposer le spin fermionique force 8_3+4_1+10_1 à -6,6 %. Le
spin COÛTE de la précision de masse (RMS EW 4,7 % → 5,6 %). Alternative valide :
8_3+6_3+7_1 (fermion, +7,5 %, garde un torus lourd 7_1).

MULTIPLET EW FINAL (masse + charge + spin cohérents) :
    W   = 7_1+7_2+8_3  (boson,   -5,8 %)   fermions=2
    Z   = 7_2+8_3+6_3  (boson,   +3,2 %)   fermions=2   [réassigné]
    H   = 7_1+8_3+10_1 (boson,   +6,2 %)   fermions=2
    top = 8_3+4_1+10_1 (fermion, -6,6 %)   fermions=1   [réassigné, spin-corrigé]
  RMS EW = 5,6 %, ordre W<Z<H<top ✓, tous les spins corrects.

RÉSERVE : la parité de spin-statistique (boson/fermion) est le contrôle robuste ;
la VALEUR exacte (Z=1 vs H=0, tous deux bosons) demande le couplage J complet
(Regge/moment cinétique), non tranché par la seule parité de tresse.
"""
from __future__ import annotations
from theory.jones import braid_index

TARGET = {"W": "boson", "Z": "boson", "H": "boson", "top": "fermion"}
FINAL = {
    "W": "7_1+7_2+8_3", "Z": "7_2+8_3+6_3",
    "H": "7_1+8_3+10_1", "top": "8_3+4_1+10_1",
}


def parity(cfg):
    comp = cfg.split("+")
    nf = sum(braid_index(k) % 2 == 0 for k in comp)
    return nf, ("fermion" if nf % 2 == 1 else "boson")


def main():
    print("Règle : FERMION ssi nb de composantes b-pair est IMPAIR.\n")
    print("Contrôle nucléon 3_1+7_2+7_2 :", parity("3_1+7_2+7_2"),
          "(attendu fermion, spin 1/2)\n")
    print("Refuté par le spin : top 8_3+10_1+9_1 →", parity("8_3+10_1+9_1"),
          "(BOSON, or top=fermion ✗ — le 9_1 b=2 inverse la parité)\n")
    print("Multiplet EW final :")
    for nm, cfg in FINAL.items():
        nf, par = parity(cfg)
        ok = "✓" if par == TARGET[nm] else "✗"
        print(f"  {nm:4s} {cfg:16s} b-pairs={nf} → {par:8s} (cible {TARGET[nm]}) {ok}")


if __name__ == "__main__":
    main()
