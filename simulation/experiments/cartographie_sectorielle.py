"""Refonte de la cartographie d'assignation : masse SECTORIELLE + discriminants geometriques.

Sous m = sqrt(sigma_chi)*S, la masse selectionne le SECTEUR (gabarit), pas le noeud.
Discriminants par etat : gabarit, parite de tresse (spin), Q = Sigma(sigma_i b_i) + Lk
(chi corrige : sigma=0 seulement 4_1/6_3), |Lk|<=4, et regles geometriques :
  - AUTO-CONJUGAISON : etat = sa propre antiparticule => contenu invariant par miroir
    (noeuds topologiquement amphichiraux, ou paires miroir K+Kbar) ET Lk=0 ;
  - paires miroir identifiables (le meme noeud 2x compte ses combos de signes).
Comparaison : ancien chi (sigma=0 pour ~14 noeuds, courbes fausses) vs nouveau.
"""
import sys, os, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from theory.jones import braid_index

MEASURED = ["3_1","4_1","5_1","5_2","6_1","6_2","6_3","7_1","7_2","8_1","8_2","8_3","9_1","10_1"]
B = {k: braid_index(k) for k in MEASURED}
AMPHI_TOPO = {"4_1","6_3","8_3"}
SIG0_NEW = {"4_1","6_3"}                       # geometriquement symetriques
SIG0_OLD = {k for k in MEASURED if k not in ("3_1","5_1","7_1","9_1")}  # ancien : tout sauf toriques
FERMION = {k for k in MEASURED if B[k] % 2 == 0}   # b pair = fermion (spin 1/2 : b=2)


def sign_options(k, scheme):
    z = SIG0_NEW if scheme == "new" else SIG0_OLD
    return (0,) if k in z else (1, -1)


def count(content, Q, scheme, self_conj=False):
    n = 0
    for signs in itertools.product(*[sign_options(k, scheme) for k in content]):
        sc = sum(s * B[k] for s, k in zip(signs, content))
        lk = Q - sc
        if abs(lk) > 4:
            continue
        if self_conj:
            if lk != 0:
                continue
            # invariance miroir : le multiensemble {(K, s)} doit etre stable sous s -> -s
            ms = sorted((k, s) for k, s in zip(content, signs))
            mm = sorted((k, -s) if k not in AMPHI_TOPO else (k, s) for k, s in zip(content, signs))
            # noeud topologiquement amphichiral : K = Kbar (le flip de signe reste le meme noeud)
            mm2 = sorted((k, -s) for k, s in zip(content, signs))
            ok = ms == mm or ms == mm2 or all(k in AMPHI_TOPO for k in content)
            if not ok:
                continue
        n += 1
    return n


def enumerate_contents(template):
    if template == "octet":      # 3_1 + 2 bosons (b impair)
        bos = [k for k in MEASURED if B[k] % 2 == 1]
        return [["3_1", a, b_] for a, b_ in itertools.combinations_with_replacement(bos, 2)]
    if template == "ew":         # triple contenant 8_3
        others = [k for k in MEASURED if k != "3_1"]
        out = []
        for a, b_ in itertools.combinations_with_replacement(others, 2):
            out.append(sorted([a, b_, "8_3"]))
        return [list(x) for x in {tuple(c) for c in out}]
    raise ValueError(template)


def run():
    print("=== Cartographie sectorielle : candidats par etat (ancien chi vs nouveau + regles) ===\n")
    print("--- SECTEUR BARYON OCTET (gabarit 3_1 + 2 bosons ; masse = loi Regge+Z3, ne discrimine pas) ---")
    contents = enumerate_contents("octet")
    for name, Q, sc in (("p (Q=+1)", 1, False), ("n (Q=0)", 0, False),
                        ("Sigma+ (Q=+1)", 1, False), ("Xi- (Q=-1)", -1, False)):
        old = sum(count(c, Q, "old") for c in contents)
        new = sum(count(c, Q, "new") for c in contents)
        print(f"  {name:14}: contenus x combos — ancien {old:4d} ; nouveau {new:4d}")
    print(f"  (contenus de gabarit : {len(contents)})")

    print("\n--- SECTEUR EW (gabarit : triple avec 8_3 ; masse = VEV, ne discrimine pas) ---")
    contents = enumerate_contents("ew")
    for name, Q, sc in (("W+ (Q=+1)", 1, False), ("Z (Q=0, AUTO-CONJ)", 0, True),
                        ("H (Q=0, AUTO-CONJ)", 0, True), ("Z sans auto-conj", 0, False)):
        old = sum(count(c, Q, "old", sc) for c in contents)
        new = sum(count(c, Q, "new", sc) for c in contents)
        print(f"  {name:20}: ancien {old:4d} ; nouveau {new:4d}")
    print(f"  (contenus de gabarit : {len(contents)})")

    print("\n--- AUDIT des assignations canoniques EW sous AUTO-CONJUGAISON ---")
    for name, content in (("Z = 7_2+8_3+9_1", ["7_2","8_3","9_1"]),
                          ("H = 7_1+8_3+10_1", ["7_1","8_3","10_1"])):
        n_new = count(sorted(content), 0, "new", self_conj=True)
        print(f"  {name:18}: candidats auto-conjugues (nouveau chi) = {n_new}"
              f"  {'<-- TENSION : contenu chiral non appariable' if n_new == 0 else ''}")

    print("\n--- MESONS auto-conjugues (pi0, eta : construction K+Kbar, gabarit 2 noeuds) ---")
    pairs = [[k, k] for k in MEASURED]     # K + K-miroir : meme noeud, signes opposes
    old = sum(count(c, 0, "old", True) for c in pairs)
    new = sum(count(c, 0, "new", True) for c in pairs)
    print(f"  pi0/eta (Q=0, auto-conj) : ancien {old} ; nouveau {new} (le gabarit K+Kbar")
    print(f"  est naturellement auto-conjugue : la regle ne coute rien la ou elle doit etre libre)")

    print("\n=== LECTURE ===")
    print("1. Octet : le nouveau chi multiplie les combos de signes (x2-3) — la charge seule")
    print("   discrimine moins ; les coupes reelles sont ailleurs (paires miroir, moments).")
    print("2. EW : l'AUTO-CONJUGAISON est la coupe puissante — elle reduit massivement Z/H")
    print("   et EXPOSE une tension : les assignations canoniques Z et H ne sont PAS")
    print("   auto-conjuguees sous le chi corrige (contenu chiral sans appariement miroir).")
    print("   Trois issues : (i) le Z/H TQNT n'est pas tenu d'etre auto-conjugue (etat +")
    print("   miroir degeneres = le Z observe) ; (ii) reassigner Z/H vers des contenus")
    print("   auto-conjugues ; (iii) le 8_3 brise sauve partiellement (topologiquement")
    print("   auto-conjugue). A trancher — c'est LE point dur expose par la refonte.")


if __name__ == "__main__":
    run()
