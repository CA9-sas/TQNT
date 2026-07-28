"""Cartographie d'assignation avec les discriminants geometriques (writhe ideal).

CORRECTION DE FOND : le classificateur de chiralite sigma (sature de courbure sur
les courbes du catalogue) alimentait chi = sigma*b dans Q = Sigma(chi) + Lk. Or
14/19 courbes etaient topologiquement fausses. Le discriminant PRINCIPIEL est le
writhe ideal (mesure 2026-07-14) : sigma_geom = 0 ssi le noeud est amphichiral ET
geometriquement symetrique (4_1, 6_3) ; sinon +-1 (y compris 8_3, brise).
NB : le SIGNE par noeud est un choix de miroir (par etat) ; l'invariant est la
CLASSE (0 vs +-1) et |Wr|.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import itertools
from theory.jones import braid_index

# |Wr| ideal mesure (campagne 2026-07-14) ; None = non mesure (7_3..7_7)
WR_IDEAL = {"3_1": 3.46, "4_1": 0.10, "5_1": 6.50, "5_2": 4.43, "6_1": 0.95,
            "6_2": 3.28, "6_3": 0.05, "7_1": 9.32, "7_2": 5.51, "7_3": None,
            "7_4": None, "7_5": None, "7_6": None, "7_7": None, "8_1": 1.99,
            "8_2": 6.25, "8_3": 7.24, "9_1": 12.21, "9_2": 6.58, "10_1": 3.14}
AMPHI = {"4_1", "6_3", "8_3"}   # topologiquement amphichiraux


def sigma_class(k):
    w = WR_IDEAL.get(k)
    if w is None:
        return "?"
    return 0 if w < 0.5 else 1   # 1 = chiral geometrique (signe libre par etat)


def run():
    print("=== Classes de chiralite geometrique (writhe ideal) vs ancien classificateur ===")
    print(f"{'noeud':6}{'b':>3}{'|Wr|':>7}  classe_geom   ancien chi (document principal)")
    OLD_CHI = {"3_1": -2, "7_2": 0}   # valeurs canoniques documentees (CLAUDE.md)
    for k in WR_IDEAL:
        try:
            b = braid_index(k)
        except ValueError:
            continue   # 9_2 hors catalogue
        sc = sigma_class(k)
        cls = "sigma=0 (sym)" if sc == 0 else ("sigma=+-1" if sc == 1 else "NON MESURE")
        old = OLD_CHI.get(k, "")
        note = ""
        if k == "7_2":
            note = "  <-- CANON chi=0 : ARTEFACT (courbe fausse) ; vrai 7_2 CHIRAL, chi=+-3"
        if k == "8_3":
            note = "  <-- amphichiral TOPOLOGIQUE mais brise : peut porter sigma=+-1"
        print(f"{k:6}{b:>3}{WR_IDEAL[k] if WR_IDEAL[k] is not None else -1:>7.2f}  {cls:13} {old}{note}")

    print("\n=== Audit des charges canoniques Q = Sigma(chi) + Lk, chi = sigma*b ===")
    print("NUCLEON 3_1+7_2+7_2 (document principal : Sigma_chi = -2, Lk_p = +3, Lk_n = +2)")
    b31, b72 = braid_index("3_1"), braid_index("7_2")
    combos = set()
    for s1 in (1, -1):
        for s2 in (1, -1):
            for s3 in (1, -1):
                combos.add(s1*b31 + s2*b72 + s3*b72)
    print(f"  Sigma_chi possibles (7_2 chiral, signes libres) : {sorted(combos)}")
    print(f"  -2 atteignable ssi les deux 7_2 sont en PAIRE MIROIR (+3-3) et 3_1 en -2 :")
    print(f"    Sigma_chi = -2 + 3 - 3 = -2  ==> le document principal SURVIT avec une PREDICTION nouvelle :")
    print(f"    les deux compagnons du nucleon sont 7_2 et 7_2-miroir (chiralites opposees).")
    print(f"  Alternative sans paire miroir : Sigma_chi in {sorted(c for c in combos if c != -2)} ->")
    print(f"    proton Q=+1 exigerait Lk in {sorted(1-c for c in combos if c != -2)} (|Lk|<=4 : partiellement ok,")
    print(f"    mais Delta(n-p) = 1 unite de Lk preserve dans tous les cas).")

    print("\nEW (tous contiennent 8_3 ; document principal utilisait chi(8_3)=0 via l'ancien sigma=0)")
    for name, content, Q in (("W", ["7_1","7_2","8_3"], 1), ("Z", ["7_2","8_3","9_1"], 0),
                             ("H", ["7_1","8_3","10_1"], 0), ("top", ["8_2","8_3","10_1"], 0)):
        bs = [braid_index(k) for k in content]
        sigs = []
        for signs in itertools.product((1,-1), repeat=3):
            sigs.append(sum(s*b for s, b in zip(signs, bs)))
        possible = sorted(set(sigs))
        lks = sorted(set(Q - s for s in possible if abs(Q - s) <= 4))
        print(f"  {name:4} {'+'.join(content):18} : Sigma_chi possibles {possible} ; "
              f"Lk requis (|Lk|<=4) : {lks if lks else 'AUCUN — probleme !'}")

    print("\n=== Impact sur le comptage de candidats (equation de charge seule) ===")
    print("Regle nouvelle : seuls 4_1 et 6_3 ont chi=0 force ; l'ancien classificateur")
    print("mettait chi=0 sur ~14 noeuds (courbes fausses lues comme amphichirales).")
    knots = [k for k in WR_IDEAL if WR_IDEAL[k] is not None]
    # pour une cible Q donnee et un contenu a 3 noeuds : nb de (signes, Lk) valides
    import math
    def count_solutions(content, Q, chi_zero_set):
        n = 0
        bs = [braid_index(k) for k in content]
        for signs in itertools.product((1, 0, -1), repeat=3):
            ok = True
            for s, k in zip(signs, content):
                if k in chi_zero_set and s != 0: ok = False
                if k not in chi_zero_set and s == 0: ok = False
            if not ok: continue
            sc = sum(s*b for s, b in zip(signs, bs))
            if abs(Q - sc) <= 4:
                n += 1
        return n
    old_zero = {k for k in knots if k not in ("3_1","4_1","5_1","7_1","9_1")}  # ancien : non-toriques lus achiraux
    new_zero = {"4_1", "6_3"}
    sample = [["3_1","7_2","7_2"], ["7_1","7_2","8_3"], ["3_1","6_1","7_2"], ["3_1","6_2","7_2"]]
    print(f"{'contenu':22}{'Q':>3}  {'solutions (ancien chi)':>24}{'(nouveau chi)':>16}")
    for content in sample:
        for Q in (0, 1):
            o = count_solutions(content, Q, old_zero)
            nn = count_solutions(content, Q, new_zero)
            print(f"{'+'.join(content):22}{Q:>3}  {o:>24}{nn:>16}")
    print("\nLecture : le nouveau classificateur MULTIPLIE les combinaisons de signes")
    print("(les noeuds redevenus chiraux ont un signe libre) mais AJOUTE la contrainte")
    print("miroir (paires opposees identifiables) et retire les faux chi=0. Le solde")
    print("net sur la cartographie complete exige la refonte avec masse sectorielle.")


if __name__ == "__main__":
    run()
