r"""
Spin-statistique des nœuds toriques T(p,q) — RÉFUTE baryon = T(3,q) unique.

Demande utilisateur : trancher le spin d'un T(3,q) fermé (le verrou de la piste
baryon = T(3,q)).

THÉORÈME (spin-statistique du ruban framé, belt trick) : un nœud FRAMÉ de
self-linking (framing) n a spin n/2 → statistique (−1)^n (n impair → fermion,
n pair → boson). Pour un nœud torique POSITIF T(p,q), le diagramme standard a
(p−1)q croisements de MÊME signe → writhe = self-linking = (p−1)q. Donc :
    statistique de T(p,q) = (−1)^{(p−1)q}.

RÉSULTAT (règle générale = parité de (p−1)q) :
    p PAIR (T(2,q), T(4,q)) : (p−1)=impair → parité=q → FERMION (q impair, nœuds).
    p IMPAIR (T(3,q), T(5,q)) : (p−1)=pair → (p−1)q PAIR → BOSON toujours.
  → T(2,q) = FERMIONS (leptons ✓) ; T(3,q) = BOSONS ; T(4,q) = fermions.

VERDICT : un T(3,q) UNIQUE est un BOSON, PAS un baryon (fermion). La piste
baryon = T(3,q) est RÉFUTÉE sur la spin-statistique. Le baryon reste l'ENTRELACS
3_1+7_2+7_2 : son 3_1 = T(2,3) est un FERMION (p=2) et fournit la statistique
fermionique correcte (règle du cadre : fermion ssi nb de composantes b-paire impair
→ le 3_1 b=2 est la composante fermionique). Un T(3,q) nu n'a pas ce 3_1 → boson.

POURQUOI '3 brins = 3 fermions' ÉCHOUE : une fois la tresse (σ_1σ_2)^q FERMÉE
(gcd=1), les 3 brins forment UNE seule boucle connectée. La statistique est celle
d'UN ruban de framing (p−1)q=2q PAIR → boson. Les 3 brins ne sont PAS 3 fermions
indépendants : c'est un objet unique.

CE QUI SURVIT de l'exploration T(3,q) ([[mass-vortex-reframe]]) :
  - le doublet p/n par FRAMING (Lk) — mais sur l'ENTRELACS (proton Lk=+3, neutron
    Lk=+2), pas sur un T(3,q) ;
  - la RÈGLE spin-statistique topologique (p pair fermion / p impair boson), qui
    VALIDE que les leptons sont des T(2,q) fermions.
CE QUI TOMBE : baryon = T(3,q) unique (les images p/n en T(3,4) étaient fausses,
T(3,4) est un boson). La bonne assignation du nucléon = entrelacs 3_1+7_2+7_2.
"""
from __future__ import annotations
from math import gcd


def statistics(p, q):
    if gcd(p, q) != 1:
        return None  # entrelacs, pas un nœud
    n = (p - 1) * q  # self-linking = writhe du diagramme torique standard
    return n, "fermion" if n % 2 else "boson"


def main():
    print("statistique de T(p,q) = (−1)^{(p−1)q}  (spin-statistique du ruban)\n")
    print(f"{'T(p,q)':9s} {'(p−1)q':>7s} {'statistique':>12s}")
    for p in (2, 3, 4, 5):
        for q in range(2, 9):
            s = statistics(p, q)
            if s:
                print(f"T({p},{q})     {s[0]:>7d} {s[1]:>12s}")
        print()
    print("Règle : p PAIR → fermion (T(2,q) leptons, T(4,q)) ; p IMPAIR → boson (T(3,q)).")
    print("→ T(3,q) unique = BOSON : baryon = T(3,q) RÉFUTÉ. Baryon = entrelacs 3_1+7_2+7_2")
    print("  (le 3_1=T(2,3) fermion donne la statistique correcte).")


if __name__ == "__main__":
    main()
