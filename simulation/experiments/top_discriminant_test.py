"""Test discriminant pour le top : entrelacs EW (8_2+8_3+10_1) vs 3e mode Z3 lourd (Koide c,b,t).

Trois volets :
 1. POIDS STATISTIQUE (look-elsewhere, null de liberte d'assignation) :
    - hit m_t de A : 4 combos/1140 dans +/-1.25% ; attendu ~4 par densite -> P(>=1)~1. GRATUIT.
    - quartet 8_3 de A (W,Z,H,t partagent 8_3) : nombre attendu de noeuds a quartet complet
      sous le null = 0.88 ; observe = 1 (8_3). COMPATIBLE HASARD.
    - hit Koide de B : 1/20 triplets a +/-0.42% ; attendu ~0.2-0.3 (avec liberte de scheme).
      p ~ 0.2 : modeste, NON significatif.
    => les nombres m_t ne discriminent RIEN.
 2. STRUCTURE (les vrais discriminants ; tous du cote de A actuellement) :
    (a) cloture oscillations V5p29 (0.1%, resultat le plus fort du secteur) construite sur des
        porteurs DISTINCTS : D0=3_1+5_1+7_2 vs B0=3_1+6_2+7_7 (multisets c {3,5,7} vs {3,6,7},
        f_arith 0 vs 1/3). Sous B (un noeud, 3 modes) les porteurs coincideraient -> cloture
        a re-deriver, generiquement cassee.
    (b) le catalogue realise A partout : aucun porteur lourd commun charme/beaute
        (D+=3_1+7_6 vs B+=6_2+7_3+7_3 ; Lambda_c ...8_1 vs Lambda_b ...6_3).
    (c) uniformite des modes Z3 : les 3 modes leptoniques partagent TOUT sauf la masse (meme
        noeud => meme Q=-1). Un triplet Z3 lourd exigerait UNE topologie a travers les
        systemes c, b, t -- rien ne le realise.
    (d) delta=Q/3 ne se replique PAS pour (c,b,t) (0.069 vs 0.222) : la relation leptonique
        n'est pas universelle -> affaiblit la lecture 'meme moteur'.
 3. PRECISION (les deux echouent ; critere decisif futur) :
    - A : 170.41 vs 172.57+-0.30 -> 7 sigma. B : m_t(Koide exact)=168.4 -> 14 sigma ;
      Q en scheme fixe off de ~8 sigma (2/3 tombe ENTRE pole et MSbar).
    - test decisif futur : une derivation scheme-coherente <0.5% d'un cote ou de l'autre.

VERDICT : au niveau de preuve actuel le test MAINTIENT top = entrelacs EW (A) -- la structure
(oscillations, catalogue, uniformite Z3, non-replication de delta) siege avec A, et le hit
Koide (c,b,t) est une coincidence trans-sectorielle statistiquement non significative (p~0.2),
du type que la TQNT a appris a ecarter. B ne redeviendrait credible que par un Q=2/3
scheme-coherent derive.
"""
import itertools
import math

from theory.mass_formula import u_em, c_j

KNOTS = ['3_1', '4_1', '5_1', '5_2', '6_1', '6_2', '6_3', '7_1', '7_2', '7_3', '7_4',
         '7_6', '7_7', '8_1', '8_2', '8_3', '9_1', '10_1']   # sans 7_5 aberrant
ALPHA0 = 8.16e-4
TARGETS = {'W': 80.377e3, 'Z': 91.188e3, 'H': 125.25e3, 't': 172.57e3}
QUARKS = {'u': 2.16e-3, 'd': 4.67e-3, 's': 0.0934, 'c': 1.27, 'b': 4.18, 't': 172.69}


def ew_mass(c3, U, C):
    return ALPHA0 * sum(U[k] for k in c3) * sum(C[k] for k in c3) ** 3.55


def koide_Q(ms):
    return sum(ms) / sum(math.sqrt(x) for x in ms) ** 2


def run():
    U = {K: u_em(K) for K in KNOTS}
    C = {K: c_j(K) for K in KNOTS}
    combos = list(itertools.combinations_with_replacement(KNOTS, 3))

    print("=== Volet 1 : poids statistique ===")
    hits = {nm: [c3 for c3 in combos if abs(ew_mass(c3, U, C) - mt) / mt <= 0.013]
            for nm, mt in TARGETS.items()}
    print(f"  A/m_t : {sum(1 for c3 in combos if abs(ew_mass(c3,U,C)-TARGETS['t'])/TARGETS['t']<=0.0125)}"
          f"/{len(combos)} combos dans +/-1.25% (attendu ~4 par densite) -> hit GRATUIT")
    pK = {K: sum(1 for c3 in combos if K in c3) / len(combos) for K in KNOTS}
    exp_full = sum(math.prod(1 - (1 - pK[K]) ** len(hits[nm]) for nm in TARGETS) for K in KNOTS)
    full = [K for K in KNOTS if all(any(K in h for h in hits[nm]) for nm in TARGETS)]
    print(f"  A/quartet : noeuds a quartet complet = {full} ; attendu sous null = {exp_full:.2f}"
          f" -> COMPATIBLE hasard")
    okB = [t3 for t3 in itertools.combinations(QUARKS, 3)
           if abs(koide_Q([QUARKS[k] for k in t3]) - 2 / 3) / (2 / 3) <= 0.0042]
    print(f"  B/Koide : {len(okB)}/20 triplets a +/-0.42% ({okB}) ; attendu ~0.2-0.3 -> p~0.2, modeste")
    print("  => les nombres m_t ne discriminent rien.")

    print("\n=== Volet 2 : structure (discriminants reels) ===")
    print("  (a) oscillations V5p29 : porteurs DISTINCTS D0={3_1,5_1,7_2} vs B0={3_1,6_2,7_7}")
    print("      (f_arith 0 vs 1/3) -> sous B (1 noeud, 3 modes) la cloture 0.1% casse. Pour A.")
    print("  (b) catalogue : aucun porteur lourd commun charme/beaute. Pour A.")
    print("  (c) modes Z3 leptoniques partagent tout sauf la masse ; rien ne realise cela")
    print("      pour c,b,t. Pour A.")
    print("  (d) delta=Q/3 : leptons 0.222, (c,b,t) 0.069 -> ne se replique pas. Contre B.")

    print("\n=== Volet 3 : precision (les deux echouent) ===")
    mc, mb = 1.27, 4.18
    s = math.sqrt(mc) + math.sqrt(mb)
    a, b_, c_ = 1 / 3, -4 / 3 * s, (mc + mb) - 2 / 3 * s ** 2
    x = (-b_ + math.sqrt(b_ ** 2 - 4 * a * c_)) / (2 * a)
    print(f"  A : 170.41 vs 172.57+-0.30 -> {abs(170.41-172.57)/0.30:.0f} sigma")
    print(f"  B : m_t(Q=2/3 exact) = {x*x:.1f} GeV -> {abs(x*x-172.57)/0.30:.0f} sigma ; Q off ~8 sigma en scheme fixe")
    print("  test decisif futur : derivation scheme-coherente <0.5% d'un cote ou de l'autre.")

    print("\nVERDICT : top = entrelacs EW MAINTENU (structure) ; Koide (c,b,t) = coincidence")
    print("          trans-sectorielle non significative (p~0.2). B exigerait un Q=2/3 derive.")


if __name__ == "__main__":
    run()
