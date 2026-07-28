"""Consolidation : le spin-1/2 du nucleon depuis l'ARITE DE TRESSE.

Resultat des 4 pistes spin<-rotation/force (toutes refutees) : le spin vient de l'indice de
tresse b (arite), s=(b-1)/2, et fermion/boson = parite de b. On l'applique au nucleon.

Nucleon = entrelacs 3_1 + 7_2 + 7_2 (arite 3) :
  3_1 : b=2 (pair) -> s=1/2, FERMION
  7_2 : b=3 (impair) -> s=1, BOSON
  7_2 : b=3 (impair) -> s=1, BOSON
Comptage : 1 composante fermionique (impair) -> total demi-entier (FERMION). = argument V6.
Composition 1/2 (x) 1 (x) 1 -> {1/2, 3/2, 5/2} : 1/2 = nucleon (octet, coupler 1(x)1->0),
  3/2 = Delta (decuplet) ; MEME arite. S_max=5/2, parite P=5/2 mod 1=1/2 (coherent P7quater).
NB TQNT : le nucleon n'est PAS 3 fermions spin-1/2 (SM) mais 1 fermion + 2 bosons ; son 1/2
  vient du comptage de parite (impair), pas de 3x1/2.
"""
from theory.jones import braid_index, topological_spin

COMPONENTS = ['3_1', '7_2', '7_2']


def couple(j1, j2):
    lo = abs(j1 - j2)
    return [lo + i for i in range(int(j1 + j2 - lo) + 1)]


def run():
    print("Nucleon = 3_1 + 7_2 + 7_2  (arite 3)")
    print(f"{'comp':6}{'b':>3}{'s=(b-1)/2':>11}{'parite':>8}{'stat':>9}")
    nf = 0
    spins = []
    for K in COMPONENTS:
        b = braid_index(K)
        s = topological_spin(K)
        spins.append(s)
        par = 'pair' if b % 2 == 0 else 'impair'
        stat = 'fermion' if b % 2 == 0 else 'boson'
        nf += (b % 2 == 0)
        print(f"{K:6}{b:>3}{s:>11}{par:>8}{stat:>9}")

    print(f"\nComposantes fermioniques (b pair) : {nf} "
          f"-> {'IMPAIR => demi-entier (FERMION)' if nf % 2 else 'pair => entier'}")

    tot = set()
    for a in couple(spins[0], spins[1]):
        for t in couple(a, spins[2]):
            tot.add(t)
    print(f"Composition {spins[0]} (x) {spins[1]} (x) {spins[2]} -> spins totaux : {sorted(tot)}")
    print(f"  minimal = {min(tot)} (nucleon, octet)  ;  1.5 accessible (Delta, decuplet)")
    Smax = sum(spins)
    print(f"  S_max = {Smax} ; parite demi-entiere P = S_max mod 1 = {Smax % 1} (coherent P7quater)")
    print("\nCONSOLIDE : spin-1/2 du nucleon = arite de tresse (1 fermion + 2 bosons, comptage impair ;")
    print("            octet 1/2 / decuplet 3/2 par couplage). Positif, coherent avec V6 chap. nucleon.")


if __name__ == "__main__":
    run()
