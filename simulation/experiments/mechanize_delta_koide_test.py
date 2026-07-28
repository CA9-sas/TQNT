"""Tenter de MECANISER la relation delta_Koide = Q/3 = 2/9.

Reduction : delta=Q/3 <=> delta=A^2/N^2 (tous deux 2/9, exigent A^2=2 = le hint sqrt2).
Geometrie : le vecteur sqrt(m) est a 45 deg EXACT de la democratie (1,1,1) [= condition Koide,
cos^2=1/2, le sqrt2] ; delta est l'angle AZIMUTAL autour de cet axe -- un MODULUS.

Mecaniser delta = fixer ce modulus azimutal. Tests :
  - Framing Chern-Simons du trefle (k=135, spin-1/2) : phase 2pi h w ne donne PAS delta
    (w=6 -> 0.206, 7% ; exact exigerait k+2=63.6, pas 137). ECHEC.
  - delta pas exactement 2/9 (0.022%, ~1-3 sigma) -> pas un nombre topologique propre.
CORRECTION TQNT : PAS de quarks u,d fondamentaux. electron, u, d = LE MEME trefle 3_1 (isospin =
  symetrie geometrique). Donc delta n'est PAS un modulus plat a briser de l'exterieur : le trefle
  heberge aussi les modes d'isospin (qui brisent la Z3 pure). delta est INTERNE a 3_1.
RESULTAT : delta pas encore mecanise (framing CS interne echoue a k=135), mais bien pose sur 3_1
  seul. delta=A^2/N^2=2/9 reste un HINT (comme sqrt2).
"""
import math
import numpy as np

ME, MM, MT = 0.51099895, 105.6583755, 1776.86
A, N = math.sqrt(2), 3


def analyse():
    r = np.array([math.sqrt(x) for x in (ME, MM, MT)])
    M = r.sum() / N
    delta = math.acos((r.max() / M - 1) / A)
    Q = (ME + MM + MT) / r.sum() ** 2

    print("=== Reduction ===")
    print(f"  delta={delta:.5f} ; Q/3={Q/3:.5f} ; A^2/N^2={A**2/N**2:.5f} ; 2/9={2/9:.5f}")
    print(f"  delta - 2/9 = {delta-2/9:+.6f} rad ({(delta-2/9)/delta*100:+.3f}%, ~1-3 sigma) -> pas exact")

    demo = np.ones(3) / math.sqrt(3)
    cp = abs(r @ demo / np.linalg.norm(r))
    print("\n=== Geometrie (vecteur sqrt(m)) ===")
    print(f"  polaire vs (1,1,1) : {math.degrees(math.acos(cp)):.2f} deg (Koide=45 EXACT, cos^2={cp**2:.4f})")
    print("  delta = azimut (MODULUS dans le plan perpendiculaire)")

    print("\n=== Mecanismes testes ===")
    k = 135
    h = 0.75 / (k + 2)
    for w in (3, 6, 9):
        print(f"  CS framing w={w}: {2*math.pi*h*w:.4f} rad (delta={delta:.4f}) -> ECHEC")
    keq = 0.75 / (delta / (2 * math.pi) / 3) - 2
    print(f"  pour delta exact via CS (w=3) il faudrait k={keq:.1f} != 135 -> ECHEC")
    print("  TQNT : e, u, d = MEME trefle 3_1 (isospin geometrique) -> delta INTERNE a 3_1,")
    print("         pas un modulus plat a briser de l'exterieur (pas de secteur quarks).")

    print("\nVERDICT : delta pas encore mecanise (interne a 3_1) ; delta=A^2/N^2=2/9 = HINT.")


if __name__ == "__main__":
    analyse()
