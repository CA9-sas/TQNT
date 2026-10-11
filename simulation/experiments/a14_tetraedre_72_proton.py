# -*- coding: utf-8 -*-
"""A-14 — Tétraèdre de nœuds 7₂ au lieu de trèfles (17/09/2026).

Proposition de l'auteur après A-13 : remplacer le trèfle par le 7₂, « ce qui
crée la stabilité dans le proton ».
Entrées TQNT (V7) : 7₂ d'indice de tresse b = 3 → spin 1 ; 7₂ LIBRE :
m = 605 MeV, τ = 3,6e-10 s (dénouage faible P15, u = 1), production par
paires seulement (libre uniquement si un champ critique sépare la paire) ;
nucléon = 3₁ + 7₂ + 7̄₂, spin ½⊗1⊗1 → octet ½ : le porteur STABLE du 7₂
est le proton (plancher protégé).
Deux versions chiffrées :
  I.  tétraèdre de quatre 7₂ libres (spin 1) ;
  II. tétraèdre de quatre protons — la molécule de méthane CH₄.
Entrées de littérature déclarées : γ_p ; C–H = 1,087 Å ; couplage scalaire
géminé H–H du méthane ≈ −12 Hz (isotopologues, ordre de grandeur) ; états
singulets nucléaires à longue durée de vie (minutes) ; espèces de spin
nucléaire A/T/E du méthane.
"""
import numpy as np
from itertools import combinations

h = 6.62607015e-34
hbar = h / (2 * np.pi)
kB = 1.380649e-23
eV = 1.602176634e-19

print("=" * 72)
print("I. TÉTRAÈDRE DE QUATRE 7₂ LIBRES (spin 1)")
print("=" * 72)
Sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
Sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2)
Sz = np.diag([1.0, 0.0, -1.0])
I3 = np.eye(3)
n = 4


def op(single, k):
    m = np.array([[1.0]])
    for i in range(n):
        m = np.kron(m, single if i == k else I3)
    return m


S = [[op(s, k) for s in (Sx, Sy, Sz)] for k in range(n)]
Stot = [sum(S[k][a] for k in range(n)) for a in range(3)]
S2 = sum(Stot[a] @ Stot[a] for a in range(3))
H = sum(sum(S[i][a] @ S[j][a] for a in range(3)) for i, j in combinations(range(n), 2))
w, v = np.linalg.eigh(H)
s2 = np.real(np.einsum("ij,jk,ki->i", v.conj().T, S2, v))
sing = np.where(np.abs(s2) < 0.1)[0]
print(f"  Espace de spin : 3⁴ = {3**n} états ; singulets (J = 0) : {len(sing)}"
      f" → un qutrit ({np.log2(len(sing)):.2f} qubit) au lieu d'un qubit")
Vs = v[:, sing]
print(f"  Énergies des singulets (unités J) : {np.round(w[sing], 6)} → dégénérés")
rank1 = max(abs(Vs[:, a_].conj() @ S[k][c] @ Vs[:, b_])
            for a_ in range(len(sing)) for b_ in range(len(sing)) for k in range(n) for c in range(3))
Q = [S[k][2] @ S[k][2] - (2 / 3) * np.eye(3**n) for k in range(n)]
rank2 = max(abs(Vs[:, a_].conj() @ Q[k] @ Vs[:, b_])
            for a_ in range(len(sing)) for b_ in range(len(sing)) for k in range(n))
print(f"  champ local (rang 1) entre singulets     : max = {rank1:.1e}")
print(f"  anisotropie locale S_z²−2/3 (rang 2)      : max = {rank2:.1e}"
      "\n  → le spin 1 ajoute un canal de bruit quadrupolaire, et la symétrie"
      "\n    l'élimine aussi au premier ordre (Wigner–Eckart).")

tau = 3.6e-10
T2max = tau / n
print(f"\n  Durée de vie d'un 7₂ libre : τ = {tau:.1e} s")
print(f"  Un qubit à quatre 7₂ se perd dès qu'un seul se dénoue : T2 ≤ τ/4 = {T2max*1e12:.0f} ps")
for Jv in (10e6, 1e9, 1e12):
    print(f"    échange J = {Jv:.0e} Hz : portes possibles avant la perte ≈ {2*Jv*T2max:.1e}")
Jneed = 1000 / (2 * T2max)
print(f"  1 000 portes exigeraient J ≈ {Jneed:.1e} Hz = {h*Jneed/eV*1e3:.0f} meV"
      f" ({h*Jneed/eV/(kB*300/eV):.1f} kT à 300 K)")
print("  Et aucune source : un 7₂ ne naît que par paire, lié aussitôt ; libre"
      "\n  seulement sous champ critique (P22). Verdict I : inutilisable.")

print("\n" + "=" * 72)
print("II. LE PORTEUR STABLE DU 7₂ : LE PROTON — tétraèdre = méthane CH₄")
print("=" * 72)
gp = 2.6752218744e8                     # rad s⁻¹ T⁻¹
ge = 1.76085963e11
print(f"  Spin du nucléon : ½⊗1⊗1 → ½ (octet) ; γ_p/2π = {gp/2/np.pi/1e6:.3f} MHz/T"
      f" ({ge/gp:.0f} fois plus faible que l'électron)")
for B in (1.0, 9.4):
    pol = np.tanh(hbar * gp * B / (2 * kB * 300))
    print(f"  polarisation thermique à 300 K sous {B:>3} T : {pol:.1e}")
rCH = 1.087e-10
rHH = rCH * np.sqrt(8 / 3)
D_HH = 1e-7 * gp**2 * hbar / rHH**3 / (2 * np.pi)
print(f"  distance H–H dans CH₄ : {rHH*1e10:.3f} Å ; couplage dipolaire H–H : {D_HH/1e3:.1f} kHz"
      "\n    (moyenné à zéro par la rotation en phase liquide ou gazeuse)")
Jg = 12.0
print(f"  couplage scalaire géminé ≈ {Jg:.0f} Hz → porte ≈ 1/(2J) = {1/(2*Jg)*1e3:.0f} ms")

print("\n  Symétrie tétraédrique des quatre spins ½ nucléaires (16 états) :")
print("    espèce A : I = 2 (5 états) ; espèce T : I = 1 (9 états) ;"
      "\n    espèce E : I = 0 (2 états) — exactement le code « deux singulets » de A-13.")
for T2_s in (1.0, 60.0, 600.0):
    print(f"    cohérence {T2_s:>5.0f} s → portes à 12 Hz ≈ {2*Jg*T2_s:.0e}")
print("  Les états E sont découplés des radiofréquences par la symétrie : pour les"
      "\n  lire ou les piloter, il faut briser la symétrie (isotopologue CH₂D₂,"
      "\n  déplacement chimique) — et c'est ce même bris qui les fait relaxer.")
print("  Lecture à 300 K : seulement en ensemble (~10²⁰ molécules), polarisation"
      "\n  ~1e-5 sans hyperpolarisation → mémoire et démonstrateur, pas processeur.")

print("\n" + "=" * 72)
print("SYNTHÈSE")
print("=" * 72)
print("  La stabilité du 7₂ dans le proton est celle de sa CHARGE et de sa liaison"
      "\n  (plancher protégé), pas celle d'une phase. Libre, le 7₂ vit 0,36 ns."
      "\n  Dans le proton, il donne un spin ½ nucléaire, faiblement couplé : cohérence"
      "\n  de la seconde à la minute, mais portes en dizaines de ms, lecture en"
      "\n  ensemble seulement. Le tétraèdre de protons existe : c'est le méthane,"
      "\n  et son espèce de spin E est le code de A-13 réalisé dans une molécule.")
