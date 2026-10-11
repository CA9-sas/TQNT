# -*- coding: utf-8 -*-
"""A-10 — Le moment angulaire détermine les états (17/09/2026).

Remarque de l'auteur après A-9 : « le moment angulaire permet de déterminer
les états ». Dans le cadre, c'est exactement ce que le théorème de lecture
du spin autorise : le taux de rotation est caché, mais j (et sa projection)
se lit. On chiffre les quatre usages :
  (1) ÉTIQUETER : |J, M⟩ pour deux trèfles (et un trèfle + moment orbital) ;
  (2) PERMETTRE : règle d'univalence — pas de superposition entre moment
      entier et demi-entier (parité du nombre de trèfles, fermions b=2) ;
  (3) LIRE : conservation du moment angulaire dans la lecture (cycle optique
      NV, blocage de Pauli singulet/triplet) ;
  (4) PROTÉGER : un bruit de champ décale l'énergie ∝ M ; la cohérence d'une
      superposition dépend de ΔM → les paires à ΔM = 0 (« horloge ») sont
      insensibles au premier ordre.
Entrées de littérature déclarées : bruit de champ quasi statique δB = 10 µT
(ordre du bain ¹³C naturel, T2* NV ~ µs), écart d'horloge Δ = 5 MHz (ordre
de la contrainte NV à champ nul), contraste de lecture NV ≈ 30 % à 300 K.
"""
import numpy as np

h = 6.62607015e-34
muB = 9.2740100783e-24
g_e = 2.00231930436
gamma = g_e * muB / h                      # Hz/T

print("=" * 72)
print("(1) ÉTIQUETER : deux trèfles de spin ½")
print("=" * 72)
for J, Ms in ((0, (0,)), (1, (-1, 0, 1))):
    for M in Ms:
        nom = "singulet" if J == 0 else "triplet"
        print(f"  |J={J}, M={M:+d}⟩  {nom:<8}  décalage Zeeman {gamma*M/1e9:+.1f} GHz/T")
print("  Avec un moment orbital relatif L : J ∈ |L−S| … L+S — l'orientation"
      "\n  non coplanaire (A-9) se range entièrement dans ces étiquettes.")

print("\n" + "=" * 72)
print("(2) PERMETTRE : univalence")
print("=" * 72)
for n in (1, 2, 3):
    spins = [0.5] * n
    Jmax = sum(spins)
    genre = "demi-entier" if (2 * Jmax) % 2 else "entier"
    print(f"  {n} trèfle(s) : J {genre} (nombre de fermions b=2 {'impair' if n % 2 else 'pair'})")
print("  Superposer J entier et J demi-entier est interdit (une rotation de 2π"
      "\n  les distingue par un signe) : pas de qubit « 1 trèfle + 2 trèfles »,"
      "\n  indépendamment de la charge. À l'intérieur d'un même secteur, tout"
      "\n  J et tout M se superposent librement.")

print("\n" + "=" * 72)
print("(3) LIRE : le moment angulaire trie les états à la lecture")
print("=" * 72)
print("  NV (triplet de deux électrons) : M = 0 fluorescent, M = ±1 passe par"
      "\n    l'état métastable (conversion intersystème) → contraste ≈ 30 % à 300 K,"
      "\n    lecture moyennée (pas de lecture en un coup à 300 K).")
print("  Boîtes quantiques : le singulet J=0 passe dans la même boîte, le triplet"
      "\n    J=1 est bloqué (principe de Pauli) → lecture en un coup, 50–100 mK.")

print("\n" + "=" * 72)
print("(4) PROTÉGER : la cohérence dépend de ΔM")
print("=" * 72)
dB = 10e-6
print(f"Bruit de champ quasi statique δB = {dB*1e6:.0f} µT (gaussien) ;"
      " T2* = √2 / (2π σ_f)")
for dM in (2, 1):
    sf = gamma * dM * dB
    T2 = np.sqrt(2) / (2 * np.pi * sf)
    print(f"  ΔM = {dM} : σ_f = {sf/1e3:6.0f} kHz → T2* = {T2*1e6:6.2f} µs")
T2_1 = np.sqrt(2) / (2 * np.pi * gamma * dB)
Delta = 5e6
eps = gamma * dB
sf0 = np.sqrt(2) * eps**2 / (2 * Delta)     # écart-type d'un décalage quadratique gaussien
T2_0 = np.sqrt(2) / (2 * np.pi * sf0)
print(f"  ΔM = 0 (paire d'horloge, M moyen nul, écart Δ = {Delta/1e6:.0f} MHz) : décalage second ordre"
      f" σ_f ≈ {sf0/1e3:.1f} kHz → T2* ≈ {T2_0*1e6:.0f} µs (×{T2_0/T2_1:.0f})")
for D_MHz in (1, 20):
    sf = np.sqrt(2) * eps**2 / (2 * D_MHz * 1e6)
    print(f"      avec Δ = {D_MHz:>2} MHz : T2* ≈ {np.sqrt(2)/(2*np.pi*sf)*1e6:.0f} µs")
print("  Singulet/triplet M=0 : même règle, avec le GRADIENT de champ à la place"
      "\n    du champ (A-8 : ×1/√(2(1−c))).")

print("\n" + "=" * 72)
print("SYNTHÈSE")
print("=" * 72)
print("  Le moment angulaire (J, M) étiquette les états, fixe les superpositions"
      "\n  permises (univalence), trie la lecture, et choisit la robustesse :"
      "\n  un qubit à ΔM = 0 gagne un facteur ~5–100 sur T2* face au bruit de champ\n  (selon l'écart Δ de la paire d'horloge)."
      "\n  Il ne crée pas de gap thermique : à 300 K la protection reste l'isolement.")
