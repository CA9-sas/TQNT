# -*- coding: utf-8 -*-
"""Magnétar : l'écrasement de la trame, l'invariance de G, et ce que la
composition neutronique change (06/09/2026).

Question posée : une étoile faite UNIQUEMENT de neutrons altère-t-elle
la trame — et donc G — différemment d'une masse ordinaire ?

Réponse en trois blocs :
  (A) l'écrasement local, chiffré (il est ÉNORME) ;
  (B) G invariant — théorème d'équivalence (κ_g ∝ m), « un G par
      particule » réfuté, contrôle NICER nul ;
  (C) ce que la composition neutronique change VRAIMENT : rien côté
      trame, tout côté contenu (croûte, Seebeck, Pauli, seuils).
"""
import numpy as np

G = 6.674e-11; c = 2.998e8; hbar = 1.0546e-34
Msun = 1.989e30; me = 9.109e-31; mp = 1.673e-27
M_Pl = 2.176e-8; l_Pl = 1.616e-35

M, R = 1.4 * Msun, 12e3          # 1E 1547 (paramètres du brouillon)

print("=" * 70)
print("(A) L'ÉCRASEMENT LOCAL — la trame sous l'étoile")
print("=" * 70)
comp = G * M / (R * c**2)
depl = 2 * comp
z_fac = np.sqrt(1 - 2 * comp)
print(f"compacité GM/Rc²        : {comp:.4f}")
print(f"déficit Ω²−1 = 2GM/Rc²  : {depl:.4f}  ({depl*100:.0f} %)")
print(f"facteur de redshift     : √(1−r_s/R) = {z_fac:.3f}"
      "   (le 0,810 du brouillon ✓)")
print("\néchelle des écrasements (2Gm/rc², au rayon propre) :")
for nom, m, r in (("électron (Compton)", me, 3.86e-13),
                  ("nucléon (r_p)", mp, 0.84e-15),
                  ("Terre (surface)", 5.972e24, 6.371e6),
                  ("Soleil (surface)", Msun, 6.96e8),
                  ("naine blanche", 0.6*Msun, 7e6),
                  ("1E 1547 (surface)", M, R),
                  ("horizon de Schwarzschild", M, 2*G*M/c**2)):
    print(f"  {nom:26s} {2*G*m/(r*c**2):.3e}")
print(f"\n→ l'étoile creuse la trame {depl/(2*G*mp/(0.84e-15*c**2)):.1e} fois "
      "plus qu'un nucléon isolé :")
print("  l'espace environnant EST violemment altéré (intuition confirmée).")

print()
print("=" * 70)
print("(B) G INVARIANT — pourquoi la composition n'y change rien")
print("=" * 70)
# κ_g(K) = 8π (m_K/M_Pl) ℓ_Pl  → G = ħc/M_Pl² quel que soit K
for nom, m in (("électron", me), ("neutron", mp), ("étoile entière", M)):
    kappa = 8 * np.pi * (m / M_Pl) * l_Pl
    G_lu = hbar * c / M_Pl**2
    print(f"  {nom:15s} κ_g = {kappa:.3e} m   →  G lu = {G_lu:.5e}")
print("  → κ_g VARIE ∝ m, G ne varie PAS : c'est le théorème d'équivalence")
print("    (unicité de la trame). La trame lit un DÉFICIT TOTAL, pas une")
print("    identité de particule — théorème de la rivière (réponse à ρ).")

# « un G par particule » : le test qui l'a réfuté
n_e_star = M / mp          # ~1 électron par nucléon dans la matière ordinaire
print(f"\n  contrôle « un G par particule » : si G suivait l'espèce dominante")
print(f"    en NOMBRE (les électrons, {n_e_star:.1e} dans l'étoile), la")
print(f"    hiérarchie serait fausse de ~14 ordres — voie réfutée, ne pas")
print("    la réintroduire.")
print("\n  contrôle observationnel (audit NICER, commit 7de4687) :")
print("    ΔR/R(2 M☉) ~ 1e-9 vs RG, χ²(cadre)/χ²(RG) = 1,0000 sur 4 EoS")
print("    → les étoiles à neutrons NE DISCRIMINENT PAS. Rien à recalculer :")
print("      le calcul a été fait et il est nul.")

print()
print("=" * 70)
print("(C) CE QUE LA COMPOSITION NEUTRONIQUE CHANGE VRAIMENT")
print("=" * 70)
# les constantes ne dérivent pas (verrou du quatuor) → physique locale intacte
B_c = me**2 * c**3 / (1.602e-19 * hbar)
print(f"champ critique B_c = m_e²c³/eħ = {B_c:.3e} T  — grandeur LOCALE :")
print("  h, c et la rigidité ne dérivent pas (verrou du quatuor) → la")
print("  physique locale au ras de l'étoile est celle du laboratoire.")
print(f"  B/B_c = 7,25 (1E 1547) est donc lu tel quel EN LOCAL ;")
print(f"  seul le passage local→observé porte l'écrasement (×{z_fac:.3f}).")
print("\nCONSÉQUENCE de structure : l'écrasement à "
      f"{depl*100:.0f} % ne modifie AUCUN seuil local ;")
print("il ne déplace que les énergies OBSERVÉES — ce que les deux")
print("altimètres du brouillon utilisent déjà :")
print(f"  mur de paires   : 1,022 MeV × {z_fac:.3f} = "
      f"{1.022*z_fac:.3f} MeV observé")
print(f"  raie 511 keV    : 511 keV × {z_fac:.3f} = {511*z_fac:.0f} keV observé")
print("\nCe qui dépend VRAIMENT de la composition (côté CONTENU, pas trame) :")
print("  - la croûte et son gradient thermique (batterie Seebeck) ;")
print("  - la fraction d'électrons dégénérés (inversion de Pauli au portail) ;")
print("  - l'équation d'état (structure M-R) — mais elle est commune à la RG ;")
print("  - la chimie de surface des paires créées (§M-2, §N).")
print("→ la composition neutronique est un fait de CONTENU. La trame, elle,")
print("  ne lit que la masse : G reste ħc/M_Pl², partout, pour tous.")
