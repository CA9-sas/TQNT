# -*- coding: utf-8 -*-
"""A-9 — Trèfles qui ne sont pas dans le même plan (17/09/2026).

Suite de A-6/A-7/A-8. Incliner un trèfle par rapport à l'autre ne crée pas
de nouvelle variable : toute configuration non coplanaire relève de l'un
de quatre cas.
  (i)   ROTATION rigide      → c'est l'état de spin (A-7/A-8) ;
  (ii)  RÉFLEXION            → le miroir 3̄₁ : autre charge, superselection ;
  (iii) DÉFORMATION à topologie fixe (Wr ↔ Tw, Lk constant) → modes
        internes à l'échelle du MeV, gelés ;
  (iv)  CHANGEMENT de topologie (Lk, type de nœud) → autre charge, interdit.
Ce que l'inclinaison CHANGE réellement : l'anisotropie des couplages
(dipolaire en 1−3cos²θ, angle magique) et l'adressage spectral (projection
du champ sur l'axe). Exemple réel : les quatre orientations NV du diamant
(axes <111>, 109,47° entre eux).
Entrées TQNT (V7) : spin = arité de tresse, théorème de lecture du spin
(taux ω caché : seule la parité 4π et j sont visibles) ; mode torsionnel
du trèfle ≈ 0,46–0,52 MeV (chaîne m_e) ; Lk = Tw + Wr.
Entrées de littérature : D(NV) = 2,87 GHz.
"""
import numpy as np

h = 6.62607015e-34
kB = 1.380649e-23
muB = 9.2740100783e-24
g_e = 2.00231930436
eV = 1.602176634e-19
mu0_4pi = 1e-7

kT300 = kB * 300 / eV
gamma = g_e * muB / h                        # Hz/T
C_dip = mu0_4pi * (g_e * muB)**2 / h         # Hz·m³

print("=" * 72)
print("(i) ROTATION : l'orientation d'un trèfle EST son spin")
print("=" * 72)
print("Théorème de lecture du spin : le taux de rotation est caché ; une"
      "\n  orientation quelconque de l'axe se décompose sur |↑⟩,|↓⟩ (spin ½)."
      "\n  Deux trèfles d'axes différents = deux spins non parallèles = un état"
      "\n  à deux spins ordinaire (A-8), plus leur moment orbital relatif (standard) :"
      "\n  aucune variable supplémentaire.")

print("\n" + "=" * 72)
print("(ii)/(iv) RÉFLEXION ou CHANGEMENT DE TOPOLOGIE : interdits")
print("=" * 72)
print("Retourner un trèfle par réflexion donne 3̄₁ (charge opposée) ; changer"
      "\n  Lk change Q = Σχ + Lk : superselection dans les deux cas.")

print("\n" + "=" * 72)
print("(iii) DÉFORMATION À TOPOLOGIE FIXE (Wr ↔ Tw, Lk constant) : gelée")
print("=" * 72)
for E_MeV in (0.46, 0.52):
    print(f"  mode interne {E_MeV:.2f} MeV : E/kT(300 K) = {E_MeV*1e9/(kT300*1e3):.1e}"
          f" → population thermique exp(−{E_MeV*1e9/(kT300*1e3):.1e}) = 0")
f_mode = 0.49e6 * eV / h
print(f"  fréquence ≈ {f_mode:.1e} Hz : ni excitée par la chaleur, ni pilotable."
      "\n  Une configuration inclinée « figée » n'existe pas : le trèfle relaxe"
      "\n  vers son plongement idéal, et l'écart se paie au MeV.")

print("\n" + "=" * 72)
print("CE QUE L'INCLINAISON CHANGE : l'anisotropie du couplage")
print("=" * 72)
r = 10e-9
J0 = C_dip / r**3
print(f"Couplage dipolaire à 10 nm, J(θ) = J0·(1−3cos²θ), J0 = {J0/1e3:.1f} kHz"
      "\n  (θ = angle entre l'axe de quantification et la droite joignant les spins)")
for th in (0, 30, 54.7356, 70, 90):
    t = np.radians(th)
    print(f"  θ = {th:7.3f}° : J = {J0*(1-3*np.cos(t)**2)/1e3:+8.2f} kHz")
tm = np.arccos(1 / np.sqrt(3))
slope = 6 * J0 * np.cos(tm) * np.sin(tm)
print(f"Angle magique {np.degrees(tm):.2f}° : couplage nul ; pente {slope/1e3:.0f} kHz/rad"
      f" → 1 mrad d'erreur laisse {slope*1e-3:.0f} Hz")
print("  → l'inclinaison est un INTERRUPTEUR de couplage (géométrique, fixé à la"
      "\n    fabrication), pas une protection de la phase.")

print("\n" + "=" * 72)
print("EXEMPLE RÉEL : les quatre orientations NV du diamant (axes <111>)")
print("=" * 72)
D = 2.87e9
B = 10e-3
cos_other = -1 / 3
print(f"Champ B = {B*1e3:.0f} mT le long d'un axe ; angle entre axes = {np.degrees(np.arccos(cos_other)):.2f}°")
f_al = gamma * B
f_ob = gamma * B * abs(cos_other)
B_perp = B * np.sqrt(1 - cos_other**2)
second = (gamma * B_perp)**2 / D
print(f"  NV aligné     : f± = D ± {f_al/1e6:.1f} MHz")
print(f"  NV incliné ×3 : f± ≈ D ± {f_ob/1e6:.1f} MHz (+ décalage transverse ≈ {second/1e6:.0f} MHz)")
print(f"  Écart d'adressage ≈ {(f_al-f_ob)/1e6:.0f} MHz ≫ largeur de raie (~kHz–MHz) :"
      "\n  les centres non coplanaires se pilotent séparément à 300 K.")
print("  Coût : le champ transverse mélange les états et raccourcit la cohérence"
      "\n  des centres inclinés (mélange ∝ (γB⊥/D)² ≈ "
      f"{(gamma*B_perp/D)**2:.1e}).")

print("\n" + "=" * 72)
print("SYNTHÈSE")
print("=" * 72)
print("Non coplanaire = spins non parallèles (licite, rien de nouveau)"
      "\n  ou miroir/topologie (interdit) ou déformation (gelée au MeV)."
      "\n  Gain réel : couplage réglable par l'angle (nul à 54,74°) et adressage"
      "\n  spectral des orientations. Aucune protection supplémentaire de la phase.")
