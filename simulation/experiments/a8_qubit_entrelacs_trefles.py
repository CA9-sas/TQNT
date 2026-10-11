# -*- coding: utf-8 -*-
"""A-8 — Qubit sur un entrelacs de trèfles (17/09/2026).

Suite de A-6 (enlacement de fluxoïde) et A-7 (spin du trèfle, éremnion).
Un entrelacs de deux trèfles offre quatre variables candidates :
  (1) l'enlacement entre composantes Lk ;
  (2) le spin collectif singulet/triplet (S, T0) ;
  (3) le condensat d'entrelacs 2×3₁ (paires de Cooper) — la voie A-6 ;
  (4) le trèfle lié à son miroir (e⁻e⁺, positronium).
Entrées TQNT (V7) : Q = Σχ + Lk ; paire de Cooper = entrelacs 2×3₁
(charge 2e, h/2e, auto-piégeage dans le puits χ commun) ; spin = arité de
tresse. Entrées de littérature déclarées : boîtes quantiques singulet-triplet
(T2* ~10 ns GaAs, ~0,4 µs Si/SiGe, 50–100 mK), positronium (τ para 125 ps,
τ ortho 142 ns, hyperfin 203,4 GHz), NV (écart 2,87 GHz).
"""
import numpy as np

h = 6.62607015e-34
kB = 1.380649e-23
muB = 9.2740100783e-24
g_e = 2.00231930436
eV = 1.602176634e-19
mu0_4pi = 1e-7

kT300 = kB * 300 / eV
print("=" * 72)
print("(1) VARIABLE = ENLACEMENT Lk ENTRE LES DEUX TRÈFLES")
print("=" * 72)
print("Q = Σχ + Lk : Lk et Lk+1 portent des charges différentes (comme n et p,"
      "\n  Lk = +2 / +3). Superposition α|Lk⟩+β|Lk+1⟩ = superposition de charges"
      "\n  → INTERDITE par la superselection, comme le qubit chiral.")
beta, cj = 0.210, 3.3                         # MeV ; c_J(3₁)
E_dLk = beta * np.sqrt(cj * cj)
print(f"  Échelle de tension pour ΔLk = ±1 (β·√(c_J c_J), 3₁–3₁) ≈ {E_dLk:.2f} MeV :"
      "\n  la barrière protège la CHARGE de l'entrelacs, pas une phase de qubit.")

print("\n" + "=" * 72)
print("(2) VARIABLE = SPIN COLLECTIF S / T0 (même charge 2e, même Lk)")
print("=" * 72)
print("Licite : S et T0 ont Q, Lk, masse identiques ; porte = échange J (électrique).")
print(f"k_B T à 300 K = {kT300*1e3:.2f} meV")
zeeman_meV_T = g_e * muB / eV * 1e3
B_kT = kT300 * 1e3 / zeeman_meV_T
print(f"Zeeman {zeeman_meV_T:.4f} meV/T → séparer T± de T0 au-delà de kT(300 K)"
      f" exigerait B ≈ {B_kT:.0f} T")
print(f"  à 100 mK : B ≈ {B_kT*0.1/300*1e3:.0f} mT suffit → d'où les 50–100 mK des boîtes quantiques")
for n_kT in (1, 4.6):
    print(f"  Échange J ≥ {n_kT}·kT(300 K) = {n_kT*kT300*1e3:.0f} meV"
          f" ({'1 % de population thermique' if n_kT > 2 else 'égalité'})")
print("  → J de l'ordre de 0,1 eV : recouvrement moléculaire (sub-nm), plus des boîtes"
      "\n    quantiques (J ~ µeV–meV, lit.) ; mais alors les orbitales excitées et le"
      "\n    croisement intersystème entrent dans la fenêtre thermique.")

print("\nSous-espace protégé S/T0 : insensible au bruit de champ COMMUN.")
print("  T2*(S/T0) / T2*(spin seul) = 1/√(2(1−c)), c = corrélation du bruit aux deux sites")
for c in (0.0, 0.5, 0.9, 0.99):
    print(f"    c = {c:4.2f} → gain ×{1/np.sqrt(2*(1-c)):.2f}")
print("  Bain nucléaire (corrélation courte, c≈0) : AUCUN gain, ×0,71.")

print("\nLe centre NV est déjà un système à deux électrons (triplet S=1) :")
C_dip = mu0_4pi * (g_e * muB)**2 / h          # Hz·m³
D_NV = 2.87e9
r_eff = (1.5 * C_dip / D_NV)**(1 / 3)
print(f"  écart dipolaire D = (3/2)·μ0/4π·(gμ_B)²/r³ = 2,87 GHz ⇔ r ≈ {r_eff*1e9:.2f} nm"
      "\n  (liaisons pendantes de la lacune) — lecture TQNT : un entrelacs de deux"
      "\n  trèfles en triplet, le qubit de spin à 300 K de A-7.")

print("\n" + "=" * 72)
print("(3) CONDENSAT D'ENTRELACS 2×3₁ — voie A-6 (fluxonium)")
print("=" * 72)
Delta_GHz = 0.05
print(f"  Gap Δ ≈ {Delta_GHz*1e3:.0f} MHz ; kT/h(300 K) = {kB*300/h/1e9:.0f} GHz"
      f" → rapport {kB*300/h/(Delta_GHz*1e9):.1e} ; condensat détruit bien avant"
      "\n  (aluminium : T_c = 1,2 K). Fonctionnement 10–20 mK, T2 0,1–1,5 ms.")

print("\n" + "=" * 72)
print("(4) TRÈFLE + MIROIR (e⁻e⁺) — positronium para/ortho")
print("=" * 72)
tau_p, tau_o, f_hfs = 125e-12, 142e-9, 203.39e9
T2 = 2 / (1 / tau_p + 1 / tau_o)
print(f"  Hyperfin {f_hfs/1e9:.1f} GHz = {h*f_hfs/eV*1e3:.3f} meV"
      f" ({h*f_hfs/eV/kT300*100:.1f} % de kT à 300 K)")
print(f"  Superposition para/ortho : T2 ≈ 2/(Γp+Γo) = {T2*1e12:.0f} ps"
      f" → {f_hfs*T2:.0f} oscillations avant la perte")
print("  Q total = 0 (miroir) : licite, mais la paire s'annihile — la"
      "\n  cohérence est bornée par l'annihilation, pas par l'environnement.")

print("\n" + "=" * 72)
print("SYNTHÈSE")
print("=" * 72)
rows = [
    ("Lk entre trèfles", "interdit", "superselection de charge"),
    ("spin S/T0", "licite", "J ≳ 0,1 eV et B ≈ 220 T à 300 K ; 50–100 mK en pratique"),
    ("condensat 2×3₁", "licite", "fluxonium, 10–20 mK, T2 ≤ 1,5 ms"),
    ("trèfle + miroir", "licite", f"annihilation, T2 ≈ {T2*1e12:.0f} ps"),
    ("triplet NV (2 e⁻)", "licite", "300 K, T2 ≈ 1,8 ms — déjà en service"),
]
for r in rows:
    print(f"  {r[0]:<20} {r[1]:<9} {r[2]}")
print("L'enlacement protège la charge et la liaison ; il ne protège aucune phase.")
