"""Formule dimensionnelle des « sorties écrasées » de trous noirs (16/09/2026, exploration hors cadre 3+1).

Hypothèse de l'utilisateur : la LUMIÈRE est la sortie écrasée d'un trou noir à 4 dimensions d'espace ; les NEUTRINOS PLATS
sont la sortie des trous noirs de nos 3 dimensions. Règle : un trou noir en d dimensions d'espace écrase une dimension
et émet des quanta « plats » vivant en d−1 dimensions.

Formules (trou noir de Schwarzschild–Tangherlini en d dimensions d'espace, D = d+1) :
  horizon      r_h^(d−2) = 16π G_D M / ((d−1) Ω_(d−1) c²),   Ω_(d−1) = 2π^(d/2)/Γ(d/2)
  température  k T_H = (d−2) ħc / (4π r_h)
  sortie       quanta vivant en d−1 dimensions ; la dimension écrasée, de taille r_h, impose ses modes :
               bosons (périodiques)      m_n c² = n ħc / r_h         → mode n = 0 : SANS MASSE (plat)
               fermions (antipériodiques) m_n c² = (n+½) ħc / r_h    → premier mode : m c² = ħc / (2 r_h)
  puissance    Stefan–Boltzmann en d−1 dimensions sur un horizon de « surface » ∝ r_h^(d−2) :
               L ∝ r_h^(d−2) T_H^d ∝ ħc²/r_h²  — la même loi en r_h⁻² dans toutes les dimensions.
"""
import math
hbar=1.0546e-34; c=2.998e8; G=6.6710e-11; kB=1.381e-23; Msun=1.989e30; eV=1.602e-19
hbarc_MeVfm = 197.327
def Omega(n): return 2*math.pi**((n+1)/2)/math.gamma((n+1)/2)
print("=== 1. La règle des modes : ce que « plat » veut dire ===")
print("  une dimension écrasée de taille R impose aux ondes des modes discrets :")
print("   bosons   m c² = n ħc/R      → n = 0 : sans masse  → LUMIÈRE (photon = mode zéro d'une dimension écrasée)")
print("   fermions m c² = (n+½) ħc/R  → n = 0 : m c² = ħc/2R → c'est EXACTEMENT la loi du neutrino du cadre (m = ħc/2R) !")
print("  → la formule du neutrino du cadre se relit : premier mode fermionique (antipériodique) d'une dimension écrasée de rayon R. \n")

print("=== 2. Neutrinos plats : sorties des trous noirs de nos 3 dimensions (d = 3 → d−1 = 2) ===")
print(f"  {'M':>12} {'r_s (m)':>10} {'k T_H (eV)':>11} {'m_plat c² = ħc/2r_s (meV)':>26}")
for M, lab in ((1e22, "1e22 kg"), (1*Msun, "1 Msun"), (10*Msun, "10 Msun"), (3e5*Msun, "3e5 Msun"), (4e6*Msun, "4e6 Msun (Sgr A*)")):
    rs = 2*G*M/c**2
    kT = hbar*c/(4*math.pi*rs)/eV
    m = hbar*c/(2*rs)/eV*1e3
    print(f"  {lab:>12} {rs:10.2e} {kT:11.2e} {m:26.2e}")
print("  → pour tout trou noir astrophysique, m_plat ≲ 1e-8 meV : « presque des neutrinos », sans masse en pratique.")
for m_meV in (50, 10, 5):
    R = hbarc_MeVfm*1e-15/(2*m_meV*1e-9)
    M = R*c**2/(2*G)
    print(f"  un vrai neutrino de {m_meV} meV ↔ dimension écrasée de R = {R*1e6:.1f} µm ↔ trou noir de r_s = R : M = {M:.1e} kg")
print("  (1e21–1e22 kg tombe dans la fenêtre des trous noirs primordiaux « astéroïdes », non contrainte — COÏNCIDENCE sans mécanisme, non retenue)\n")

print("=== 3. La lumière : sortie écrasée d'un trou noir à 4 dimensions (d = 4 → d−1 = 3) ===")
print("  le photon est le mode zéro (n = 0) : sans masse quelle que soit la taille R₄ de la 4e dimension.")
print("  mais les modes n ≥ 1 existent : photons « excités » de masse n ħc/R₄. Non vus jusqu'aux énergies du LHC (~1 TeV) :")
for E_TeV in (1.0, 1.4):
    R4 = hbarc_MeVfm/(E_TeV*1e6)
    print(f"   ħc/R₄ > {E_TeV} TeV → R₄ < {R4:.1e} fm   (tube du cadre D = 2,1e-2 fm ; longueur de Planck 1,6e-20 fm)")
print("  → la 4e dimension, si la lumière y a sa source, est plus petite que 1e-4 fm : 200 fois plus fine que le tube des nœuds.")
print("  un trou noir « à 4 dimensions » dont l'horizon mesure r_h ≲ R₄ :")
for d in (3, 4):
    print(f"   d = {d} : k T_H = (d−2) ħc/(4π r_h) → pour r_h = 1e-4 fm : {(d-2)*hbarc_MeVfm/(4*math.pi*1e-4)/1e3:.0f} GeV")
print("  → ces trous noirs émettent à l'échelle du TeV–GeV : leurs sorties « écrasées » seraient des photons très énergétiques.\n")

print("=== 4. La loi de puissance commune ===")
print("  L ∝ ħc²/r_h² dans toutes les dimensions ; en 3D (Hawking) : L = ħc⁶/(15360 π G² M²) = (1/3840π)·ħc²/r_s²")
rs = 2*G*3e5*Msun/c**2
print(f"  point rouge de 3e5 Msun : ħc²/r_s² = {hbar*c**2/rs**2:.1e} W → L_Hawking ~ {hbar*c**2/rs**2/(3840*math.pi):.1e} W : la sortie thermique reste invisible ;")
print("  la fuite qui éteindrait les X (R-15) n'est pas thermique : elle exige la loi η_fuite, que la formule ne fixe pas.\n")
print("BILAN : la formule dimensionnelle — m c² = (n + s/2) ħc/R, s = 0 bosons, 1 fermions, R = taille de la dimension écrasée —")
print("relit le photon comme mode zéro (sans masse) et le neutrino du cadre comme premier mode fermionique (m = ħc/2R) :")
print("c'est la même structure que la formule de masse des neutrinos déjà adoptée. Les neutrinos plats des trous noirs")
print("astrophysiques en sont la limite R = r_s → m ≲ 1e-8 meV. La lumière « venue de la 4e dimension » impose R₄ < 1e-4 fm (LHC).")
