"""c_min du réservoir refait avec la reconnexion faible (16/09/2026).

La dérivation d'origine (derive_cmin_reservoir.py) : σ(c) = σ_EM α^(c−18), σ_EM = 1e-25 cm² (Thomson),
bain = baryons seuls ; voie 1 « né découplé » Γ/H(T_c) < 1, voie 2 « survie » N_coll < 1 → c_min = 23.
Deux défauts relevés depuis : σ_EM = 1e-25 est EXCLU pour les nœuds neutres (chapitre DM : neutralisation
par framing, couplage par polarisabilité) ; le bain e± (2,5e9 × les baryons) était omis.

Refaite ici avec les processus du cadre, séparés selon ce qu'ils font :
  (1) DESTRUCTION (change le type de nœud) = reconnexion P15 faible, radiative :
      σv = α G_F² Q²/π, Q = 140,7 MeV (e±) ou 140,2 + m_μ (μ±), × Vassiliev α^(c−18), × sites s ∈ [1, c].
  (2) COUPLAGE CINÉTIQUE (élastique, sans changement de type) = polarisabilité de l'anneau conducteur,
      α_pol = R_K³ (3,8e-4 fm³ à c = 23, R_K ∝ M) ; potentiel −αħc α_pol/(2r⁴) ; Born : f = π α α_pol E q/4.
"""
import math
alpha = 1/137.036; hbarc = 197.327; hbar = 6.582e-22; MPl = 1.221e22
v = 246220.0; GF = 1/(math.sqrt(2)*v**2)
me, mmu = 0.511, 105.66
Tc = 62.0
def M(c): return 310.0 + 70.1*(c - 3)
def H(T): return 1.66*math.sqrt(15.2 if T > 30 else 10.75)*T**2/MPl
def n_fermion(m, T, g=4):
    if m < T/3: return 0.365*T**3*g/4
    return g*(m*T/(2*math.pi))**1.5*math.exp(-m/T)
print("=== (1) Survie : reconnexion faible radiative (destruction) ===")
ne = n_fermion(me, Tc); nmu = n_fermion(mmu, Tc)
Qe, Qmu = 140.7, 140.2 + mmu
sv_e = alpha*GF**2*Qe**2/math.pi; sv_mu = alpha*GF**2*Qmu**2/math.pi
G0 = (ne*sv_e + nmu*sv_mu)/H(Tc)
print(f"  n_e± = {ne:.2e} MeV³, n_μ± = {nmu:.2e} MeV³ ; σv_e = {sv_e:.2e}, σv_μ = {sv_mu:.2e} MeV⁻²")
print(f"  Γ/H(T_c) hors Vassiliev = {G0:.0f} (e± {ne*sv_e/H(Tc):.0f} + μ± {nmu*sv_mu/H(Tc):.0f}) ; événements intégrés ≈ Γ/H(T_c)")
print(f"  {'c':>3} {'M (GeV)':>7} {'N (1 site)':>11} {'N (c sites)':>11}  verdict")
cmin1 = cminc = None
for c in range(18, 27):
    N1 = G0*alpha**max(0, c-18); Nc = N1*c
    ver = "survit" if Nc < 1 else ("survit (1 site)" if N1 < 1 else "détruit")
    if N1 < 1 and cmin1 is None: cmin1 = c
    if Nc < 1 and cminc is None: cminc = c
    print(f"  {c:3d} {M(c)/1e3:7.2f} {N1:11.2e} {Nc:11.2e}  {ver}")
print(f"  → c_min(survie) = {cminc} (tous sites) ; {cmin1} (un site)\n")

print("=== (2) Couplage cinétique : polarisabilité (élastique) ===")
print("  la voie « né découplé » d'origine exigeait Γ/H(T_c) < 1 ; avec la polarisabilité :")
for c in (20, 23, 26):
    ap = 3.8e-4*(M(c)/M(23))**3                  # fm³
    E = 3*Tc/hbarc; q = E                          # fm⁻¹
    f = math.pi*alpha*ap*E*q/4                     # fm
    sig = 4*math.pi*f**2                           # fm²
    n = 0.365*(Tc/hbarc)**3                        # fm⁻³
    Gam = n*sig*3e23                               # s⁻¹ (c = 3e23 fm/s)
    gam = Gam*(3*Tc/M(c))                          # relaxation d'impulsion
    Hs = H(Tc)/hbar
    r = gam/Hs
    Tkd = Tc*r**(-1/6)
    print(f"  c = {c} : α_pol = {ap:.1e} fm³, σ_el = {sig*1e-26:.1e} cm², Γ/H = {Gam/Hs:.1e}, relaxation d'impulsion γ/H = {r:.1e}"
          f" → découplage cinétique T_kd ≈ {Tkd:.0f} MeV (γ/H ∝ T⁶)")
print("  → AUCUN nœud neutre n'est « né découplé » : le critère d'origine, pris au sens strict, ne laisse aucune matière noire.")
print("  → le critère physique est un découplage cinétique bien avant le keV (pas d'amortissement des structures) : T_kd ≈ 4-5 MeV,")
print("    comme une matière noire froide ordinaire. Il ne sélectionne aucun c.\n")

print("=== (3) Conséquence : le réservoir commence à c = 20 ===")
P = lambda c: 0.68**(c - 23)
old = sum(P(c)*M(c) for c in range(23, 120))
for cm in (22, 21, 20):
    new = sum(P(c)*M(c) for c in range(cm, 120))
    print(f"  c_min = {cm} : masse du réservoir ×{new/old:.2f} → Ω = 0,6 × {new/old:.2f} = {0.6*new/old:.2f}  (comptage mesuré à l'ancien T_c : 3,9 → {3.9*new/old:.1f})")
frac = {c: P(c)*M(c)/sum(P(k)*M(k) for k in range(20, 120)) for c in (20, 21, 22, 23)}
print("  poids en masse du nouveau bord : " + ", ".join(f"c={c} {100*f:.0f} %" for c, f in frac.items()))
print("  réserve : la série 0,68 par croisement est prolongée de trois pas sous 23 (même loi de formation, pas de nombre nouveau).\n")
print("=== (4) Les classes 20-22 face aux données ===")
for c in (20, 21, 22):
    s_eq = 1.1e-48*(M(c)/M(23))**6
    print(f"  c = {c} : M = {M(c)/1e3:.2f} GeV, σ_eq(polarisabilité) ≈ {s_eq:.1e} cm² — DS-50 à ~1e-41 : invisible ; XQC/CRESST-surface : invisibles")
print("  (l'exclusion historique XQC/CRESST des tranches 19-22 reposait sur σ_EM α^(c−18) avec σ_EM = 1e-25, exclu depuis)")
