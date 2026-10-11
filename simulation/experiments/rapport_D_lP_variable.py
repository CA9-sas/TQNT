"""Le rapport D/ℓ_P varie au début : chiffrage (16/09/2026, demande explicite).
D = tube du plateau (0,0214 fm), ℓ_P = ħ/(M_Pl c). On (1) écrit D/ℓ_P dans le cadre,
(2) pose D/ℓ_P(t) = r(t)·(D/ℓ_P)₀ et suit ce qui en découle, (3) borne r(t).
"""
import math
alpha = 1/137.035999; MPl = 1.22089e19; sqs = 0.41847; ml = 0.310; ell31 = 16.3724  # L/D
S_oct = 938.2/418.47
hbarc = 0.1973269804  # GeV fm
lP = hbarc/MPl        # fm
D = hbarc*(ml/sqs**2)/ell31 # fm : L_phys = ħc·m_l/σ_χ, D = L/ℓ(3₁)
print("=== 1. D/ℓ_P dans le cadre ===")
print(f"  D = m_ℓ/(σ_χ ℓ(3₁)) = {D:.4f} fm ; ℓ_P = {lP:.3e} fm ; D/ℓ_P = {D/lP:.3e}")
ratio_th = 0.75*S_oct/(ell31*math.sqrt(12/7*alpha**18))
print(f"  identité de hiérarchie (m_base = M_Pl√(12/7 α¹⁸) = √σ_χ·𝒮_octet, m_ℓ = ¾√σ_χ) :")
print(f"  D/ℓ_P = (3/4)·𝒮_octet / (ℓ(3₁)·√(12/7·α¹⁸)) = {ratio_th:.3e}  ({100*(ratio_th/(D/lP)-1):+.2f} %)")
print("  → D/ℓ_P est un NOMBRE PUR fixé par α = 1/(k+2), k = 135 entier, et par 𝒮_octet : le faire varier, c'est rompre l'identité")
print("    de hiérarchie m/M_Pl = √(12/7 α¹⁸) — c'est-à-dire faire varier M_Pl/√σ_χ à α fixe.\n")

print("=== 2. Si D/ℓ_P = r(t)·(D/ℓ_P)₀ (α fixe, échelles de la matière fixes) ===")
print("  M_Pl(t) = r·M_Pl,0 → G(t) = ħc/M_Pl² = G₀/r² ; ℓ_P ∝ r. r < 1 tôt = trame PAUVRE = gravité FORTE au début.")
print(f"  {'grandeur':>42} {'loi en r':>12}   commentaire")
for g, law, com in [
    ("G pour la matière", "r⁻²", "seule chose qui change physiquement"),
    ("H(T) = √(8πGρ/3)", "r⁻¹", "expansion plus rapide tôt"),
    ("t(T)", "r", "l'histoire en T est la même, plus courte en secondes"),
    ("M_horizon(T), M_DM causale(T)", "r³", "horizons plus légers tôt"),
    ("v_éch du nuage causal √(GM/R)", "r⁰", "G·r³/r → INVARIANT"),
    ("E_liaison par éremnion GMm/R", "r⁰", "INVARIANT"),
    ("δ_DM requis ρ_rad/ρ_DM (décrochage)", "r⁰", "G se simplifie → INVARIANT"),
    ("ρ_vir = 8ρ_ta(T_ta) → R_F(T_ta)", "r⁰", "compacité ↔ époque INCHANGÉE"),
    ("seuil PBH δ_c", "r⁰", "sans dimension → INVARIANT"),
    ("T_c du gel (c_eff σ_χ D₀)", "r⁰", "pas de G dedans"),
    ("Q = ρ/v³, borne de Pauli 2m⁴/h³", "r⁰", "espace des phases INCHANGÉ"),
    ("Ω_GW du gel ∝ G", "r⁻²", "1e-44 → toujours invisible"),
]:
    print(f"  {g:>42} {law:>12}   {com}")
print()
print("=== 3. Bornes sur r(t) ===")
for ep, t, bound in [("BBN (Y_p 1,5 %, D/H)", "1-200 s", "|ΔG/G| < 0,2 → 0,91 < r < 1,1"),
                     ("CMB (Planck, hauteur des pics)", "3,8e5 ans", "|ΔG/G| < 0,09 → |1−r| < 4,5 %"),
                     ("LLR + pulsars", "aujourd'hui", "|Ġ/G| < 1e-13/an → |1−r| < 3e-4 sur 4,5 Ga"),
                     ("gel → BBN", "1,6e-4 s → 1 s", "AUCUNE contrainte directe : fenêtre libre pour r(t)")]:
    print(f"  {ep:>32} {t:>16} : {bound}")
print()
print("=== 4. Ce que r < 1 dans la fenêtre gel→BBN ferait à la matière noire ===")
for r in (1.0, 0.3, 0.1, 0.01):
    G = 1/r**2
    # M_DM causale à T=7 keV (standard 6e5) ∝ r³ ; T où M_causale = 3e5 : M ∝ T⁻³ r³ → T ∝ r
    print(f"  r = {r:5.2f} : G/G₀ = {G:6.0f} ; M_DM causale(gel) = {2.1e-7*r**3:.1e} Msun ; T(M_causale=3e5) = {7*r:.2f} keV (si r encore < 1 là) ;"
          f" v_éch, δ requis, R_F(T_ta) : inchangés")
print("  → la gravité forte tôt raccourcit l'horloge et allège les horizons ; elle n'abaisse ni le δ isocourbe requis (1e3-1e4), ni la borne de")
print("    Liouville, ni le seuil de la perle légère. Le pic isocourbe reste le prix ; r(t) ne le paie pas.")
print(f"\n  rappel : un r ≠ 1 subsistant à la BBN change Y_p de ~0,4·ΔG/G : r = 0,9 → +8 % sur Y_p (exclu à 1,5 %) ; d'où r → 1 avant ~1 s.")
