"""La température du fond cosmologique depuis la trame — ce qui est calculable.

CE QU'AUCUNE THÉORIE NE PEUT FAIRE : dériver T₀ = 2,725 K comme nombre absolu.
T tombe en 1/a — « 2,725 K » est une étiquette d'époque (équivalente à l'âge
de l'univers aujourd'hui). Demander T₀ à une théorie, c'est lui demander
« quand est maintenant ».

CE QUE LA TRAME PEUT FAIRE :
(1) dériver l'INVARIANT sans dimension η_B = n_b/n_γ (la miette CP imprimée
    à la condensation χ — document principal V6 : ORDRE DE GRANDEUR dérivé, la valeur
    fine attend le calcul du gel d'annihilation lié au vev, lui-même
    désormais dérivé → chantier nommé) ;
(2) fournir H₀ = 67,2 sous le théorème de platitude (notre θ* SMICA) ;
(3) prédire la PERFECTION du corps noir et T(z) = T₀(1+z) exactement
    (E = hf identitaire + zéro dispersion → le spectre de Planck est un
    invariant adiabatique transporté sans déformation ; distorsions μ et y
    au-delà du standard = ZÉRO — FIRAS à ~50 ppm est une vérification).

LE CALCUL : T₀ = [π² n_b / (2ζ(3) η_B)]^{1/3} · ħc/k_B
avec n_b = ω_b ρ_c/(h² m_p) — recensement baryonique INDÉPENDANT du CMB
(mesures de dispersion des FRB, relation de Macquart) + notre H₀ trame.
Le CMB lui-même ne peut pas fournir ω_b ici sans circularité (les hauteurs
de pics contraignent ρ_b/ρ_γ, donc supposent T₀).
"""
import numpy as np

HBAR_C = 197.3269804e-9      # eV·m
KB = 8.617333262e-5          # eV/K
ZETA3 = 1.2020569
M_P = 938.272e6              # eV
RHO_C_H2 = 1.87834e-26       # kg/m³ (ρ_c/h²)
KG_EV = 5.60959e35           # eV par kg

def t0_from(eta_B, omega_b):
    n_b = omega_b * RHO_C_H2 * KG_EV / M_P          # m⁻³
    n_g = n_b / eta_B
    kT = (n_g * np.pi**2 / (2 * ZETA3)) ** (1/3) * HBAR_C   # eV
    return kT / KB, n_b, n_g

# --- recensement baryonique indépendant du CMB : FRB (Macquart 2020, ±~40 %)
H_TRAME = 0.672                       # notre θ* SMICA + platitude-théorème
OMEGA_B_FRB = 0.051 * (H_TRAME / 0.70)   # Ω_b(FRB) ∝ h⁻¹ ; erreur ~+40/-45 %
omega_b_frb = OMEGA_B_FRB * H_TRAME**2
print(f'recensement FRB + H0 trame : ω_b = {omega_b_frb:.4f} (±~40 %)')

# (a) η_B mesuré (BBN/CMB : 6,1e-10) — contrôle de la chaîne
T0, n_b, n_g = t0_from(6.1e-10, omega_b_frb)
print(f'(a) η_B mesuré 6,1e-10 : n_b = {n_b:.3f} /m³, n_γ = {n_g:.2e} /m³')
print(f'    → T0 = {T0:.2f} K  (mesuré FIRAS : 2,7255 K ; erreur FRB ±40 % → ±13 % sur T0)')

# (b) η_B de la trame : ORDRE DE GRANDEUR (document principal V6 — gel d'annihilation non calculé)
print('(b) η_B trame (ordre dérivé, [2e-10, 2e-9]) :')
for eta in (2e-10, 6e-10, 2e-9):
    T0o, _, _ = t0_from(eta, omega_b_frb)
    print(f'    η_B = {eta:.0e} → T0 = {T0o:.2f} K')
print('    la racine cubique adoucit : un ordre sur η_B → ×2,15 sur T0.')
print()
print('BILAN : T0 = [π² n_b/(2ζ(3) η_B)]^(1/3) — la trame fournit η_B (ordre)')
print('et H0 (théorème) ; le recensement FRB fournit n_b sans le CMB.')
print('Aujourd\'hui : T0 ~ 2-6 K depuis la trame seule ; avec le gel')
print('d\'annihilation calculé (vev dérivé disponible), la cible est ~±15 %,')
print('bornée ensuite par le recensement baryonique. La prédiction NETTE de')
print('la trame n\'est pas la valeur mais la FORME : corps noir parfait,')
print('T(z) = T0(1+z) exact, distorsions μ/y nulles au-delà du standard.')
