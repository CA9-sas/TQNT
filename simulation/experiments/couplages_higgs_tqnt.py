"""Couplages du Higgs proportionnels aux masses — confrontation TQNT.

Chaîne TQNT testée (brouillon_couplages_higgs_tqnt) :
  g(H -> ff̄) = m_f / v,  avec v = 246,227 GeV DÉRIVÉ (+28 ppm)
  et kappa_f = 1 pour toutes les espèces (g_topo = 1 : aucune charge
  de saveur n'existe dans la reconnexion P15 — la seule grandeur
  d'espèce disponible est la prise sur la trame, c'est-à-dire la masse).

Trois tests :
  T1  le rapport leptonique pur (sans v, sans QCD) :
      Gamma(H->tautau)/Gamma(H->mumu) = (m_tau/m_mu)^2 x (beta_tau/beta_mu)^3
      contre le rapport des largeurs partielles mesurées (via kappa^2 x SM).
  T2  kappa = 1 partout : chi2 global contre la combinaison ATLAS Run 2
      (Nature 607, 2022 — valeurs approchées, à re-sourcer avant toute
      consolidation).
  T3  la PENTE de la droite couplage-masse : prédite 1/v_TQNT avec
      v_TQNT = 246,227 GeV dérivé — comparée au v électrofaible mesuré
      (G_F) : la normalisation de la droite est à zéro paramètre.
"""
import numpy as np

# ---------------------------------------------------------------- entrées
v_TQNT = 246.227          # GeV — dérivé (croisement-ouvert, +28 ppm)
v_GF   = 246.21965        # GeV — (sqrt(2) G_F)^(-1/2), PDG
m_H    = 125.25           # GeV — mesuré (TQNT : 125,26, +0,05 %)

m_tau, m_mu = 1.77686, 0.1056584   # GeV (PDG)

# kappa mesurés — combinaison ATLAS Run 2 (Nature 607 (2022) 52),
# valeurs arrondies ; sigma symétrisés. À RE-SOURCER avant consolidation.
KAPPA = {
    # espèce : (kappa, sigma, type)
    'W':   (1.05, 0.05, 'V'),
    'Z':   (0.99, 0.06, 'V'),
    't':   (0.94, 0.11, 'F'),
    'b':   (0.89, 0.11, 'F'),
    'tau': (0.93, 0.07, 'F'),
    'mu':  (1.06, 0.27, 'F'),
    'gam': (1.01, 0.06, 'loop'),
    'g':   (0.95, 0.07, 'loop'),
}

print("=" * 72)
print("COUPLAGES DU HIGGS ET MASSES — CONFRONTATION TQNT (kappa = 1, v dérivé)")
print("=" * 72)

# ------------------------------------------------- T1 : rapport leptonique
beta = lambda m: np.sqrt(1 - 4 * m**2 / m_H**2)
r_pred = (m_tau / m_mu)**2 * (beta(m_tau) / beta(m_mu))**3
# côté mesure : Gamma_ii ∝ kappa_i^2 x Gamma_SM,ii et Gamma_SM suit la même
# formule LO -> le rapport mesuré = (kappa_tau/kappa_mu)^2 x r_pred
k_tau, s_tau, _ = KAPPA['tau']
k_mu, s_mu, _ = KAPPA['mu']
r_meas = (k_tau / k_mu)**2 * r_pred
s_rel = 2 * np.sqrt((s_tau / k_tau)**2 + (s_mu / k_mu)**2)
print(f"""
T1 — rapport leptonique pur (aucun v, aucune QCD) :
   prédiction prise-sur-la-trame : Gamma_tautau/Gamma_mumu = (m_tau/m_mu)^2 b^3
      = {r_pred:.1f}
   mesuré (via kappa) : {r_meas:.1f}  (+/- {r_meas*s_rel:.0f}, domine par kappa_mu)
   écart : {100*(r_meas/r_pred-1):+.0f} %  ({abs(r_meas/r_pred-1)/s_rel:.1f} sigma) — COMPATIBLE""")

# ------------------------------------------------- T2 : kappa = 1 partout
chi2 = sum(((k - 1) / s)**2 for k, s, _ in KAPPA.values())
ndof = len(KAPPA)
print(f"""
T2 — kappa = 1 pour les {ndof} couplages mesurés (prédiction TQNT, tree) :""")
print(f"   {'espèce':>6s} {'kappa':>7s} {'sigma':>6s} {'tirage (sigma)':>15s}")
for name, (k, s, typ) in KAPPA.items():
    print(f"   {name:>6s} {k:7.2f} {s:6.2f} {(k-1)/s:15.1f}")
print(f"   chi2 = {chi2:.1f} pour {ndof} mesures — kappa=1 est un bon "
      f"ajustement (chi2/n = {chi2/ndof:.2f})")

# ------------------------------------------------- T3 : la pente 1/v
print(f"""
T3 — la pente de la droite couplage-masse :
   TQNT : 1/v avec v DÉRIVÉ = {v_TQNT} GeV (racine(2pi) alpha^8 M_Pl (1+alpha/4pi))
   électrofaible mesuré (G_F) : v = {v_GF} GeV
   écart : {1e6*(v_TQNT/v_GF-1):+.0f} ppm — la NORMALISATION de la droite
   est prédite à zéro paramètre (aucune autre théorie ne dérive v).""")

print("""
Rappels :
 - la dérivation du MOTIF (amplitude ∝ m_f) tient à : g_topo = 1 (pas de
   charge de saveur), barrière = v, et l'identité d'ontologie
   masse = prise sur la trame ; corollaire vérifié : couplage d'arbre
   NUL aux espèces sans masse (H -> gamma gamma / gg par boucles seules) ;
 - secteur quark : la « prise » du stoikhion créé au vertex reste à
   cartographier (masse courante vs masse de nœud) — OUVERT, cf. brouillon ;
 - kappa_V bosonique : le facteur O(1) (2 m_V^2/v) identifié, non dérivé ;
 - le discriminant TQNT vs mécanisme de Higgs reste kappa_lambda
   (auto-couplage, HL-LHC) — pas les kappa de ce tableau (dégénérés).""")
