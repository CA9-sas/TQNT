"""Bilan des décalages possibles des 6 raies sombres (1,07-1,64 GeV) sous la trame.

Prépare les FENÊTRES DE RECHERCHE de la line search Fermi-LAT 18 ans en
intégrant ce que la trame de l'espace-temps prédit — et interdit — comme
décalages. Sort output/fermi_fenetres_raies.json consommé par l'analyse.

Canaux, avec statut :
[THÉORÈME]     Propagation : ZÉRO décalage et zéro dispersion — la trame est
               graduée en action (pas = h), pas de maille spatiale ; un photon
               de 1 GeV arrive du centre galactique (8 kpc) sans dérive
               d'énergie à tout ordre (V6 §trame-ondulant ; nul GRB
               E_QG > 9,3 E_P). Une raie DÉCALÉE différemment selon la
               distance de la source réfuterait la trame.
[DÉRIVÉ]       Émission dans un puits (S4‴, horloges ralenties) : décalage
               gravitationnel z ≈ |Φ|/c² au lieu d'émission (halo/bulbe)
               ~ 2 ppm — négligeable devant σ_E = 7 %.
[MESURÉ]       Doppler du halo : dispersion σ_v ~ 220 km/s → élargissement
               ~ 0,07 %, décalage moyen nul (halo au repos galactocentrique)
               — négligeable.
[CALIBRÉ]      Ancre de transmutation m_ℓ : les 6 masses sont géométriques
               (ACPR) × m_ℓ = 310 MeV, connue à ±4-7 % (pincer_ml_plateau :
               l'optimum du secteur est à −2σ du 310 calibré). Décalage
               COHÉRENT : le sextuplet glisse EN BLOC, les rapports
               E_i/E_j restent fixés par la géométrie (ratios d'entiers de
               rails). → LA bonne recherche est un ajustement de MOTIF :
               6 raies aux rapports fixes, UNE échelle libre s ∈ [0,93, 1,07].
[EXPLORATOIRE] Torsion périphérique S4″ : si un état strict porte n demi-tours
               d'anneau, M(n) = sqrt(M₀² + n·πσ_χ), πσ_χ = 0,5501 GeV²
               (le quantum V-P universel). Satellites possibles dans la bande —
               listés à titre exploratoire (S4″ est consolidé pour les hadrons
               de la trame ; son extension aux entrelacs sombres N'EST PAS
               canonique — brouillon, pas V6).
"""
import json
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / 'output'
OUT.mkdir(exist_ok=True)

# Les 6 états stricts — masses ACPR canoniques (2026-07-22)
STATES = [
    ('4_1+8_3',        1.07),
    ('6_3+6_3',        1.10),
    ('6_3+8_3',        1.22),
    ('8_3+8_3',        1.35),
    ('4_1+4_1+8_3',    1.47),
    ('6_3+6_3+6_3',    1.64),
]
PI_SIGMA = 0.5501          # GeV² — ½ tour de ruban (S4″, quantum V-P)
SCALE_BAND = 0.07          # ±7 % : ancre m_ℓ (enveloppe large)
SIG_E_INST = 0.07          # dispersion instrumentale LAT ~1 GeV

# --- bilan des canaux négligeables (chiffré, pour le registre)
PHI_GC = 2.0e-6            # |Φ|/c² au bulbe (v_esc ~ 600 km/s)
SIG_V = 220e3 / 3e8        # dispersion de vitesses du halo
print('=== Bilan des décalages (fraction de E) ===')
print(f'  propagation trame      : 0 exactement   [THÉORÈME — zéro dispersion]')
print(f'  gravitationnel émission: ~{PHI_GC:.0e}       [DÉRIVÉ S4‴ — négligeable]')
print(f'  Doppler halo (élarg.)  : ~{SIG_V:.0e}       [MESURÉ — négligeable]')
print(f'  ancre m_ℓ (cohérent)   : ±{SCALE_BAND:.0%}          [CALIBRÉ — DOMINANT]')
print(f'  → net : le sextuplet peut glisser EN BLOC de ±7 % ; les rapports')
print(f'    E_i/E_j sont FIXES (géométrie ACPR). Recherche de MOTIF requise.')

# --- fenêtres et satellites
windows = []
print('\n=== Fenêtres de recherche (raies primaires) ===')
for name, E0 in STATES:
    lo = E0 * (1 - SCALE_BAND)
    hi = E0 * (1 + SCALE_BAND)
    sat1 = float(np.sqrt(E0**2 + PI_SIGMA))       # 1 demi-tour
    sat2 = float(np.sqrt(E0**2 + 2 * PI_SIGMA))   # 1 tour
    windows.append({'state': name, 'E0_GeV': E0,
                    'window_GeV': [round(lo, 3), round(hi, 3)],
                    'satellite_demi_tour_GeV': round(sat1, 3),
                    'satellite_tour_GeV': round(sat2, 3)})
    print(f'  {name:14s} E0 = {E0:.2f}  fenêtre [{lo:.2f}, {hi:.2f}]  '
          f'satellites S4″ (explor.) : {sat1:.2f} / {sat2:.2f} GeV')

ratios = [round(E / STATES[0][1], 4) for _, E in STATES]
print('\n=== Motif (invariant sous le glissement d\'ancre) ===')
print('  rapports E_i/E_1 :', ratios)
print('  à σ_E = 7 %, raies 1 et 2 confondues (écart 2,8 %) → 5 structures.')

out = {
    'comment': 'Fenêtres de recherche des 6 raies sombres TQNT — bilan trame',
    'propagation_shift': 0.0,
    'propagation_status': 'THEOREME (zéro dispersion de la trame)',
    'grav_emission_shift': PHI_GC,
    'doppler_broadening': SIG_V,
    'anchor_scale_band': SCALE_BAND,
    'anchor_note': 'glissement COHÉRENT (m_ℓ ±7 %) — rapports fixes',
    'pattern_ratios_to_E1': ratios,
    'pi_sigma_GeV2': PI_SIGMA,
    'satellites_status': 'EXPLORATOIRE (extension S4″ hors du document principal — brouillon)',
    'instrument_sigma_E': SIG_E_INST,
    'lines': windows,
}
path = OUT / 'fermi_fenetres_raies.json'
path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print('\nécrit :', path)
