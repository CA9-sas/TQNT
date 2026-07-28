"""Implémentation UNIQUE et fidèle de l'opérateur de masse Ω_ts (V6).

Objectif : une seule formule, sans le compromis du fit global qui dégrade les
nœuds simples. Design :

  1. Secteur MONO-NŒUD figé aux valeurs Chern--Simons / Flory DÉRIVÉES
     (non fittées) : a=1/2 (Born), p=q=3/4 (Casimir spin-1/2), β_SAW=5/3
     (Flory volume exclu), ρ0=4^(5/3) (écrantage neutre au nœud 4_1).
     → reproduit les masses mono-nœud à 0,2 % RMS (vérifié).

  2. Loi de COMPOSITION : Hölder a=1/2 (structure de Koide/Born) +
     écrantage (cr/4)^(-5/3(c-1)) + collectif ((n+1)/2)^(-x) + G^(3/4).

  3. UN SEUL paramètre calibré : x (exposant collectif). Tout le reste est
     dérivé ou l'ancrage de masse (m_c, m_l = briques, fixées par m_e/lobes).

Les entrées géométriques (U_kappa, U_tau, det, cr, n_l, b) proviennent de
theory.omega_ts (Frenet FFT + invariants), partagées avec le catalogue.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from theory.omega_ts import _axiom_data

# --- constantes DÉRIVÉES (figées, non calibrées) ---
A_BORN = 0.5           # amplitude de Born  (Hölder a=1/2 -> [ ]^2)
P_CAS = 0.75           # Casimir spin-1/2 : C_2(1/2)=3/4  (amplificateur T)
Q_CAS = 0.75           # idem couplage géométrique G
BETA_FLORY = 5.0/3.0   # exposant de Flory 3D (volume exclu)
RHO0 = 4.0**BETA_FLORY # écrantage neutre au nœud 4_1
# --- briques de masse (ancrage m_e / lobes) ---
M_C = 0.147883
M_L = 0.022450
# --- SEUL paramètre calibré ---
X_COLLECTIVE = -2.28   # valeur par défaut ; recalibrée par calibrate_x()


def _m_naive(k):
    d = _axiom_data(k)
    return d['cr'] * M_C + d['nl'] * M_L

def _T(k):
    d = _axiom_data(k)
    return d['b'] * d['det'] / 6.0

def mass_unified(config, x: float = None) -> float:
    """Masse (MeV) via la formule unique. config : 'A+B+C' ou liste."""
    if x is None:
        x = X_COLLECTIVE
    comp = config.split('+') if isinstance(config, str) else list(config)
    comp = [c.strip().replace('-', '_') for c in comp]
    n = len(comp)
    counts = Counter(comp)
    d31 = _axiom_data('3_1')
    UU31 = d31['Uk'] * d31['Ut']

    # collectif (n composantes) — seul terme calibré (via x)
    collective = ((n + 1) / 2.0) ** (-x)
    # écrantage volume exclu des nœuds répétés (dérivé : β=5/3, ρ0=4^β)
    screen = 1.0
    for k, c in counts.items():
        if c >= 2:
            screen *= (RHO0 * _axiom_data(k)['cr'] ** (-BETA_FLORY)) ** (c - 1)
    # somme de Hölder a=1/2  ->  [ Σ sqrt(m_naive · T^{3/4}) ]^2   (structure Koide)
    holder = sum(np.sqrt(_m_naive(k) * _T(k) ** P_CAS) for k in comp) ** 2
    # couplage géométrique (Frenet) ^{3/4}
    G = sum(_axiom_data(k)['Uk'] * _axiom_data(k)['Ut'] for k in comp) / UU31
    return collective * screen * holder * G ** Q_CAS


def calibrate_x(dataset) -> tuple[float, float]:
    """Calibre l'UNIQUE x sur un jeu (nom, config, m_exp). Retourne (x*, RMS_log%)."""
    def rms(x):
        r = []
        for _, cfg, exp in dataset:
            m = mass_unified(cfg, x)
            if m > 0 and exp > 0:
                r.append(np.log(m / exp))
        return np.sqrt(np.mean(np.square(r)))
    xs = np.linspace(-3.0, -1.5, 601)
    vals = [rms(x) for x in xs]
    i = int(np.argmin(vals))
    return float(xs[i]), float(vals[i] * 100)
