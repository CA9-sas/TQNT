"""Theorie Nodale — Framework de validation predictive

Separe les particules SM en set de calibration et set de prediction.
Optimise eta sur le set de calibration, puis predit les masses du set
de prediction. Permet d'evaluer le pouvoir predictif reel du modele.

Calibration par defaut : e-, mu-, W+ (3 points, 1 parametre eta).
Prediction sur les 10 restantes.
"""
from __future__ import annotations
import numpy as np
from theory.multi_knot import SM_ASSIGNMENTS, cornell_equilibrium


# Calibration par defaut : 3 points bien mesures couvrant le spectre
DEFAULT_CALIBRATION = ["e-", "mu-", "W+"]


def split_calibration_prediction(
    cal_names: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Separe SM_ASSIGNMENTS en set de calibration et prediction.

    Parameters
    ----------
    cal_names : noms des particules de calibration (default: e-, mu-, W+)

    Returns
    -------
    (calibration_set, prediction_set)
    """
    if cal_names is None:
        cal_names = DEFAULT_CALIBRATION
    cal_set = [p for p in SM_ASSIGNMENTS if p["name"] in cal_names]
    pred_set = [p for p in SM_ASSIGNMENTS if p["name"] not in cal_names]
    return cal_set, pred_set


def optimize_eta(calibration_set: list[dict]) -> float:
    """Optimise eta par moindres carres log sur le set de calibration.

    eta = exp(mean(log(m_sm) - log(M_raw)))
    """
    log_ratio = []
    for p in calibration_set:
        eq = cornell_equilibrium(p["knots"], p["k"])
        log_ratio.append(np.log(p["mass_sm"]) - np.log(eq["mass_raw"]))
    return float(np.exp(np.mean(log_ratio)))


def predict_masses(eta: float, prediction_set: list[dict]) -> list[dict]:
    """Predit les masses du set de prediction avec eta donne.

    Returns list of dicts with keys:
        name, m_predicted, m_sm, ratio, error_pct
    """
    results = []
    for p in prediction_set:
        eq = cornell_equilibrium(p["knots"], p["k"])
        m_pred = eta * eq["mass_raw"]
        ratio = m_pred / p["mass_sm"]
        results.append({
            "name": p["name"],
            "m_predicted": m_pred,
            "m_sm": p["mass_sm"],
            "ratio": ratio,
            "error_pct": abs(ratio - 1.0) * 100,
        })
    return results


def count_degrees_of_freedom() -> dict:
    """Compte les parametres libres vs observables.

    Parametres libres :
    - eta (1 parametre de calibration masse)
    - gamma (1 parametre dans rho_J)
    - Assignations knots+k par particule (non continus)

    Observables predites :
    - 13 masses SM
    """
    n_particles = len(SM_ASSIGNMENTS)
    n_continuous_params = 2  # eta, gamma
    n_observables = n_particles  # 13 masses
    return {
        "n_particles": n_particles,
        "n_continuous_params": n_continuous_params,
        "param_names": ["eta", "gamma"],
        "n_observables": n_observables,
        "n_predictions": n_observables - n_continuous_params,
    }


def validation_report(cal_names: list[str] | None = None) -> dict:
    """Rapport complet de validation predictive.

    Parameters
    ----------
    cal_names : noms des particules de calibration

    Returns
    -------
    dict with keys: eta, calibration, predictions, chi2, max_error_pct,
                    mean_error_pct, dof
    """
    cal_set, pred_set = split_calibration_prediction(cal_names)
    eta = optimize_eta(cal_set)
    predictions = predict_masses(eta, pred_set)

    # Calibration errors
    cal_results = predict_masses(eta, cal_set)

    # Chi2 on predictions (log scale)
    log_errors = [np.log(p["ratio"])**2 for p in predictions]
    chi2 = float(np.sum(log_errors))

    error_pcts = [p["error_pct"] for p in predictions]

    return {
        "eta": eta,
        "cal_names": [p["name"] for p in cal_set],
        "calibration": cal_results,
        "predictions": predictions,
        "chi2_log": chi2,
        "max_error_pct": max(error_pcts) if error_pcts else 0.0,
        "mean_error_pct": float(np.mean(error_pcts)) if error_pcts else 0.0,
        "dof": count_degrees_of_freedom(),
    }
