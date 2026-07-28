"""
Moment magnétique de l'électron en TQNT (nœud 3_1).

Reproduit :
  1) modèle un-corps (zitterbewegung topologique)  -> mu_e = mu_B, g=2
  2) gamma=2 depuis la topologie du 3_1 (Fary-Milnor + indice de tresse)
  3) P23 (champ du piège) 29 décades trop petit pour l'anomalie
  4) a_e = alpha/2pi = 2 * (1/4pi) * alpha, avec alpha = 1/(k+2) natif

Usage :  PYTHONPATH=. python3 -m simulation.experiments.electron_g2_topological
"""
from __future__ import annotations
import numpy as np
from theory.knot_catalog import _trefoil_curve, _unknot_curve

# --- constantes CODATA ---
e   = 1.602176634e-19
m_e = 9.1093837015e-31
c   = 299792458.0
hbar = 1.054571817e-34
mu0 = 1.25663706212e-6
mu_B = e * hbar / (2 * m_e)
a_e_meas = 1.15965218076e-3


def total_curvature(curve: np.ndarray) -> float:
    """oint kappa ds = sum |r' x r''| / |r'|^2 (dérivées par indice)."""
    r1 = np.gradient(curve, axis=0, edge_order=2)
    r2 = np.gradient(r1, axis=0, edge_order=2)
    num = np.linalg.norm(np.cross(r1, r2), axis=1)
    den = np.linalg.norm(r1, axis=1) ** 2
    return float(np.sum(num / den))


def winding_z(curve: np.ndarray) -> float:
    th = np.unwrap(np.arctan2(curve[:, 1], curve[:, 0]))
    return float((th[-1] - th[0]) / (2 * np.pi))


def one_body_model() -> dict:
    r = hbar / (2 * m_e * c)            # = lambda_C / 2  (contrainte luminale)
    w = 2 * m_e * c ** 2 / hbar         # pulsation zitterbewegung
    spin = m_e * (w * r) * r
    mu_loop = 0.5 * e * w * r ** 2
    F_cp = m_e * w ** 2 * r
    F_tension = m_e * c ** 2 / r
    return dict(r=r, w=w, v=w * r, spin_over_hbar=spin / hbar,
                mu_loop_over_muB=mu_loop / mu_B,
                mu_pred_over_muB=2 * mu_loop / mu_B,   # gamma=2
                Fcp_over_tension=F_cp / F_tension)


def g2_from_topology(N: int = 200000) -> dict:
    circle = _unknot_curve(N)[:-1]
    tref = _trefoil_curve(N)[:-1]
    return dict(curv_circle_over_pi=total_curvature(circle) / np.pi,
                curv_trefoil_over_pi=total_curvature(tref) / np.pi,
                ratio=total_curvature(tref) / total_curvature(circle),
                winding_trefoil=winding_z(tref))


def p23_in_trap(B: float = 5.0) -> dict:
    eps = 0.0                            # electron = noeud unique 3_1, n=1
    k_p = m_e * c ** 2
    da = (2 * mu_B ** 2 * B / k_p) * (1 + eps)
    return dict(da_p23=da, ratio_to_anomaly=da / a_e_meas,
                decades_too_small=np.log10(a_e_meas / da))


def anomaly_alpha(k: int = 135) -> dict:
    alpha = 1 / (k + 2)                  # natif TQNT
    # estimation crue auto-champ
    r = hbar / (2 * m_e * c)
    w = 2 * m_e * c ** 2 / hbar
    I = e * w / (2 * np.pi)
    B_self = mu0 * I / (2 * r)
    a_crude = mu_B * B_self / (m_e * c ** 2)
    # série QED
    c1, c2, c3, c4 = 0.5, -0.328478965579, 1.181241456, -1.91245
    def series(al):
        x = al / np.pi
        return c1 * x + c2 * x ** 2 + c3 * x ** 3 + c4 * x ** 4
    return dict(alpha=alpha, a_crude_over_alpha=a_crude / alpha,
                crude_over_exact=(a_crude / alpha) / (1 / (2 * np.pi)),
                a_e_tqnt=series(alpha),
                a_e_codata=series(1 / 137.035999084),
                a_e_meas=a_e_meas)


if __name__ == "__main__":
    print("1) MODELE UN-CORPS")
    for k, v in one_body_model().items():
        print(f"   {k:22s} = {v:.6f}")
    print("\n2) g=2 TOPOLOGIQUE (Fary-Milnor + tresse)")
    for k, v in g2_from_topology().items():
        print(f"   {k:22s} = {v:.4f}")
    print("\n3) P23 DANS LE PIEGE (B=5 T)")
    for k, v in p23_in_trap().items():
        print(f"   {k:22s} = {v:.4e}")
    print("\n4) a_e = alpha/2pi = 2*(1/4pi)*alpha, alpha natif 1/(k+2)")
    for k, v in anomaly_alpha().items():
        print(f"   {k:22s} = {v:.6e}")
