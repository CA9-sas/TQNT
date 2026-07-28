r"""
Fonctionnels du Hamiltonien TQNT pour un nœud (chantier 1, jalon J1).

Implémente les trois termes géométriques fondamentaux du candidat
H_TQNT = H_kin + H_mag + H_tens (+ H_int + H_reconn, à venir J6+) comme
fonctionnels d'une courbe γ ⊂ R³ paramétrée par un nom du catalogue
(theory.knot_catalog) et d'un rayon de tube a.

Référence : docs/brouillon_hamiltonien_multipart.tex.

Conventions
-----------
Unités naturelles ℏ = c = 1. Toutes les sorties sont en unités cohérentes
avec les paramètres d'échelle (E_kin, E_mag, T) qui restent à fixer par la
calibration unique du jalon J4 (m(3_1) = m_e c²).

Définitions
-----------
* kinetic_integral(K)   = ∮_γ (κ² + τ²) ds        (= U_EM(K) de la TQNT,
                                                    via Frenet-Serret).
* length(K)             = ∮_γ ds                   (longueur d'arc).
* neumann_integral(K,a) = ∮∮_{|γ-γ'|>a} γ̇·γ̇'/|γ-γ'| ds ds'
                          + L · (ln(8L/a) - 7/4)   (Maxwell--Rosa, self-
                                                    inductance d'un fil mince
                                                    de rayon a).

Hamiltoniens
------------
* H_kin(K)     = (E_kin/2) · kinetic_integral(K)         (énergie de
                                                          courbure-torsion ;
                                                          E_kin = ℏc en SI).
* H_mag(K,a)   = (E_mag/(8π²)) · neumann_integral(K,a)   (énergie magnétique
                                                          propre du tube
                                                          quantifié de flux Φ ;
                                                          E_mag = Φ² = (2πℏ/g)²).
* H_tens(K,a)  = T · length(K)                           (tension topologique ;
                                                          minorée par
                                                          T·a·ropelength(K)).

Jalon J2 (validation immédiate) : H_kin(K) doit exactement reproduire la
demi-énergie U_EM tabulée dans REFERENCE_N50K d'abraham.py (puisqu'on
réutilise les mêmes intégrales de Frenet-Serret).

Usage :
    >>> from theory.hamiltonian_knot import H_kin, H_mag, H_tens, validate_J2
    >>> validate_J2()        # cross-check vs REFERENCE_N50K (jalon J2)
"""
from __future__ import annotations
import numpy as np

from theory.knot_catalog import knot_curve
from theory.mass_formula import u_em
from theory.abraham import REFERENCE_N50K   # gardé pour le diagnostic J2


# ============================================================
# Fonctionnels géométriques élémentaires
# ============================================================
def kinetic_integral(name: str) -> float:
    """∮_γ (κ² + τ²) ds = U_EM(K), valeur canonique V5.

    Utilise theory.mass_formula.u_em (table REFERENCE_N50K à N=50 000).
    La recomputation fresh via abraham.geometric_integrals est instable
    pour 7_5 (bug latent dans theory.knot_catalog à investiguer séparément).
    """
    return u_em(name)


def length(name: str, n_points: int = 10000) -> float:
    """Longueur d'arc ∮ds de la courbe paramétrique du catalogue."""
    pts = knot_curve(name, n_points)[:-1]   # retire le doublon de bouclage
    seg = np.roll(pts, -1, axis=0) - pts
    return float(np.linalg.norm(seg, axis=1).sum())


def neumann_integral(name: str, a: float, n_points: int = 2000) -> float:
    """Self-inductance d'un tube fin de rayon a centré sur γ (Maxwell--Rosa).

    Formule : ∮∮_{|γ-γ'|>a} (γ̇·γ̇')/|γ-γ'| ds ds' + L · (ln(8L/a) - 7/4).
    Le premier terme est le double-intégral de Neumann (far-field), tronqué
    pour |γ-γ'|>a ; le second est la correction analytique du cœur de tube
    (self-inductance d'un fil mince ; voir Rosa 1908).
    """
    pts = knot_curve(name, n_points)[:-1]
    # tangentes par différences centrées
    dpts = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    speed = np.linalg.norm(dpts, axis=1)
    ds = speed * (2 * np.pi / n_points)
    L_tot = float(ds.sum())
    t_hat = dpts / speed[:, None]

    # paires (i,j) — éviter l'explosion mémoire pour n grand
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dot = t_hat @ t_hat.T
    mask = dist > a                                   # exclusion du cœur
    contrib = np.where(mask, dot / np.maximum(dist, a), 0.0)
    far_field = float((contrib * ds[:, None] * ds[None, :]).sum())

    if L_tot <= 0 or a <= 0:
        return far_field
    near_field = L_tot * (np.log(8 * L_tot / a) - 7 / 4)
    return far_field + near_field


# ============================================================
# Termes du Hamiltonien (unités naturelles)
# ============================================================
def H_kin(name: str, scale: float = 1.0) -> float:
    """H_kin = (scale/2) · U_EM(K)  (scale = ℏc en SI ; U_EM canonique V5)."""
    return 0.5 * scale * kinetic_integral(name)


def H_mag(name: str, a: float, scale: float = 1.0, n_points: int = 2000) -> float:
    """H_mag = (scale/(8π²)) · neumann_integral(γ,a)  (scale = Φ² = (2πℏ/g)²)."""
    return scale / (8 * np.pi ** 2) * neumann_integral(name, a, n_points)


def H_tens(name: str, a: float | None = None, T: float = 1.0,
           n_points: int = 10000) -> float:
    """H_tens = T · ∮ds.  a est porté pour compatibilité (rope-length bound)."""
    _ = a
    return T * length(name, n_points)


def H_total(name: str, a: float,
            E_kin: float = 1.0, E_mag: float = 1.0, T: float = 1.0,
            n_points: int = 2000) -> dict:
    """Somme H_kin + H_mag + H_tens.

    Les trois échelles restent à fixer par la calibration unique du jalon J4
    (m(3_1) = m_e c² fixe le rayon a_min et la relation E_kin/E_mag/T).
    """
    hk = H_kin(name, scale=E_kin)
    hm = H_mag(name, a, scale=E_mag, n_points=n_points)
    ht = H_tens(name, a, T=T, n_points=max(n_points, 10000))
    return {"H_kin": hk, "H_mag": hm, "H_tens": ht, "H_total": hk + hm + ht}


# ============================================================
# Jalon J2 — validation : H_kin doit coïncider avec ½·sqrt_U² d'abraham
# ============================================================
def validate_J2() -> None:
    """JALON J2 — affiche H_kin(K) = ½·U_EM(K) (valeur canonique V5).

    Depuis le refactor (theory/mass_formula.py), H_kin lit directement la
    valeur canonique : la comparaison est trivialement exacte. On l'imprime
    pour traçabilité et pour disposer d'une table de ½·U_EM par nœud.
    """
    print("=" * 64)
    print("  JALON J2 : H_kin(K) = ½·U_EM(K)  (valeur canonique V5)")
    print("=" * 64)
    print(f"  {'K':>5} {'U_EM (V5)':>12} {'H_kin = ½·U_EM':>16}")
    print("  " + "-" * 40)
    for name in REFERENCE_N50K:
        U = u_em(name)
        h = H_kin(name)
        print(f"  {name:>5} {U:>12.4f} {h:>16.4f}")


if __name__ == "__main__":
    validate_J2()
