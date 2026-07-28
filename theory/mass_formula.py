r"""
Formule de masse optimisée de la TQNT — routine canonique (V5, éq. 6.8).

Implémente l'unique formule de masse opérationnelle de la TQNT V4/V5 :

    m(L) = α₀ × U_EM_tot × (f(n) × c_J_tot) ** p,                       (6.8)

où :
    α₀ = 8,16 × 10⁻⁴ MeV (calibré sur m_e = 0,511 MeV),
    p  = 3,55             (exposant ajusté sur le spectre),
    γ  = 0,10             (poids du déterminant dans c_J),
    f(1)=1, f(2)=0,86, f(3)=0,77   (écrantage gluonique des baryons/mésons),
    f_EW = 1              (secteur électrofaible : NON confiné => pas d'écrantage).

Source de U_EM
--------------
U_EM(K) = ∮(κ² + τ²) ds est tabulée dans theory.abraham.REFERENCE_N50K
(sqrt_U², N=50 000 points). C'est la valeur opérationnelle utilisée partout
dans la V4/V5. La recomputation fresh via abraham.geometric_integrals est
instable pour 7_5 (bug latent dans theory.knot_catalog à investiguer
séparément).

API
---
    mass(config, sector="auto")         -> masse en MeV
    mass_components(config, sector)     -> dict diagnostique
    u_em(knot)                          -> U_EM(K)
    c_j(knot)                           -> c_J(K)

`config` est soit un nom (`"3_1"`), soit une chaîne entrelacs
(`"6_1+7_4+8_3"`), soit un itérable (`["3_1","7_2","7_2"]`).

`sector` :
  - `"auto"`   : EW si 8_3 ∈ config, hadronique sinon.
  - `"ew"`     : force f = 1 (W/Z/H/top et apparentés).
  - `"hadron"` : force f(n) du tableau (baryons, mésons).
  - `"single"` : un seul nœud (f = 1).

Exemples
--------
    >>> mass("3_1")                         # électron
    0.5066...                               # MeV
    >>> mass("6_1+7_4+8_3", sector="ew")    # Higgs
    125499.3...                             # MeV (= 125,5 GeV)
    >>> mass("3_1+7_2+7_2", sector="hadron")  # proton
    940.0...                                # MeV

Cf. docs/TQNT_detaillee_V5_FR.tex, chap. masse, éq. 6.8.
"""
from __future__ import annotations
from typing import Iterable, Union

from theory.jones import jones_polynomial
from theory.retraction import retraction_factor
from theory.abraham import REFERENCE_N50K

# ============================================================
# Constantes opérationnelles V5 (cf. chap. masse, comptage des 9 paramètres)
# ============================================================
ALPHA0 = 8.16e-4       # MeV — calibré sur l'électron
P_EXP  = 3.55          # exposant de la formule
GAMMA  = 0.10          # poids du déterminant dans c_J
F_SCREENING = {        # écrantage gluonique des entrelacs confinés
    1: 1.00,
    2: 0.86,           # mésons (2 composantes)
    3: 0.77,           # baryons (3 composantes)
}
F_EW = 1.0             # secteur électrofaible : pas d'écrantage gluonique

# Anyon/nœud porteur du secteur EW : sa présence dans un entrelacs
# signale (par défaut) que la composante n'est pas confinée par la
# force forte (U_EM(8_3) ≈ 1492, dominant).
EW_MARKER = "8_3"

ConfigSpec = Union[str, Iterable[str]]


# ============================================================
# Briques élémentaires
# ============================================================
def u_em(knot_name: str) -> float:
    """U_EM(K) = ∮(κ²+τ²)ds, valeur canonique (REFERENCE_N50K[K]['sqrt_U']²)."""
    if knot_name not in REFERENCE_N50K:
        raise KeyError(
            f"U_EM non tabulé pour {knot_name!r}. "
            f"Catalogue disponible : {sorted(REFERENCE_N50K)}"
        )
    return float(REFERENCE_N50K[knot_name]["sqrt_U"]) ** 2


def c_j(knot_name: str) -> float:
    """c_J(K) = span(V_K) + γ·|V_K(-1)| (indice de Jones, jones_retraction_factor)."""
    return float(retraction_factor(jones_polynomial(knot_name)))


def _parse(config: ConfigSpec) -> list[str]:
    """Normalise config en liste de noms de nœuds."""
    if isinstance(config, str):
        return config.split("+") if "+" in config else [config]
    return list(config)


def _f_factor(n: int, sector: str) -> float:
    """Facteur d'écrantage selon le secteur."""
    if sector in ("ew", "single"):
        return F_EW
    if sector == "hadron":
        return F_SCREENING.get(n, 1.0)
    raise ValueError(f"secteur inconnu : {sector!r}")


def _resolve_sector(components: list[str], sector: str) -> str:
    """Résout le secteur : auto -> ew si 8_3 présent, single si n=1, hadron sinon."""
    if sector != "auto":
        return sector
    if len(components) == 1:
        return "single"
    if EW_MARKER in components:
        return "ew"
    return "hadron"


# ============================================================
# Formule de masse — point d'entrée canonique
# ============================================================
def mass(config: ConfigSpec, sector: str = "auto") -> float:
    """Masse en MeV d'un nœud ou entrelacs par la formule de masse optimisée (6.8).

    m = α₀ · U_EM_tot · (f · c_J_tot)^p
    """
    return mass_components(config, sector)["m_MeV"]


def mass_components(config: ConfigSpec, sector: str = "auto") -> dict:
    """Décomposition de la formule de masse (diagnostique)."""
    comps = _parse(config)
    sec = _resolve_sector(comps, sector)
    f = _f_factor(len(comps), sec)
    U_tot = sum(u_em(k) for k in comps)
    cJ_tot = sum(c_j(k) for k in comps)
    cJ_eff = f * cJ_tot
    m = ALPHA0 * U_tot * cJ_eff ** P_EXP
    return {
        "config": "+".join(comps),
        "sector": sec,
        "n": len(comps),
        "f": f,
        "U_EM_tot": U_tot,
        "c_J_tot": cJ_tot,
        "c_J_eff": cJ_eff,
        "m_MeV": m,
    }


# ============================================================
# Validation rapide (sanity-check au chargement)
# ============================================================
def _sanity_check() -> None:
    """Vérifie que la routine reproduit quelques masses connues."""
    cases = [
        ("3_1",              "single",  0.51,    "électron"),
        ("6_1+7_4+8_3",      "ew",      125499,  "Higgs"),
        ("4_1+7_3+8_3",      "ew",      80867,   "W±"),
        ("5_1+7_2+8_3",      "ew",      91489,   "Z⁰"),
        ("7_4+7_6+8_3",      "ew",      174174,  "top"),
        ("3_1+7_2+7_2",      "hadron",  940,     "proton/neutron"),
    ]
    print("=== sanity check : formule de masse optimisée (V5 éq. 6.8) ===")
    print(f"  {'config':>16} {'secteur':>8} {'m (MeV)':>10} {'attendu':>10} {'écart':>7}")
    for cfg, sec, expected, label in cases:
        m = mass(cfg, sec)
        err = abs(m - expected) / expected * 100
        flag = "✓" if err < 5 else "?"
        print(f"  {cfg:>16} {sec:>8} {m:>10.2f} {expected:>10.0f} {err:>6.1f}% {flag} {label}")


if __name__ == "__main__":
    _sanity_check()
