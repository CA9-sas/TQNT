r"""
Chantier 1 — J3, J4, J5 : test décisif du Hamiltonien multi-particule.

Objectif (cf. docs/brouillon_hamiltonien_multipart.tex) : tester si une
fonctionnelle géométrique simple h(K) reproduit le spectre V4 pour les
19 nœuds simples du catalogue, après une SEULE calibration sur
m(3_1) = m_e = 0,511 MeV.

* J3 : tabule les quantités géométriques (U_EM, c_J, L) et la prédiction
       de la formule de masse optimisée V5 (éq. 6.8, theory.mass_formula).
* J4 : pour chaque candidat h(K), calibre ε = m_e / h(3_1).
* J5 : prédit m_pred(K) = ε · h(K) sur les 18 autres nœuds et compare à
       m_V4(K) (la formule 6.8 elle-même). Mesure : RMS log10(m_pred/m_V4).

Lecture des résultats :
  - V4-product (U_EM · c_J^3.55) donne trivialement RMS = 0 (auto-référent).
  - Si un candidat plus SIMPLE (U_EM seul, c_J seul, L seul, U_EM·L) donne
    RMS ~ 0,2 (facteur 1,6), il y a une voie vers une formule sans
    exposant ajusté.
  - Si tous les candidats simples donnent RMS > 1 (facteur 10), l'exposant
    3,55 capture quelque chose d'essentiel qui devra émerger du H_TQNT
    complet (Neumann + tension), pas d'une seule loi de puissance.

Usage :  python3 -m simulation.experiments.mass_variational
"""
from __future__ import annotations
import numpy as np

from theory.mass_formula import u_em, c_j, mass
from theory.abraham import REFERENCE_N50K

KNOTS = list(REFERENCE_N50K.keys())
M_E = 0.51099895  # MeV


# ============================================================
# J3 — tableau des quantités géométriques
# ============================================================
def collect_geometric() -> dict[str, tuple[float, float, float, float]]:
    """Retourne {K: (U_EM, c_J, L, m_V4_MeV)}."""
    return {K: (u_em(K), c_j(K), REFERENCE_N50K[K]["L"], mass(K, "single"))
            for K in KNOTS}


def print_J3_table(data: dict) -> None:
    print("=" * 72)
    print("  JALON J3 — quantités géométriques + masse V4 (formule 6.8)")
    print("=" * 72)
    print(f"  {'K':>5} {'U_EM':>10} {'c_J':>8} {'L':>8} {'m_V4 (MeV)':>14}")
    print("  " + "-" * 50)
    for K, (U, cJ, L, m) in data.items():
        print(f"  {K:>5} {U:>10.2f} {cJ:>8.2f} {L:>8.2f} {m:>14.4f}")
    print()


# ============================================================
# J4 + J5 — calibration unique + prédiction + écart vs V4
# ============================================================
def test_candidate(data: dict, name: str, h: callable
                   ) -> tuple[float, float, float, list[tuple]]:
    """Calibre ε sur m(3_1)=m_e, prédit, et mesure log10(pred/V4)."""
    K0 = "3_1"
    U0, cJ0, L0, m0 = data[K0]
    h0 = h(U0, cJ0, L0)
    eps = M_E / h0
    devs = []
    rows = []
    for K, (U, cJ, L, mV4) in data.items():
        m_pred = eps * h(U, cJ, L)
        log_dev = np.log10(m_pred / mV4)
        devs.append(log_dev)
        rows.append((K, m_pred, mV4, log_dev))
    devs = np.array(devs)
    rms = float(np.sqrt(np.mean(devs ** 2)))
    max_dev = float(np.max(np.abs(devs)))
    return eps, rms, max_dev, rows


def best_exponent_for_form(data: dict, form_kind: str) -> tuple[float, float]:
    """Cherche l'exposant p qui minimise le RMS pour les formes 'cJ^p',
    'U·cJ^p', 'L^p'. Retourne (p_optim, RMS_optim)."""
    def make_h(kind, p):
        if kind == "cJ^p":   return lambda U, cJ, L: cJ ** p
        if kind == "U·cJ^p": return lambda U, cJ, L: U * cJ ** p
        if kind == "L^p":    return lambda U, cJ, L: L ** p
        raise ValueError(kind)
    best_p, best_rms = None, np.inf
    for p in np.linspace(0.5, 6.0, 56):
        _, rms, _, _ = test_candidate(data, form_kind, make_h(form_kind, p))
        if rms < best_rms:
            best_rms = rms
            best_p = p
    return float(best_p), float(best_rms)


def main():
    data = collect_geometric()
    print_J3_table(data)

    # Candidats à 1 paramètre (eps seul, calibré sur m_e) ---------------------
    candidates = {
        "U_EM seul":                    lambda U, cJ, L: U,
        "c_J seul":                     lambda U, cJ, L: cJ,
        "L seul":                       lambda U, cJ, L: L,
        "c_J^3.55 seul":                lambda U, cJ, L: cJ ** 3.55,
        "U_EM × L":                     lambda U, cJ, L: U * L,
        "U_EM / L":                     lambda U, cJ, L: U / L,
        "U_EM × c_J":                   lambda U, cJ, L: U * cJ,
        "U_EM × c_J^3.55 (V4)":         lambda U, cJ, L: U * cJ ** 3.55,
        "L × c_J^3.55":                 lambda U, cJ, L: L * cJ ** 3.55,
    }

    print("=" * 72)
    print("  JALONS J4-J5 — candidats à 1 paramètre, calibrés sur m(3_1)=m_e")
    print("=" * 72)
    header = f"  {'candidat':>32} {'ε calibré':>14} {'RMS log10':>11} {'max':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    results = []
    for name, hfunc in candidates.items():
        eps, rms, mx, rows = test_candidate(data, name, hfunc)
        results.append((name, rms, mx, eps, rows))
        print(f"  {name:>32} {eps:>14.3e} {rms:>11.3f} {mx:>7.2f}")

    print("\n  Classement par RMS log10 (plus petit = mieux, 0 = identique V4) :")
    for name, rms, mx, _, _ in sorted(results, key=lambda x: x[1]):
        ratio = 10 ** rms
        print(f"    {name:>32}  RMS = {rms:.3f}  (facteur typique {ratio:.2f}×)")

    # Recherche de l'exposant optimal pour les formes paramétriques ----------
    print("\n  Exposants optimaux (recherche par bisection RMS) :")
    for form in ["cJ^p", "U·cJ^p", "L^p"]:
        p_opt, rms_opt = best_exponent_for_form(data, form)
        print(f"    {form:>10} : p_opt = {p_opt:.2f}  ->  RMS = {rms_opt:.3f}")

    # Détail du meilleur candidat NON V4 -------------------------------------
    non_v4 = [r for r in results if "V4" not in r[0]]
    best = min(non_v4, key=lambda x: x[1])
    print(f"\n  Meilleur candidat NON-V4 : {best[0]}  (RMS = {best[1]:.3f})")
    print(f"  Détail nœud par nœud :")
    print(f"    {'K':>5} {'m_pred (MeV)':>14} {'m_V4 (MeV)':>12} {'log10 dev':>11}")
    for K, m_pred, mV4, dev in best[4]:
        flag = " " if abs(dev) < 0.3 else "*"
        print(f"    {K:>5} {m_pred:>14.4f} {mV4:>12.4f} {dev:>+11.3f} {flag}")
    print("\n  (* = écart > 0,3 décade ≈ facteur 2)")
    print("=" * 72)


if __name__ == "__main__":
    main()
