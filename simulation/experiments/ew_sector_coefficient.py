r"""
Coefficient de secteur électrofaible : il N'EXISTE PAS (contraste avec les baryons).

Suite à baryon_confinement_coefficient.py (un facteur de secteur constant ramène
l'octet u/d/s + Δ à 2 %, ordre reproduit), on cherche le même pour l'EW
(W, Z, H, top — assignations autoritatives ew_vev_derivation).

RÉSULTAT : NÉGATIF. Un coefficient constant ne fait que caler la moyenne
(plancher RMS 18 %), et surtout la géométrie **brouille l'ordre** :
    ordre exp  : W < Z < H < top
    ordre préd : W < Z < top < H   (top prédit plus léger que H)
    résidus    : W -3 %, Z +24 %, H +9 %, top -24 %  (mixed-sign)

POURQUOI (diagnostic à l'échange d'un seul nœud) :
  - H = 7_1+8_3+10_1  vs  top = 8_2+8_3+10_1  ne diffèrent QUE par 7_1 vs 8_2.
    Φ_F(7_1)=7029  >  Φ_F(8_2)=2337  → la géométrie dit top < H.
    Mais physiquement top(172) > H(125) : le 8_2 DEVRAIT peser PLUS que le 7_1.
    L'ordre est INVERSÉ par la géométrie.
  - Z = 7_2+8_3+9_1 : le 9_1 (torus T(2,9)) a Φ_F=10920 (énorme) → Z gonflé +24 %.

Un COEFFICIENT CONSTANT ne peut pas corriger un ordre BROUILLÉ (il déplace la
moyenne, pas le classement). CONTRASTE avec l'octet baryonique, dont l'ordre
p<Λ<Σ<Ξ EST suivi par la géométrie (échanges de nœuds légers 3_1/5_2/7_2 corrélés
à la masse) — d'où le coefficient qui y marche à 2 %.

INTERPRÉTATION. Deux lectures, non tranchées :
  (a) fondamental : les masses EW ne sont pas suivies par la géométrie des nœuds ;
  (b) assignations : W/Z/H/top ont été CALIBRÉES sur l'ancien Ω_ts phénoménologique
      (courbes fausses, cf. [[catalog-curves-invalid-Ukappa-intrinsic]]) — donc
      possiblement artefactuelles au même titre que les courbes. Le 8_2 (top) et le
      9_1 (Z) sont les coupables géométriques ; une ré-assignation les changerait.
Cohérent avec [[electroweak-sector-true-knots]] : « mixed-sign deviation pattern,
no global recalibration possible ». Le présent test le CONFIRME sur la géométrie
correcte et l'explique (échanges 1-nœud trop sensibles, ordre inversé).
"""
from __future__ import annotations
import itertools
import numpy as np
from collections import Counter
import theory.omega_ts as ots
from theory.abraham import REFERENCE_N50K as GEO

AMPHI = {"4_1", "6_3", "8_3"}
EW = [("W", "7_1+7_2+8_3", 80377.), ("Z", "7_2+8_3+9_1", 91188.),
      ("H", "7_1+8_3+10_1", 125250.), ("top", "8_2+8_3+10_1", 172500.)]


def _install_ideal(f=0.5):
    new = {k: (GEO[k]["int_kappa"] ** 2 / GEO[k]["L"],
               GEO[k]["int_abs_tau"] ** 2 / (f * GEO[k]["L"])) for k in GEO}
    orig = ots._axiom_data
    ots._axiom_data = lambda name: {**orig(name),
        **({"Uk": new[name][0], "Ut": new[name][1]} if name in new else {})}


def PhiF(K):
    d = ots._axiom_data(K)
    return d["Uk"] ** (5 / 3) * d["Ut"]


def base(cfg, s):
    from simulation.experiments.omega_unified import (
        _m_naive, _T, RHO0, BETA_FLORY, P_CAS, Q_CAS)
    comp = [x.strip().replace("-", "_") for x in cfg.split("+")]
    n = len(comp); cnt = Counter(comp)
    screen = 1.0
    for k, cc in cnt.items():
        if cc >= 2:
            screen *= (RHO0 * ots._axiom_data(k)["cr"] ** (-BETA_FLORY)) ** (cc - 1)
    b = [_m_naive(k) * _T(k) ** P_CAS for k in comp]
    holder = (sum(np.sqrt(bi) for bi in b)) ** 2
    G = sum(PhiF(k) for k in comp) / PhiF("3_1")
    pair = 1.0
    for i, j in itertools.combinations(range(n), 2):
        if comp[i] not in AMPHI and comp[j] not in AMPHI:
            pair *= s
    return holder * G ** Q_CAS * pair


def main():
    from scipy.optimize import minimize
    _install_ideal()
    s = minimize(lambda p: np.std([np.log(base(cfg, p[0])) - np.log(mx)
                 for _, cfg, mx in EW]), [0.5], method="Nelder-Mead").x[0]
    off = np.mean([np.log(mx) - np.log(base(cfg, s)) for _, cfg, mx in EW])
    print(f"EW seul, meilleur coefficient de secteur (échelle), s={s:.3f}\n")
    preds = []
    for nm, cfg, mx in EW:
        p = base(cfg, s) * np.exp(off); preds.append((nm, p, mx))
        print(f"  {nm:4s} préd={p:8.0f} exp={mx:8.0f} ({(p - mx) / mx * 100:+.0f}%)")
    r = np.sqrt(np.mean([((p - mx) / mx) ** 2 for _, p, mx in preds])) * 100
    opred = [nm for nm, _, _ in sorted(preds, key=lambda t: t[1])]
    oexp = [nm for nm, _, _ in sorted(preds, key=lambda t: t[2])]
    print(f"\n  RMS EW = {r:.0f}%  (plancher : un scalaire ne bouge que la moyenne)")
    print(f"  ordre exp  : {' < '.join(oexp)}")
    print(f"  ordre préd : {' < '.join(opred)}  {'OK' if opred == oexp else '≠ BROUILLÉ'}")
    print(f"\n  Coupables : Φ_F(8_2)={PhiF('8_2'):.0f} < Φ_F(7_1)={PhiF('7_1'):.0f} "
          f"(top<H inversé) ; Φ_F(9_1)={PhiF('9_1'):.0f} gonfle Z.")
    print("  => PAS de coefficient EW (contraste baryons) ; ordre non suivi par la géométrie.")


if __name__ == "__main__":
    main()
