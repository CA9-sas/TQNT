r"""
Recherche combinatoire sur les PARTIES de l'équation de masse (>200 particules).

But (demande utilisateur) : jouer sur la combinatoire des parties de la formule
pour trouver un meilleur RMS sur 200+ particules du PDG.

Formule générale paramétrée (les « parties » recombinables), best-match sur un
catalogue FIXE de 1539 configs (1–3 nœuds, 19 nœuds Abraham) :
    m(cfg) = scale · ((n+1)/2)^{-X} · Π_pair(ρ0·cr^{-B})^{c-1}
             · [ Σ_i (m_naive,i · (b_i det_i)^β)^A ]^{1/A}
             · [ Σ_i U_κ,i^u · U_τ,i^v ]^Q
    m_naive = cr·M_C + n_l·M_L.
11 exposants libres : M_C, M_L, β, A(Hölder), u, v, Q, X, B, ρ0, scale.

RÉSULTAT — le RMS descend très bas, MAIS c'est du surajustement (contrôlé) :
    canonique (exposants FIXES physiques)  : RMS = 24 %
    ajusté au jeu complet (211)            : RMS = 0,32 %   <-- « la formule à 0,3 % »
    NULL (masses ALÉATOIRES, même fit)     : RMS = 4,1 %    (plancher de sur-génération)
    CV 5-fold (test held-out)              : 13 % ± 24  (train 0,8 %)
Le 0,32 % est SOUS le plancher null de 4,1 % → il exploite la flexibilité (11
exposants) pour poser 147 des 1539 configs sur les 211 masses. La CV le prouve :
train 0,8 % → test 13 % (gap 12 points). Best-match sur 1539 configs matche
n'importe quel jeu de ~200 masses à ~4 % (null), donc le RMS absolu bas N'EST PAS
une prédiction. Cohérent avec le §Validation spectre complet (échec 93 % à
assignation FIXE) et [[assignment-freedom-quantified-V6]].

OBSERVATION D'ÉTUDE (le vrai signal) : sous ajustement libre, deux exposants
RESTENT proches de leur valeur physique canonique —
    A (Hölder)  → 0,44 ≈ 1/2  (Koide/Born)
    Q (couplage)→ 0,73 ≈ 3/4  (Casimir spin-1/2)
— tandis que les autres DÉRIVENT vers le non-physique (M_L < 0, β → 0,14, X → −5,7).
Autrement dit : la structure Koide-1/2 + Casimir-3/4 est ROBUSTE (contrainte par
les données même en fit libre), le reste est mou/absorbant. C'est cela qu'il faut
étudier, pas le 0,32 % (artefact).

FORMULE À 0,3 % (paramètres ajustés, pour étude — À NE PAS présenter comme
prédictive) :
    M_C=0,0839  M_L=-0,0136  β=0,145  A=0,437  u=0,559  v=1,580
    Q=0,731  X=-5,708  B=3,202  ρ0=15,90  scale=0,441
    (canonique : 0,148 / 0,0225 / 0,75 / 0,5 / 1 / 1 / 0,75 / -2,45 / 1,65 / 10,6 / 1)
"""
from __future__ import annotations
import itertools
import numpy as np
from collections import Counter
from theory.jones import braid_index, jones_polynomial, jones_determinant
from theory.abraham import REFERENCE_N50K as GEO

NL = {"3_1": 3, "4_1": 2, "5_1": 5, "5_2": 2, "6_1": 2, "6_2": 3, "6_3": 3,
      "7_1": 7, "7_2": 2, "7_3": 3, "7_4": 2, "7_5": 3, "7_6": 2, "7_7": 3,
      "8_1": 2, "8_2": 3, "8_3": 3, "9_1": 9, "10_1": 2}
D = {K: dict(cr=int(K.split("_")[0]), nl=NL[K], b=braid_index(K),
             det=jones_determinant(jones_polynomial(K)),
             Uk=GEO[K]["int_kappa"] ** 2 / GEO[K]["L"],
             Ut=GEO[K]["int_abs_tau"] ** 2 / (0.5 * GEO[K]["L"])) for K in GEO}
CFG = [c for n in (1, 2, 3) for c in itertools.combinations_with_replacement(list(GEO), n)]

# La formule à 0,3 % (ajustée) et la canonique (exposants physiques fixes).
FIT_03 = [0.0839, -0.0136, 0.1451, 0.4369, 0.5588, 1.5799,
          0.7311, -5.7082, 3.2022, 15.9023, 0.4410]
CANON = [0.148, 0.0225, 0.75, 0.5, 1.0, 1.0, 0.75, -2.45, 1.65, 10.6, 1.0]

# masses PDG (>200) : (nom, MeV) — voir docstring du dépôt pour la liste complète.
PDG_MASS = None   # rempli par _load_pdg() ; liste embarquée ci-dessous


def masses(p):
    MC, ML, beta, A, u, v, Q, X, B, rho0, scale = p
    out = []
    for comp in CFG:
        n = len(comp); cnt = Counter(comp)
        try:
            ws = [(D[k]["cr"] * MC + D[k]["nl"] * ML) * ((D[k]["b"] * D[k]["det"]) ** beta)
                  for k in comp]
            if any(w <= 0 for w in ws):
                out.append(np.nan); continue
            sk = sum(w ** A for w in ws) ** (1 / A)
            G = sum(D[k]["Uk"] ** u * D[k]["Ut"] ** v for k in comp)
            pair = 1.0
            for k, c in cnt.items():
                if c >= 2:
                    pair *= (rho0 * D[k]["cr"] ** (-B)) ** (c - 1)
            out.append(scale * ((n + 1) / 2) ** (-X) * pair * sk * G ** Q)
        except Exception:
            out.append(np.nan)
    return np.array(out)


def rms_bestmatch(p, mass):
    mc = masses(p); mc = mc[np.isfinite(mc) & (mc > 0)]
    if len(mc) < 50:
        return 9.9
    return float(np.sqrt(np.mean(
        np.min(np.abs(np.log(mass)[:, None] - np.log(mc)[None, :]), axis=1) ** 2)))


PDG_MASS = np.array([
    0.511, 105.658, 1776.86, 134.977, 139.57, 493.677, 497.611, 547.862, 957.78, 1864.84, 1869.66, 1968.34,
    5279.34, 5279.66, 5366.92, 6274.9, 775.26, 782.65, 1019.461, 891.66, 3096.9, 3686.1, 3773.7, 4039.0,
    4191.0, 4415.0, 9460.3, 10023.26, 10355.2, 10579.4, 10885.2, 11000.0, 2006.85, 2010.26, 2112.2, 5324.71,
    980.0, 500.0, 990.0, 1370.0, 1506.0, 1704.0, 1230.0, 1229.5, 1272.0, 1281.9, 1426.4, 1166.0,
    1425.6, 1318.0, 1275.5, 1517.4, 1474.0, 1706.0, 2983.9, 3637.5, 3525.4, 3414.71, 3510.67, 3556.17,
    9398.7, 9999.0, 9899.3, 10259.8, 9859.44, 9892.78, 9912.21, 10232.5, 10255.46, 10268.65, 10163.7, 938.272,
    939.565, 1115.683, 1189.37, 1192.642, 1197.449, 1314.86, 1321.71, 1232.0, 1232.0, 1232.0, 1232.0, 1382.83,
    1383.7, 1387.2, 1531.8, 1535.0, 1672.45, 1440.0, 1515.0, 1535.0, 1655.0, 1675.0, 1685.0, 1720.0,
    1710.0, 1720.0, 1875.0, 1900.0, 2180.0, 1600.0, 1620.0, 1700.0, 1880.0, 1910.0, 1920.0, 1930.0,
    1405.1, 1519.0, 1600.0, 1670.0, 1690.0, 1800.0, 1810.0, 1820.0, 1830.0, 1890.0, 1660.0, 1670.0,
    1750.0, 1775.0, 1915.0, 1690.0, 1823.0, 1950.0, 2025.0, 2286.46, 2453.97, 2518.0, 2467.94, 2470.9,
    2695.2, 2592.25, 2628.11, 2856.1, 2881.63, 2939.6, 2801.0, 2578.0, 2645.9, 2792.0, 2816.7, 2969.4,
    2765.9, 3621.6, 5619.6, 5811.0, 5832.5, 5791.9, 5797.0, 6045.2, 5912.2, 5920.1, 5935.0, 5949.0,
    6227.0, 6316.0, 1300.0, 1294.0, 1409.8, 1475.0, 1465.0, 1720.0, 1410.0, 1670.0, 1680.0, 1414.0,
    1718.0, 1773.0, 1819.0, 1776.0, 1688.8, 1667.0, 1854.0, 1967.0, 2018.0, 1815.0, 2045.0, 3871.69,
    3918.4, 4146.5, 4274.0, 4222.0, 4374.0, 4630.0, 3886.6, 4024.1, 4478.0, 10607.2, 10652.2, 4311.9,
    4380.0, 4449.8, 4337.0, 4458.8, 3875.0, 6900.0, 3700.0, 80369.0, 91188.0, 125250.0, 173100.0, 2100.0,
    2030.0, 3055.9, 3000.4, 2250.0, 2280.0, 2420.0, 2020.0,
])


def main():
    print(f"PDG : {len(PDG_MASS)} particules ; catalogue : {len(CFG)} configs")
    print(f"  canonique (exposants fixes) : RMS = {rms_bestmatch(CANON, PDG_MASS) * 100:.1f}%")
    print(f"  formule à 0,3 % (ajustée)   : RMS = {rms_bestmatch(FIT_03, PDG_MASS) * 100:.2f}%")
    rng = np.random.default_rng(100)
    lo, hi = np.log(PDG_MASS.min()), np.log(PDG_MASS.max())
    nulls = [rms_bestmatch(FIT_03, np.exp(rng.uniform(lo, hi, len(PDG_MASS)))) for _ in range(3)]
    print(f"  NULL (FIT_03 sur masses aleatoires) : RMS = {np.mean(nulls) * 100:.1f}%")
    print("  (plancher null RE-AJUSTE = 4,1 % ; CV test held-out = 13 %.)")
    print("  => le 0,32 % est du surajustement : voir docstring pour les 4 controles.")


if __name__ == "__main__":
    main()
