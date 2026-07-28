"""Mesure des ensembles d'auto-contact des noeuds ideaux (22/07).

BUT : tester la fonctionnelle habillee m = sigma*(l - gamma*l_contact)
(mecanisme : partage du champ chi au contact du tube avec lui-meme --
le meme principe que le tube commun de la demi-substitution mesonique).
Il faut +5,65 % sur le ratio 5_2/3_1 : le 5_2 doit avoir PLUS d'arc
LIBRE (hors contact) que le 3_1, en proportion.

DONNEES : base Gilbert/katlas 'Ideal knots' (SONO, series de Fourier,
D=1) -- verifiee contre ACPR a -0,005 % sur 3_1 et 5_2. Le tube a un
rayon 0,5 en unites fichier ; longueurs converties en RAYONS (x2).

CRITERE DE CONTACT : d_min(s) <= 1+eps (diametre), paires non locales
(distance d'arc > 1,6 ~ pi*rho_min) ; les arcs a courbure saturee
(kappa >= 0,95*2) sont signales a part (kinks).
"""
import re
import numpy as np

import gzip as _gzip
import os as _os
IDEAL = _os.path.join(_os.path.dirname(__file__), "..", "..",
                      "docs", "references",
                      "katlas_ideal_knots_gilbert.txt.gz")
# base 'Ideal knots' de Brian Gilbert (katlas.org/wiki/Ideal_knots),
# SONO 512 pts en series de Fourier ; verifiee contre ACPR a ~0,01 %
ACPR = {"3_1": 32.7448, "4_1": 42.0928, "5_1": 47.2016, "5_2": 49.4704,
        "6_1": 56.7150, "6_3": 57.8435, "7_1": 61.4109, "7_2": 63.8956,
        "8_3": 71.1655, "9_1": 75.7252}
WANT = {"3_1": "3:1:1", "4_1": "4:1:1", "5_1": "5:1:1", "5_2": "5:1:2",
        "6_1": "6:1:1", "6_3": "6:1:3", "7_1": "7:1:1", "7_2": "7:1:2",
        "8_3": "8:1:3", "9_1": "9:1:1"}
ML, MS_DATA = 310.0, 494.8
N = 4096


def load_fourier(text, kid):
    m = re.search(r'<AB Id="%s"[^>]*>(.*?)</AB>' % re.escape(kid),
                  text, re.S)
    if m is None:                      # bloc sans fermeture explicite ?
        m = re.search(r'<AB Id="%s"[^>]*>(.*?)<AB ' % re.escape(kid),
                      text, re.S)
    body = m.group(1)
    A, B = [], []
    for c in re.finditer(r'<Coeff I="\s*\d+" A="([^"]*)" B="([^"]*)"', body):
        A.append([float(x) for x in c.group(1).split(",")])
        B.append([float(x) for x in c.group(2).split(",")])
    return np.array(A), np.array(B)


def curve(A, B, n=N):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = np.arange(1, len(A) + 1)
    ct = np.cos(np.outer(t, k))        # (n, K)
    st = np.sin(np.outer(t, k))
    P = ct @ A + st @ B
    dP = (-st * k) @ A + (ct * k) @ B
    d2P = (-ct * k**2) @ A + (-st * k**2) @ B
    return P, dP, d2P


def analyze(name, kid, text):
    A, B = load_fourier(text, kid)
    P, dP, d2P = curve(A, B)
    sp = np.linalg.norm(dP, axis=1)                    # |r'|
    dt = 2 * np.pi / N
    ds = sp * dt
    L = ds.sum()                                       # unites D=1
    cr = np.cross(dP, d2P)
    kappa = np.linalg.norm(cr, axis=1) / sp**3
    # abscisse curviligne
    s = np.concatenate([[0.0], np.cumsum(ds)])[:-1]
    # distances non locales (exclusion d'arc 1.6, tore d'arc)
    D2 = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    darc = np.abs(s[:, None] - s[None, :])
    darc = np.minimum(darc, L - darc)
    D2[darc < 1.6] = np.inf
    dmin = D2.min(axis=1)
    rl_rad = 2 * L                                     # rayons (R=1)
    out = {"L_rad": rl_rad, "dmin": dmin, "ds": ds, "kappa": kappa}
    print("%-4s  RL=%8.4f rayons (ACPR %8.4f, %+.3f %%)   gap min %.4f" %
          (name, rl_rad, ACPR[name], 100 * (rl_rad / ACPR[name] - 1),
           dmin.min()))
    return out


def main():
    text = _gzip.open(IDEAL, "rt").read()
    res = {k: analyze(k, v, text) for k, v in WANT.items()}

    print("\n" + "=" * 78)
    print("FRACTIONS DE CONTACT (d_min <= 1+eps) et arcs libres (rayons)")
    print("=" * 78)
    print("%-5s %9s | %s | %s" %
          ("noeud", "RL", "  ".join("fc(%d%%)" % e for e in (1, 2, 5)),
           "L_libre(2%) rayons   kink5%"))
    for name, r in res.items():
        fcs = []
        for eps in (0.01, 0.02, 0.05):
            mask = r["dmin"] <= 1.0 + eps
            fcs.append((r["ds"][mask].sum() / (r["L_rad"] / 2)))
        free = r["L_rad"] * (1 - fcs[1])
        kink = r["ds"][r["kappa"] >= 0.95 * 2.0].sum() / (r["L_rad"] / 2)
        print("%-5s %9.4f |  %5.3f   %5.3f   %5.3f |   %7.3f        %5.3f"
              % (name, r["L_rad"], *fcs, free, kink))

    print("\n" + "=" * 78)
    print("TEST DE LA FONCTIONNELLE m = sigma*(l - gamma*l_c)")
    print("=" * 78)
    tgt = MS_DATA / ML                                  # 1.59613
    for eps in (0.01, 0.02, 0.05):
        l3, l5 = res["3_1"]["L_rad"], res["5_2"]["L_rad"]
        c3 = 2 * res["3_1"]["ds"][res["3_1"]["dmin"] <= 1 + eps].sum()
        c5 = 2 * res["5_2"]["ds"][res["5_2"]["dmin"] <= 1 + eps].sum()
        den = tgt * c3 - c5
        gamma = (tgt * l3 - l5) / den if abs(den) > 1e-9 else np.nan
        print("  eps=%d%% : l_c(3_1)=%7.3f  l_c(5_2)=%7.3f  -> gamma=%8.4f"
              % (100 * eps, c3, c5, gamma))
    print("""
LECTURE : gamma est la fraction d'energie de tube economisee par unite
d'arc en contact. Candidats a mecanisme : 1/2 (partage du tube, comme
la demi-substitution), 1/3, 1/4. Si gamma sort O(1) plausible ET si les
toriques (5_1, 7_1, 9_1) ont une fraction de contact ~egale a celle du
3_1 (rescaling uniforme absorbe par la calibration m_l), la
fonctionnelle est viable ; sinon, echec declare.""")


if __name__ == "__main__":
    main()
