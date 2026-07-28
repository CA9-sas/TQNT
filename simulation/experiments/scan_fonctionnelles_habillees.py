"""Scan des fonctionnelles geometriques pour l'habillage m_s (+5,6 %) (22/07).

CAHIER DES CHARGES (fonctionnelle habillee m = sigma*l*(1 + beta*f), f=F/l) :
  (a) f PLAT sur la famille torique 3_1, 5_1, 7_1, 9_1 (le reservoir DM
      T(2,q) et la calibration m_l doivent survivre au rescaling commun) ;
  (b) f(5_2) - f(3_1) tel que le ratio effectif passe de 1,5108 a 1,5961
      (+5,65 %) avec un coefficient beta NATUREL (mecanisme, regle du null) ;
  (c) predictions : les autres non-toriques (4_1, 6_1, 6_3, 7_2, 8_3)
      recoivent leur habillage -> etats stricts DM, beta_x.

Fonctionnelles mesurees sur les formes ideales katlas (Fourier, analytiques,
ACPR a ~0,01 %) : courbure totale, courbure carree, torsion totale (les
formes sont lisses -- l'obstacle 'segments droits' des polylignes SONO
n'existe pas ici), |torsion|, writhe (Gauss), multiplicite de contact.
NB : l'echec de la fonctionnelle de CONTACT (contact ~100 % partout,
measure_contact_sets.py) est acquis -- ici on cherche les differentiels
de FORME, pas de contact.
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
WANT = {"3_1": "3:1:1", "5_1": "5:1:1", "7_1": "7:1:1", "9_1": "9:1:1",
        "4_1": "4:1:1", "5_2": "5:1:2", "6_1": "6:1:1", "6_3": "6:1:3",
        "7_2": "7:1:2", "8_3": "8:1:3"}
TORUS = ["3_1", "5_1", "7_1", "9_1"]
ML, MS = 310.0, 494.8
N = 4096


def load(text, kid):
    m = re.search(r'<AB Id="%s"[^>]*>(.*?)</AB>' % re.escape(kid), text, re.S)
    body = m.group(1)
    A = [[float(x) for x in c.group(1).split(",")] for c in
         re.finditer(r'<Coeff I="\s*\d+" A="([^"]*)" B="([^"]*)"', body)]
    B = [[float(x) for x in c.group(2).split(",")] for c in
         re.finditer(r'<Coeff I="\s*\d+" A="([^"]*)" B="([^"]*)"', body)]
    return np.array(A), np.array(B)


def frames(A, B, n=N):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = np.arange(1, len(A) + 1)
    ct, st = np.cos(np.outer(t, k)), np.sin(np.outer(t, k))
    P = ct @ A + st @ B
    d1 = (-st * k) @ A + (ct * k) @ B
    d2 = (-ct * k**2) @ A + (-st * k**2) @ B
    d3 = (st * k**3) @ A + (-ct * k**3) @ B
    return P, d1, d2, d3


def writhe(P):
    dP = np.roll(P, -1, 0) - P
    M = P + 0.5 * dP
    R = M[:, None, :] - M[None, :, :]
    nr = np.linalg.norm(R, axis=2)
    np.fill_diagonal(nr, np.inf)
    cr = np.cross(dP[:, None, :], dP[None, :, :])
    integ = (cr * R).sum(axis=2) / nr**3
    return integ.sum() / (4 * np.pi)


def contact_mult(P, s, L):
    """multiplicite moyenne : nb d'arcs distincts en contact par point."""
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    darc = np.abs(s[:, None] - s[None, :])
    darc = np.minimum(darc, L - darc)
    D[darc < 1.6] = np.inf
    touch = D <= 1.02
    mult = np.zeros(len(P))
    for i in range(len(P)):
        j = np.where(touch[i])[0]
        if len(j) == 0:
            continue
        # compte les blocs contigus d'indices en contact
        mult[i] = 1 + np.sum(np.diff(j) > 5)
    return mult.mean()


def main():
    text = _gzip.open(IDEAL, "rt").read()
    rows = {}
    for name, kid in WANT.items():
        A, B = load(text, kid)
        P, d1, d2, d3 = frames(A, B)
        sp = np.linalg.norm(d1, axis=1)
        dt = 2 * np.pi / N
        ds = sp * dt
        L = ds.sum()
        s = np.concatenate([[0.0], np.cumsum(ds)])[:-1]
        cr = np.cross(d1, d2)
        ncr = np.linalg.norm(cr, axis=1)
        kappa = ncr / sp**3
        tau = np.einsum("ij,ij->i", cr, d3) / np.maximum(ncr**2, 1e-30)
        lowk = kappa < 0.05                      # zones tau mal defini
        F = {
            "totk":  (kappa * ds).sum(),                        # rad
            "k2":    (kappa**2 * ds).sum(),
            "tottau": (tau * ds).sum(),                         # signe
            "abst":  (np.abs(tau) * ds).sum(),
            "Wr":    abs(writhe(P[::4])),
            "mult":  contact_mult(P[::4], s[::4], L),
        }
        rows[name] = (2 * L, F, lowk.mean())
    tgt = MS / ML                                                # 1.59613
    l3, l5 = rows["3_1"][0], rows["5_2"][0]
    bare = l5 / l3
    print("nœud   RL(ray)   totκ/2π    ∮κ²/ℓ    ∮τ/2π    ∮|τ|/ℓ    |Wr|   mult  (κ<0,05)")
    for n_, (L, F, fl) in rows.items():
        print("%-4s %9.3f %8.4f %9.4f %8.4f %8.4f %7.3f %6.2f  %4.2f" %
              (n_, L, F["totk"] / (2 * np.pi), F["k2"] / L,
               F["tottau"] / (2 * np.pi), F["abst"] / L, F["Wr"],
               F["mult"], fl))
    print("\nCible : ratio_eff = %.5f (nu %.5f, boost +%.2f %%)"
          % (tgt, bare, 100 * (tgt / bare - 1)))
    print("=" * 78)
    print("TEST par fonctionnelle : m = sigma*(l + b*F) -> b requis, et check")
    print("torique (f=F/l relatif au 3_1 ; il faut ~0 pour 5_1/7_1/9_1)")
    print("=" * 78)
    for key in ("totk", "k2", "tottau", "abst", "Wr", "mult"):
        F3 = rows["3_1"][1][key]
        F5 = rows["5_2"][1][key]
        den = F5 - tgt * F3            # (l5+bF5)/(l3+bF3)=tgt -> b
        b = (tgt * l3 - l5) / den if abs(den) > 1e-12 else np.inf
        # dispersion torique de f = F/l (normalisee au 3_1)
        fs = [rows[t][1][key] / rows[t][0] for t in TORUS]
        f0 = fs[0]
        disp = ", ".join("%+.1f%%" % (100 * (x / f0 - 1)) for x in fs[1:])
        # boost qu'infligerait b aux toriques lourds (9_1)
        print("%-6s : F3=%9.4f F5=%9.4f  b=%9.4f | f/f(3_1) toriques : %s"
              % (key, F3, F5, b, disp))


if __name__ == "__main__":
    main()
