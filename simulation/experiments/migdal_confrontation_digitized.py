"""Confrontation Migdal du bord du reservoir sur COURBES NUMERISEES (16/07).

CIBLE CANONIQUE (budget asymetrique) : reservoir domine par le bord
c = 23-25 (fractions 33.7/22.3/14.7 %), M = 1.71/1.78/1.85 GeV,
sigma(c) = sigma_EM * alpha^(c-18) avec sigma_EM = 1e-25 nominal
(echelle Thomson) / 1e-28 conservateur (marge -3 dex declaree).

COURBES NUMERISEES (lecture visuelle des figures publiees, +-0.3 dex) :
  - DarkSide-50 2022 (arXiv:2207.11966, Fig. 11 NQ, ANCRAGE abstract :
    sigma_SI(3 GeV) = 6e-43 cm^2) -- meilleure limite mondiale 1.2-3.6 GeV.
  - XENON1T Migdal 2019 (arXiv:1907.12771) -- secondaire a ces masses.
  - CRESST-III 2019 -- secondaire.
Les limites publiees supposent f_DM = 1 : pour une sous-fraction f, la
limite effective est sigma_lim / f.

VERIFICATION DE PENETRATION (LNGS, 3800 mwe) : colonne ~3.8e5 g/cm^2 ;
collisions = n_noyaux * sigma_A(coherent). L'attenuation ne mord qu'a
sigma_p >~ 1e-31 cm^2 : les limites souterraines s'appliquent a tout le
domaine teste ici.
"""
import math

ALPHA = 1 / 137.036
# tranches canoniques du bord (budget asymetrique)
EDGE = [(23, 1.71, 0.337), (24, 1.78, 0.223), (25, 1.85, 0.147),
        (26, 1.92, 0.098), (27, 2.00, 0.065)]

# DS-50 2022 NQ, numerisation Fig. 11 (ancree 6e-43 @ 3 GeV)
DS50 = [(1.2, 3e-40), (1.4, 4e-41), (1.6, 1.2e-41), (1.8, 5e-42),
        (2.0, 2.5e-42), (2.5, 1.1e-42), (3.0, 6e-43), (4.0, 4.5e-43)]


def lim_ds50(M):
    import bisect
    xs = [math.log10(a) for a, _ in DS50]
    ys = [math.log10(b) for _, b in DS50]
    x = math.log10(M)
    if x <= xs[0]:
        return 10 ** ys[0]
    i = bisect.bisect(xs, x) - 1
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return 10 ** (ys[i] + t * (ys[i + 1] - ys[i]))


def sigma_c(c, sig_em):
    return sig_em * ALPHA ** (c - 18)


def penetration_check(sig_p, M):
    """Collisions dans 3.8e5 g/cm^2 de roche (A~24, coherent A^2 mu^2)."""
    A = 24.0
    mu_p = M * 0.938 / (M + 0.938)
    mu_A = M * A * 0.931 / (M + A * 0.931)
    sig_A = sig_p * A ** 2 * (mu_A / mu_p) ** 2
    n_col = 3.8e5 / (A * 1.66e-24) * sig_A
    return n_col


def run():
    print("=== Confrontation Migdal, courbes numerisees (DS-50 2022) ===\n")
    print("[0] Penetration LNGS (collisions dans la colonne, doit etre <~ qq) :")
    for sig in (1e-36, 1e-33, 1e-31):
        print("    sigma_p = %.0e : N_coll = %.1e %s" % (
            sig, penetration_check(sig, 1.71),
            "(limites applicables)" if penetration_check(sig, 1.71) < 3 else "(BLOQUE avant le detecteur)"))
    print()

    for lab, sig_em in (("NOMINAL sigma_EM = 1e-25 (Thomson)", 1e-25),
                        ("CONSERVATEUR sigma_EM = 1e-28", 1e-28),
                        ("PINCE sigma_EM = 1e-30 (neutralite)", 1e-30)):
        print("[%s]" % lab)
        print("    c    M(GeV)  f      sigma      lim_DS50   lim/f      verdict")
        excl = 0.0
        for c, M, f in EDGE:
            sig = sigma_c(c, sig_em)
            lim = lim_ds50(M)
            eff = lim / f
            dex = math.log10(sig / eff)
            v = "EXCLU (+%.1f dex)" % dex if dex > 0.3 else (
                "TENSION (%+.1f dex)" % dex if dex > -0.3 else "compatible (%+.1f dex)" % dex)
            if dex > 0.3:
                excl += f
            print("    %d  %5.2f  %5.3f  %9.1e  %9.1e  %9.1e  %s" % (
                c, M, f, sig, lim, eff, v))
        print("    -> fraction du reservoir exclue : %.0f%%\n" % (100 * excl))

    print("VERDICT :")
    print(" - Sous DS-50 2022 (meilleure limite mondiale 1.2-3.6 GeV, ancree")
    print("   6e-43@3GeV), le bord NOMINAL (Thomson 1e-25) est EXCLU a ~5 dex")
    print("   (c=23), et meme le CONSERVATEUR (1e-28) laisse c=23 exclu (+2 dex).")
    print(" - MAIS la normalisation Thomson est inadaptee : les noeuds du")
    print("   reservoir sont NEUTRES (Q=0) -- leur couplage EM residuel passe")
    print("   par la polarisabilite/rayon de charge, tres au-dessous de Thomson")
    print("   (cf. neutron). La confrontation PINCE la normalisation :")
    print("   sigma_EM(effectif, neutre) <= ~1e-30 cm^2 pour que le bord survive.")
    print(" - A sigma_EM = 1e-30 : c=23 tombe EXACTEMENT sur la limite DS-50")
    print("   actuelle (2.1e-41 vs 2.1e-41) -- le bord vit A LA FRONTIERE de")
    print("   sensibilite : cible immediate de DS-20k / XLZD (Migdal).")
    print(" - Caveats : numerisation +-0.3 dex ; halo standard (rho=0.3,")
    print("   v0=238 km/s) ; couplage SI equivalent (ordre) ; fractions du")
    print("   budget (P_occ etc. O(1)).")


if __name__ == "__main__":
    run()
