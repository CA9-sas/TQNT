"""(b) Tentative m_l / sqrt(sigma_chi) : scan de formes closes AVEC auto-null (17/07).

CIBLES (les deux echelles calibrees restantes du secteur leger) :
  r1 = m_l / sqrt(sigma_chi) = 310/417.65 = 0.74225
  r2 = D.sqrt(sigma_chi)/hbarc = 0.04533  (diametre de tube, equivalent)
  r3 = sqrt(sigma_chi)/T_c = 417.65/67 = 6.2336

DISCIPLINE : tout candidat doit passer le NULL DE FORMES (combien de
formes closes « simples » tombent a la meme precision ?) avant d'etre
presente comme indice -- la regle du null appliquee a nous-memes.
"""
import math
import itertools

TARGETS = {
    "m_l/sqrt(sigma)": 310.0 / 417.65,
    "D*sqrt(sigma)/hbarc": 0.021423 * 417.65 / 197.33,
    "sqrt(sigma)/T_c": 417.65 / 67.0,
}
CANDIDATES = {
    "m_l/sqrt(sigma)": [("3/4 (= 3 Lambda_chi/2 / sqrt)", 0.75)],
    "D*sqrt(sigma)/hbarc": [("2 pi alpha", 2 * math.pi / 137.036),
                            ("1/(7 pi)", 1 / (7 * math.pi))],
    "sqrt(sigma)/T_c": [("2 pi", 2 * math.pi)],
}
FACTORS = {"1": 1.0, "pi": math.pi, "1/pi": 1 / math.pi,
           "pi^2": math.pi ** 2, "1/pi^2": 1 / math.pi ** 2,
           "sqrt2": 2 ** 0.5, "1/sqrt2": 2 ** -0.5,
           "sqrt3": 3 ** 0.5, "e": math.e, "1/e": 1 / math.e,
           "alpha": 1 / 137.036, "1/alpha": 137.036,
           "2pi.alpha": 2 * math.pi / 137.036}


def forms():
    for p, q in itertools.product(range(1, 13), range(1, 13)):
        if math.gcd(p, q) != 1:
            continue
        for fn, fv in FACTORS.items():
            yield ("%d/%d x %s" % (p, q, fn), p / q * fv)


def run():
    print("=== m_l, sqrt(sigma_chi) : formes closes + auto-null ===\n")
    all_forms = list(forms())
    print("famille de formes : %d (p/q <= 12, %d facteurs)\n" % (
        len(all_forms), len(FACTORS)))
    for name, t in TARGETS.items():
        hits = [(n, v, (v / t - 1)) for (n, v) in all_forms
                if abs(v / t - 1) < 0.012]
        print("[%s = %.5f]  formes a <1.2%% : %d" % (name, t, len(hits)))
        for n, v, d in sorted(hits, key=lambda x: abs(x[2]))[:4]:
            print("    %-18s %+.2f %%" % (n, 100 * d))
        for cn, cv in CANDIDATES.get(name, []):
            d = cv / t - 1
            print("    candidat physique : %-22s %+.2f %%" % (cn, 100 * d))
        # significance : densite de formes par % autour de la cible
        dens = sum(1 for (_, v) in all_forms if abs(v / t - 1) < 0.05) / 10.0
        print("    densite locale ~%.1f formes/%% -> P(hasard <1%%) ~ %.0f %%\n" % (
            dens, min(100, 100 * (1 - math.exp(-2 * dens / 1.0)) )))
    print("VERDICT :")
    print(" - AUCUN candidat n'est significatif au null de formes : la densite")
    print("   de coincidences simples est de plusieurs formes par pourcent --")
    print("   3/4 (m_l), 2 pi alpha (D), 2 pi (sqrt(sigma)/T_c) sont des")
    print("   COINCIDENCES tant qu'un mecanisme ne les porte pas.")
    print(" - m_l et sqrt(sigma_chi) restent CALIBRES ; la voie de derivation")
    print("   est physique (profil du tube par le solveur chi / condensat),")
    print("   pas arithmetique. Cartographie consignee, statut OUVERT.")


if __name__ == "__main__":
    run()
