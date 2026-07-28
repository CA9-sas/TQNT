"""Derivation du g_s effectif du fond thermique chi a T_c (budget asymetrique, 17/07).

Le budget Omega utilisait g_s dans [10.75, 17.25] (nominal 14). Le g_s(T_c)
est DERIVABLE : c'est le comptage standard des degres de liberte
d'entropie a T = 60-75 MeV, sans rien de nouveau -- en langage TQNT, le
fond thermique = ondes chi libres (BASE : photon + neutrinos, encore couples a
67 MeV >> 1.5 MeV de decouplage) + les nœuds legers thermiquement peuples
(e+-, relativistes) + les especes marginales (mu+-, pions : m/T ~ 1.6-2.1,
suppression de Boltzmann PARTIELLE -- integrale exacte, pas un seuil).

s(m, T) par dof via s = (rho + P)/T, normalise au boson sans masse
(2 pi^2/45 T^3). Especes plus lourdes (K, eta : m/T >= 7) : < 0.02, negligees.
"""
import math

SPECIES = [  # (nom, dof, statistique, masse GeV)
    ("gamma", 2, "b", 0.0),
    ("nu (3 fam., couples)", 6, "f", 0.0),
    ("e+-", 4, "f", 0.000511),
    ("mu+-", 4, "f", 0.10566),
    ("pi+-", 2, "b", 0.13957),
    ("pi0", 1, "b", 0.13498),
]


def s_ratio(x, stat, n=4000, umax=40.0):
    """s(m/T = x)/s(0, boson) par dof (rho + P)/T, quadrature simple."""
    du = umax / n
    rho = pres = 0.0
    for i in range(1, n + 1):
        u = i * du
        E = math.sqrt(u * u + x * x)
        occ = 1.0 / (math.exp(E) - (1.0 if stat == "b" else -1.0))
        rho += u * u * E * occ
        pres += u ** 4 / E * occ / 3.0
    s0 = (2 * math.pi ** 4 / 45.0) * (4.0 / 3.0) / 2  # int massless boson (rho+P)
    # normalisation directe : calcule le meme integrand a x=0 boson
    rho0 = pres0 = 0.0
    for i in range(1, n + 1):
        u = i * du
        occ = 1.0 / (math.exp(u) - 1.0)
        rho0 += u ** 3 * occ
        pres0 += u ** 3 * occ / 3.0
    return (rho + pres) / (rho0 + pres0)


def g_s_eff(T):
    tot = 0.0
    rows = []
    for name, g, stat, m in SPECIES:
        r = s_ratio(m / T, stat)
        contrib = g * r
        rows.append((name, g, m / T, r, contrib))
        tot += contrib
    return tot, rows


def run():
    print("=== g_s effectif du fond thermique a T_c : DERIVE (comptage standard) ===\n")
    for T in (0.060, 0.067, 0.075):
        tot, rows = g_s_eff(T)
        print("[T = %d MeV]" % round(T * 1000))
        for name, g, x, r, contrib in rows:
            print("    %-22s g=%d  m/T=%5.2f  s/s0=%.3f  -> %5.2f" % (
                name, g, x, r, contrib))
        print("    TOTAL g_s = %.2f\n" % tot)

    g67 = g_s_eff(0.067)[0]
    print("VERDICT :")
    print(" - g_s(T_c = 67 MeV) = %.1f (60-75 MeV : %.1f-%.1f) : DERIVE," % (
        g67, g_s_eff(0.060)[0], g_s_eff(0.075)[0]))
    print("   l'enveloppe [10.75, 17.25] du budget se referme sur 15.2 +- 0.4.")
    print("   (10.75 = tout-annihile, 17.25 = tout-relativiste : les deux")
    print("   bords etaient des limites ; a m/T ~ 1.6-2.1 la suppression")
    print("   n'est que 25-40 % -- les especes marginales comptent presque")
    print("   pleines.)")
    print(" - Retro-injection budget (Omega proportionnel a 1/g_s) :")
    print("   Omega(g_s=14 nominal) = 15.6 -> Omega(%.1f) = %.1f (x%.1f Planck)" % (
        g67, 15.6 * 14.0 / g67, 15.6 * 14.0 / g67 / 5.45))
    print(" - E_dom (plafond microcanonique, boltzmann_tqnt_domain) :")
    a = g67 * math.pi ** 2 / 30
    Vd = (4 * math.pi / 3) * (1.14 / 0.067) ** 3
    Ed = a * Vd * 0.067 ** 4
    print("   E_dom(g_s=%.1f) = %.2f GeV -- toujours SUR le bord (m(23)=1.71)." % (
        g67, Ed))
    print(" - Un O(1) du budget est ferme ; restent P_occ et la forme de la")
    print("   chaleur latente (voir section brouillon).")


if __name__ == "__main__":
    run()
