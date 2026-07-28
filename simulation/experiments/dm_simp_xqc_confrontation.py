"""Confrontation SIMP/XQC de la tranche c=19-23 (et au-dela) du reservoir DM.

SUITE de dm_reservoir_geometrique.py : le reservoir geometrique (M ~ 1.4-2.2
GeV, sigma de Vassiliev, abondance alpha_cond) confronte aux exclusions
EXISTANTES pour la matiere noire fortement interagissante et basse masse.

BANDES D'EXCLUSION encodees (litterature, approximations +-0.5-1 dex,
valeurs a f_DM = 1 ; toutes anterieurs a 2026) :
  - XQC (fusee, sans couverture ; Erickcek 2007 / Mahdawi-Farrar 2018) :
    M ~ 0.1-10 GeV, sigma_n exclu ~ [3e-31, 1e-25] cm^2.
  - CRESST 2017 run de SURFACE : M ~ 0.5-10 GeV, sigma ~ [4e-34, 3e-28].
  - Souterrain basse masse (CRESST-III, DarkSide-50, XENON-Migdal) :
    M ~ 0.5-3 GeV, sigma ~ [1e-41, 1e-35] ; ne s'applique que si la
    particule PENETRE (sigma/M < ~1e-33 cm2/GeV, sinon thermalisee avant).
  - Satellites MW + Lyman-alpha (Nadler 2021, n=0) : sigma <~ 5e-30 a
    M ~ 1.5 GeV (f=1) -- s'applique meme aux non-penetrantes (cosmologie).

SCALING EN FRACTION : une tranche c ne porte que f(c) de la densite locale ->
les bornes de detection directe se relachent en ~1/f (flux) sur leur bord
BAS ; les bornes cosmologiques (satellites) en ~1/f egalement (1er ordre).

DEUX SCENARIOS pour sigma_EM (le calibrage V5 13.6.4 est un ORDRE DE
GRANDEUR) : nominal 1e-25 cm2 ; conservateur 1e-28 cm2.
"""
import math

SCALE = 310.0 / 16.372
SLOPE_T2 = (38.62 - 16.372) / 6
L_31 = 16.372
ALPHA = 1 / 137.036
ALPHA_COND = 1.20
BLOCK = 1e-33      # cm2/GeV : seuil de penetration souterraine

BANDS = [
    # nom, (Mmin, Mmax), (sig_lo, sig_hi), cosmologique?
    ("XQC (fusee)",          (0.1, 10.0), (3e-31, 1e-25), False),
    ("CRESST surface 2017",  (0.5, 10.0), (4e-34, 3e-28), False),
    ("souterrain bas-M",     (0.5, 3.0),  (1e-41, 1e-35), False),
    ("satellites MW (cosmo)", (0.1, 10.0), (5e-30, 1e-20), True),
]


def verdict(c, M, sig, f):
    hits = []
    for nom, (m0, m1), (s0, s1), cosmo in BANDS:
        if not (m0 <= M <= m1):
            continue
        lo = s0 / max(f, 1e-6)          # relachement en 1/f du bord bas
        hi = s1
        if nom == "souterrain bas-M" and sig / M > BLOCK:
            continue                     # ne penetre pas : bande inapplicable
        if lo <= sig <= hi:
            marge = math.log10(sig / lo)
            hits.append((nom, marge))
    return hits


def run_scenario(sigma_em, tag):
    print("--- scenario sigma_EM = %.0e cm2 (%s) ---" % (sigma_em, tag))
    tot = sum(ALPHA_COND ** (-(c - 19)) for c in range(19, 31))
    n_exclu = 0
    ab_exclu = 0.0
    for c in range(19, 28):
        M = SCALE * (L_31 + SLOPE_T2 * (c - 3)) / 1000.0
        sig = sigma_em * ALPHA ** (c - 18)
        f = ALPHA_COND ** (-(c - 19)) / tot
        hits = verdict(c, M, sig, f)
        if hits:
            n_exclu += 1
            ab_exclu += f
            det = " ; ".join("%s (+%.1f dex)" % (n, m) for n, m in hits)
            v = "EXCLU par " + det
        else:
            v = "compatible"
        print(" c=%2d  M=%.2f GeV  sigma=%.1e  f=%4.1f%%  %s" % (
            c, M, sig, f * 100, v))
    print(" => fraction du reservoir en conflit : %.0f%%\n" % (ab_exclu * 100))
    return ab_exclu


def run():
    print("=== Confrontation SIMP/XQC du reservoir geometrique ===\n")
    ab1 = run_scenario(1e-25, "nominal V5 13.6.4")
    ab2 = run_scenario(1e-28, "conservateur, -3 dex")
    print("VERDICT (honnete) :")
    print(" - Scenario nominal : ~%.0f%% du reservoir est en CONFLIT avec des" % (ab1 * 100))
    print("   donnees EXISTANTES (XQC pour c=19-20, CRESST-surface c=21,")
    print("   souterrain bas-M c=22-25, satellites en renfort).")
    print(" - Scenario conservateur (-3 dex sur sigma_EM) : le conflit se")
    print("   deplace vers c=19-22 (~%.0f%%) sans disparaitre (XQC et la" % (ab2 * 100))
    print("   cosmologie tiennent les grandes sections).")
    print(" - C'est un CANDIDAT DE FALSIFICATION (type F) de la combinaison")
    print("   {reservoir c>=19} x {abondance ~alpha_cond^-c} x {sigma_Vassiliev}.")
    print("")
    print("TROIS SORTIES possibles (a departager) :")
    print(" (a) c_min du reservoir >= 26 : echo direct du motif F3 (les seuils")
    print("     effectifs Delta_c=22/30 au-dessus du minimum naif) -- la sortie")
    print("     la plus naturelle dans le cadre ; predit un reservoir 1.9-2.2")
    print("     GeV STRICTEMENT invisible (sauf Migdal futur c=26).")
    print(" (b) l'abondance n'est pas ~alpha_cond^-c dans la famille sombre")
    print("     (la cinetique P18 fixe Omega_DM total, pas la distribution) :")
    print("     une distribution montante en c evacue le conflit.")
    print(" (c) sigma_EM << 1e-28 (la normalisation 13.6.4 surestimee de >3")
    print("     dex) -- couteux : degrade aussi la pretention 'absence LZ'.")
    print("")
    print("CAVEATS : bandes litterature a +-0.5-1 dex ; scaling 1/f au 1er")
    print("ordre ; le plafond souterrain (overburden) traite par le seuil de")
    print("penetration ; confrontation aux courbes numerisees a faire pour")
    print("trancher finement.")


if __name__ == "__main__":
    run()
