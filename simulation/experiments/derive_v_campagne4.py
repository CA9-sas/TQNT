"""Derivation de la valeur de v (E_relax) : QUATRIEME campagne, routes pre-declarees.

ETAT : v = 246 GeV est identifie (E_relax = vev), calibre (1 des 10 masse).
Trois voies naturelles FERMEES (bloc D) : fluxoide nucleonique (x140),
elastique de twist 8_3 (x10-24), auto-coherence de Compton (degeneree en
echelle). Meilleure consistance croisee existante : quartet EW -> 242 GeV
(-1.7%), via le moteur c_J archive (pas une derivation independante).

CRITERES PRE-DECLARES (avant tout calcul) :
  - DERIVATION : mecanisme nomme AVANT evaluation, zero choix ajustable
    (echelle et structure forcees par le mecanisme), atterrissage < 1 %.
  - VOIE OUVERTE (ordre de grandeur) : 1-15 %, mecanisme partiel.
  - FERMEE : > 15 % ou mecanisme refute.
  - Toute forme dont la motivation EST l'atterrissage (trouvee en scannant)
    -> QUARANTAINE par la regle du null (l'espace de recherche est declare).

ROUTES PRE-DECLAREES (mecanisme d'abord) :
  R1. Peau WKB : E_relax = sqrt(sigma_chi) * e^{2pi}. Mecanisme : la barriere
      de reconnexion = l'inverse de la probabilite de passage P22 au regime
      peau xi=1 (Gamma ~ e^{-2pi}), portee par l'echelle unique du secteur
      masse. RESERVE declaree : xi=1 est le regime 'peau baryonique' dont
      l'application cosmologique a ete refutee ; deux choix implicites
      (echelle, xi) -> au mieux une voie, jamais une derivation.
  R2. Deconfinement : E_relax = E_K * e^{2pi}, E_K = hbar*c/r_link (echelle
      canonique DERIVEE ~200 MeV). Meme mecanisme, echelle de deconfinement.
  R3. Quartet EW : E_relax = sqrt(2) * m(entrelacs le plus lourd du quartet).
      Mecanisme : l'etat qui SATURE la barriere (cout de reconnexion = la
      barriere elle-meme, 'couplage unite') ; le sqrt(2) est celui de Fermi.
      STATUT declare : re-formulation (deplace la question vers la masse du
      top-entrelacs, elle-meme moteur archive) -- consistance, pas derivation.
  R4. Dimension quantique totale : E_relax = sqrt(sigma_chi) * D(SU(2)_135),
      D = 1/S_00. Mecanisme : la barriere comme cout de desintrication dans
      la TQFT de niveau k=135 (le niveau identifie par alpha_EM).
"""
import math

V_MEAS = 246.22            # GeV
SQRT_SIGMA = 0.41765       # GeV
E_K = 0.1993               # GeV (hbar c / r_link, r_link = 0.990 fm)
M_TOP_CAT = 170.41         # GeV (top-entrelacs catalogue)
M_TOP_PDG = 172.57         # GeV
K_CS = 135
F_PI = 0.0921              # GeV


def verdict(dev):
    a = abs(dev)
    if a < 0.01:
        return "< 1%  : candidat DERIVATION (si zero bouton)"
    if a < 0.15:
        return "1-15% : ordre de grandeur, voie ouverte au mieux"
    return "> 15% : FERMEE"


def show(name, v_pred, note=""):
    dev = v_pred / V_MEAS - 1
    print("  %-42s v = %8.2f GeV  (%+6.1f%%)  %s" % (name, v_pred, dev*100, verdict(dev)))
    if note:
        print("      %s" % note)


def run():
    print("=== Valeur de v : 4e campagne (routes pre-declarees, criteres fixes) ===\n")
    print("Cible : v = %.2f GeV ; ratio a deriver v/sqrt(sigma_chi) = %.2f\n" % (
        V_MEAS, V_MEAS/SQRT_SIGMA))

    print("[R1] Peau WKB (barriere = inverse nucleation a xi=1) :")
    show("sqrt(sigma_chi) * e^{2pi}", SQRT_SIGMA * math.exp(2*math.pi),
         "le 2pi est DERIVE (P22 WKB) mais xi=1 est le regime peau non fonde ;")
    print("      2 choix implicites (echelle, xi) -> ordre de grandeur seulement.\n")

    print("[R2] Deconfinement * e^{2pi} :")
    show("E_K * e^{2pi}", E_K * math.exp(2*math.pi))
    print()

    print("[R3] Quartet EW (etat qui sature la barriere, sqrt(2) de Fermi) :")
    show("sqrt(2) * m_top-entrelacs (catalogue)", math.sqrt(2) * M_TOP_CAT,
         "re-formulation : renvoie a la masse du top-entrelacs (moteur archive) ;")
    show("sqrt(2) * m_t (PDG, lecture y_t=1)", math.sqrt(2) * M_TOP_PDG,
         "consistance croisee, PAS une derivation independante (statut declare).")
    print()

    print("[R4] Dimension quantique totale SU(2)_135 (cout de desintrication TQFT) :")
    s = math.sin(math.pi / (K_CS + 2))
    D_tot = math.sqrt((K_CS + 2) / (2 * s * s))
    show("sqrt(sigma_chi) * D(k=135), D=1/S00=%.1f" % D_tot, SQRT_SIGMA * D_tot)
    print()

    print("[QUARANTAINE] Coincidences trouvees en scannant (~30 formes essayees au")
    print("  fil des campagnes ; p(atterrissage<0.2% quelque part) ~ 10%) -- SANS")
    print("  mecanisme, la motivation est l'atterrissage : ECARTEES (regle du null) :")
    for name, val in [
        ("sqrt(sigma_chi)*sqrt(6 pi k(k+2))", SQRT_SIGMA*math.sqrt(6*math.pi*K_CS*(K_CS+2))),
        ("5 e^{2pi} * f_pi", 5*math.exp(2*math.pi)*F_PI),
    ]:
        print("    %-38s -> %7.2f GeV (%+.2f%%)" % (name, val, (val/V_MEAS-1)*100))
    print("    (+ DeltaTw^2*e a +3.3% et K_sigma-Cornell a 2.3%, deja ecartees au bloc D)\n")

    print("VERDICT (criteres pre-declares) : LE MUR TIENT.")
    print("  - Aucune route ne remplit le critere de derivation (<1% avec zero bouton).")
    print("  - R1 donne l'ordre de grandeur (-9%) avec un ingredient derive (le 2pi")
    print("    WKB) mais deux choix non fondes : voie ouverte, pas plus.")
    print("  - R2 (-57%) et R4 (-39%) : FERMEES.")
    print("  - R3 (-2.1% / -0.9%) : la meilleure consistance, mais re-formulation")
    print("    (la barriere <-> le sommet du quartet EW) -- deplace le mur, ne le")
    print("    perce pas.")
    print("  - Les atterrissages exquis (<0.2%) sont exactement le profil des cinq")
    print("    mirages attrapes cette session : quarantaine.")
    print("  v reste le parametre calibre du secteur EW (1 des 10 masse). Total des")
    print("  voies fermees : fluxoide, twist, Compton, deconfinement, dim. quantique.")


if __name__ == "__main__":
    run()
