"""Chiffrer la prediction << Gamma commun decoherence/dissipation >> (20/07).

Banc propre : NANOSPHERE LEVITEE (optomecanique). La decoherence d'une
superposition spatiale et le chauffage sont le MEME coefficient de
diffusion d'impulsion D_p, mesurable par deux instruments independants :

    Gamma_dec(Dx) = D_p Dx^2 / hbar^2      (visibilite interferometrique)
    Gamma_th      = D_p / (2 m hbar w)      (rechauffage, phonons/s, PSD)
    x_zpf^2 = hbar/(2 m w)  =>  Gamma_dec(x_zpf) = Gamma_th  EXACTEMENT.

Contenu TQNT propre (au-dela de l'identite, heritee de la mecanique
quantique des mesures efficientes) :
 (a) zeta = Gamma_dec(x_zpf)/Gamma_th = 1 par ONTOLOGIE : chaque
     diffusion d'impulsion environnementale est un contact ENREGISTRE
     (reconnexion) -- apres soustraction du bruit classique (qui
     chauffe sans decoherer, zeta < 1) ; zeta > 1 interdit ;
 (b) RESIDUEL STRICTEMENT NUL apres comptabilite des canaux
     (lambda_GRW = lambda_CSL = 0) -- chiffre contre les planchers
     de chauffage CSL (GRW et Adler).

Caveat honnete (conducteurs) : tau_p != tau_phi (localisation faible)
ne refute PAS l'identite -- un diffuseur STATIQUE n'a pas de partenaire
dynamique (pas de reconnexion, pas d'enregistrement, pas d'entropie a
cet evenement) ; l'identite par evenement vit aux contacts INELASTIQUES,
et q_dot = rho J^2 est le bilan de regime permanent.
"""
import math

HBAR = 1.054571817e-34
KB = 1.380649e-23
C = 2.99792458e8
AMU = 1.66053906660e-27

# ---------------- banc de reference : nanosphere silice ----------------
R = 71.5e-9            # rayon (Delic 2020)
RHO = 2200.0           # silice kg/m^3
LAM = 1064e-9          # pince optique
P_TWEEZER = 0.2        # W
W0 = 0.6e-6            # col du faisceau
OMEGA_M = 2 * math.pi * 1.0e5   # mode mecanique 100 kHz
EPS = 2.1              # permittivite silice a 1064 nm
T_GAS = 300.0
P_GAS = 1e-6           # Pa (= 1e-8 mbar)
M_GAS = 28 * AMU       # N2

M = RHO * 4 / 3 * math.pi * R**3
X_ZPF = math.sqrt(HBAR / (2 * M * OMEGA_M))


def d_recoil():
    """Diffusion d'impulsion par recul de photons (Rayleigh)."""
    k = 2 * math.pi / LAM
    alpha_p = R**3 * (EPS - 1) / (EPS + 2)          # alpha/(4 pi eps0)
    sigma = (8 * math.pi / 3) * k**4 * alpha_p**2   # section Rayleigh
    i0 = 2 * P_TWEEZER / (math.pi * W0**2)
    p_scat = sigma * i0
    n_dot = p_scat / (HBAR * k * C)                 # photons diffuses /s
    return 0.4 * (HBAR * k) ** 2 * n_dot, p_scat    # facteur dipolaire 2/5


def d_gas():
    """Diffusion d'impulsion par collisions de gaz (regime moleculaire)."""
    n = P_GAS / (KB * T_GAS)
    vbar = math.sqrt(8 * KB * T_GAS / (math.pi * M_GAS))
    rate = n * vbar * math.pi * R**2                # collisions /s
    dp2 = 2 * M_GAS * KB * T_GAS                    # <Dp^2> par collision
    return rate * dp2, rate


def d_csl(lam_csl, r_c=1e-7):
    """Diffusion d'impulsion CSL, sphere homogene (facteur O(1))."""
    x = R / r_c
    f = (6 / x**4) * (1 + math.exp(-x * x)
                      - (2 / x**2) * (1 - math.exp(-x * x)))
    return 0.5 * lam_csl * (HBAR / r_c) ** 2 * (M / (1 * AMU)) ** 2 * f


def gamma_th(d_p):
    return d_p / (2 * M * HBAR * OMEGA_M)


def run():
    print("=== Chiffrage : Gamma commun decoherence/dissipation ===\n")
    print("Banc : sphere silice R = %.1f nm, m = %.2e kg, w/2pi = %.0f kHz" % (
        R * 1e9, M, OMEGA_M / (2 * math.pi * 1e3)))
    print("       pince %.0f mW @ %.0f nm ; gaz %.0e mbar ; x_zpf = %.1f pm\n"
          % (P_TWEEZER * 1e3, LAM * 1e9, P_GAS / 100, X_ZPF * 1e12))

    print("[1] L'IDENTITE (une seule quantite D_p, deux instruments) :")
    print("    Gamma_dec(Dx) = D_p Dx^2/hbar^2  et  Gamma_th = D_p/(2 m hbar w)")
    print("    avec x_zpf^2 = hbar/(2 m w)  =>  Gamma_dec(x_zpf) = Gamma_th")
    d_r, p_scat = d_recoil()
    print("    verification numerique (canal recul) : ratio = %.10f\n" % (
        (d_r * X_ZPF**2 / HBAR**2) / gamma_th(d_r)))

    print("[2] CHIFFRES par canal (phonons/s = Gamma_dec a x_zpf) :")
    g_r = gamma_th(d_r)
    d_g, rate_g = d_gas()
    g_g = gamma_th(d_g)
    print("    recul de photons : Gamma = %.1e ph/s  (P_scat = %.0f uW)" % (
        g_r, p_scat * 1e6))
    print("      -> ordre 1e4-1e5 conforme aux mesures (Jain 2016 recul")
    print("         direct ; Delic 2020 rechauffage etat fondamental)")
    print("    collisions de gaz : Gamma = %.1e ph/s  (%.0f coll/s)" % (
        g_g, rate_g))
    print("    -> recul DOMINANT a 1e-8 mbar (x%.0f)\n" % (g_r / g_g))

    print("[3] PREDICTION CHIFFREE (contenu TQNT propre) :")
    print("    (a) zeta = Gamma_dec(x_zpf)/Gamma_th = 1 EXACTEMENT apres")
    print("        soustraction du bruit classique (intensite/phase laser :")
    print("        chauffe SANS decoherer, zeta < 1) ; zeta > 1 INTERDIT.")
    print("        Test : interferometrie (visibilite) vs PSD (rechauffage)")
    print("        sur le meme objet -- aucun parametre libre.")
    print("    (b) residuel non environnemental STRICTEMENT NUL :")
    for name, lam_csl in (("GRW  (lambda = 1e-16 /s)", 1e-16),
                          ("Adler (lambda ~ 1e-8 /s)", 1e-8)):
        g_c = gamma_th(d_csl(lam_csl))
        print("        %-26s : %.1e ph/s (%s)" % (
            name, g_c,
            "1e10 SOUS le recul, hors de portee" if lam_csl < 1e-12
            else "ACCESSIBLE piege sombre UHV"))
    print("        TQNT : 0 strict -- toute detection d'un exces apres")
    print("        comptabilite des canaux refuterait. (Ordres littérature :")
    print("        emission X et LISA Pathfinder pressent deja lambda vers")
    print("        ~1e-11/1e-12 ; le plancher GRW 1e-16 reste ouvert.)\n")

    print("[4] CAVEAT CONDUCTEURS (honnete) : tau_p != tau_phi ne refute pas.")
    tau_p, tau_phi = 3e-14, 1e-9   # Au ~1 K : ordres (residuel / loc. faible)
    print("    Au a ~1 K : tau_p ~ %.0e s, tau_phi ~ %.0e s (rapport %.0e)" % (
        tau_p, tau_phi, tau_phi / tau_p))
    print("    Diffuseur STATIQUE = pas de partenaire dynamique = pas de")
    print("    reconnexion : ni enregistrement ni entropie a cet evenement ;")
    print("    l'identite par evenement vit aux contacts INELASTIQUES, et")
    print("    q_dot = rho J^2 est un bilan de regime permanent. Corollaire")
    print("    structurel : le desordre statique seul ne dephase JAMAIS --")
    print("    exactement ce que montre la localisation faible.\n")

    print("BILAN : l'identite Gamma_dec(x_zpf) = Gamma_th est HERITEE de la")
    print("mecanique quantique des mesures efficientes ; l'apport TQNT =")
    print("son ANCRAGE ontologique (meme evenement de reconnexion) + les")
    print("deux chiffres falsifiables : zeta = 1 (jamais > 1) et residuel")
    print("collapse = 0 (vs plancher Adler %.0f ph/s accessible)." % (
        gamma_th(d_csl(1e-8))))


if __name__ == "__main__":
    run()
