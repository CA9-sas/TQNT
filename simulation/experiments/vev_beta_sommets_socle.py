"""VEV electrofaible et beta_otimes par les sommets amplifies du socle (2026-07-20).

IDEE (utilisateur) : recalculer v et beta_otimes avec les derniers elements
(socle ondulant, echelle par croisement du raccord, T_c derive) en DETERMINANT
LES SOMMETS AMPLIFIES par les particules.

DISCIPLINE : le mur v est CERTIFIE (etude_v_mpl_mur.py, null 225 formes,
1 grappe sqrt(2pi) alpha^8 attendue a ~12 % par hasard) -> tout candidat doit
DEPASSER ce null (mecanisme + convergence independante), sinon v reste calibre.
beta_otimes : mecanisme fluxoide archive (derive_beta_otimes_test.py, H2) --
on ne recalcule pas, on teste ce que le socle AJOUTE.

ROUTE A (v) -- l'echelle par croisement appliquee a la barriere faible :
  Le raccord (sec:fc-raccord) a DERIVE l'echelle par croisement pour G :
  alpha^0 = fond d'ondes, alpha^1 = interface (+1), alpha^18 = nucleon
  (m_base/M_Pl = sqrt(12/7) alpha^18 ; 18 = boucles d'Euler c_tot+1 = 17+1).
  Lecture socle : chaque croisement descend d'UN barreau la hierarchie des
  sommets amplifies (un facteur alpha par croisement).
  CANDIDAT : la barriere de relaxation P15 vit sur l'echelle du circuit
  RECONNECTE. v = A * alpha^n * M_Pl avec :
    - n = 8 : DEUX lectures topologiques disponibles (declare : la
      degenerescence AFFAIBLIT le pin) --
      (a) c(8_3) = 8 nu (le noeud commun du quartet EW, porteur de structure) ;
      (b) c(7_2)+1 = 8 boucles d'Euler du compagnon twist -- MEME regle que le
          nucleon (c+1), et le 7_2 est PRECISEMENT l'objet dont Lk bascule
          dans la desintegration beta (P15) ;
    - A = sqrt(2pi) : mesure de sommet gaussienne (le socle n'amplifie que le
      sommet ; l'energie du paquet complet / hauteur du sommet = sqrt(2pi)
      en unites de la dent) -- LECTURE, pas derivation.
  Verdict attendu : accord fin, mais UNE grappe = 12 % de hasard au null ;
  sans convergence independante, v RESTE calibre.

ROUTE B (beta_otimes) -- ce que le socle ajoute au fluxoide :
  B1. d_tube = lambdabar_C(p) (mesure a 1 %, V6 torsion) : le socle le LIT --
      le sommet amplifie par le nucleon a la taille Compton (E=hf identitaire,
      un sommet par periode) ; la section du tube EST le sommet amplifie.
      Consolidation d'une ENTREE, pas un nombre nouveau.
  B2. r_link : le circuit d'enlacement est UN sommet du quantum qu'il piege
      (E_K = hbar c / r_link ~ 200 MeV : mode fondamental d'anneau,
      lambda = 2 pi r). Lecture de coherence.
  B3. Discretisation en sommets : le circuit porte N = 2 pi r / lambdabar_C
      sommets ; correction polygone vs cercle sur L_ind -> chiffree ici.
  Verdict attendu : aucun levier sur le +8 % (residuel = geometrie non
  circulaire du vrai circuit, route SONO) ; beta_otimes RESTE calibre.
"""
import math

# --- constantes ---
HBARC_MEVFM = 197.3269804
M_PL_GEV = 1.220890e19          # sqrt(hbar c / G) CODATA (convention raccord)
M_BASE_MEV = 938.240            # nucleon de base (convention 2026-07-19)
V_MEAS_GEV = 246.21965          # vev mesure (G_F)
ALPHA_EXACT = 1.0 / 137.0       # 1/(k+2), k = 135
ALPHA_MEAS = 1.0 / 137.035999   # CODATA
ALPHA_DEF = math.cos(math.pi / 137.0) / 137.0   # candidat P1 raffine (-0,21 ppm)

# --- entrees beta_otimes (chaines anterieures ; cf. derive_beta_otimes_test) ---
R_LINK_FM = 0.99
A_TUBE_FM = 0.209               # rayon du tube = lambdabar_C(p)
TARGET_DM = 1.293               # MeV


def route_a_v():
    print("=" * 72)
    print("ROUTE A -- v par l'echelle par croisement (sommets du socle)")
    print("=" * 72)
    amp = math.sqrt(2 * math.pi)
    for name, al in [("alpha = 1/137 exact", ALPHA_EXACT),
                     ("alpha mesure", ALPHA_MEAS),
                     ("alpha deforme cos(pi/137)/137", ALPHA_DEF)]:
        v = amp * al**8 * M_PL_GEV
        print("  v = sqrt(2pi) alpha^8 M_Pl [%s] = %.3f GeV  (%+.3f %%)"
              % (name, v, 100 * (v / V_MEAS_GEV - 1)))

    # Identite sans M_Pl (via m_base/M_Pl = sqrt(12/7) alpha^18) :
    # v/m_base = sqrt(2pi)/sqrt(12/7) * alpha^(8-18) ... attention :
    # m_base = sqrt(12/7) alpha^18 M_Pl -> v/m_base = sqrt(2pi/(12/7)) alpha^-10 ?
    # Non : v/m_base = sqrt(2pi) alpha^8 / (sqrt(12/7) alpha^18)
    #               = sqrt(7 pi / 6) * alpha^-10. Verif numerique :
    ratio_pred = math.sqrt(7 * math.pi / 6) * ALPHA_EXACT**(-10) / 137.0**9
    # (reecriture propre ci-dessous)
    v_from_base = M_BASE_MEV / 1000 * math.sqrt(7 * math.pi / 6) * (1 / ALPHA_EXACT)**10 \
        * ALPHA_EXACT**9  # = m_base * sqrt(7pi/6) * 137
    print("\n  Identite sans M_Pl : v = m_base * sqrt(7pi/6) * (k+2)")
    print("    = %.4f GeV * %.4f * 137 = %.2f GeV (%+.3f %% ; restatement de la"
          % (M_BASE_MEV / 1000, math.sqrt(7 * math.pi / 6), v_from_base,
             100 * (v_from_base / V_MEAS_GEV - 1)))
    print("    meme forme via m_base/M_Pl = sqrt(12/7) alpha^18 -- PAS independant)")

    # Consequence tau_mu si v_pred adopte (levier v^4)
    v_pred = amp * ALPHA_DEF**8 * M_PL_GEV
    print("\n  Consequence tau_mu (levier v^4) : delta = %+.2f %% (vs -0,44 %% "
          "radiatif\n  a v mesure -- compatible a l'ordre des corrections)"
          % (100 * ((V_MEAS_GEV / v_pred)**4 - 1)))

    # Census de formes (mini-null local, DECLARE) : A * alpha^n * M_Pl
    print("\n  MINI-NULL DECLARE : formes A*alpha^n*M_Pl a moins de 0,5 %% de v :")
    amps = [("1", 1.0), ("sqrt2", math.sqrt(2)), ("2", 2.0), ("sqrt3", math.sqrt(3)),
            ("pi/2", math.pi / 2), ("sqrt(pi)", math.sqrt(math.pi)),
            ("sqrt(2pi)", amp), ("pi", math.pi), ("2pi", 2 * math.pi),
            ("e", math.e), ("sqrt(12/7)", math.sqrt(12 / 7)), ("4pi", 4 * math.pi)]
    hits = 0
    for an, av in amps:
        for n in range(6, 11):
            v = av * ALPHA_EXACT**n * M_PL_GEV
            if abs(v / V_MEAS_GEV - 1) < 0.005:
                hits += 1
                print("    HIT  A=%-10s n=%d -> %.2f GeV (%+.3f %%)"
                      % (an, n, v, 100 * (v / V_MEAS_GEV - 1)))
    print("    %d hit(s) / %d formes testees ; fenetre +-0,5 %% ;" % (hits, 60))
    print("    attendu par hasard ~ 60 * 0,01 / (ln(137)/2) ~ 0,25 -> 1 hit")
    print("    n'est PAS significatif seul (coherent avec le null 225 formes).")

    print("\n  LECTURES DE L'EXPOSANT 8 (degenerescence declaree) :")
    print("    (a) c(8_3) = 8 nu -- noeud commun du quartet EW ;")
    print("    (b) c(7_2)+1 = 8 boucles d'Euler -- MEME regle que le nucleon")
    print("        (c_tot+1 = 18), et le 7_2 est l'objet reconnecte (dLk beta).")
    print("    DEUX lectures pour le meme entier = liberte de forme residuelle ;")
    print("    la designation physique (b) est la plus coherente (regle unifiee")
    print("    + objet du processus P15) mais elle n'est pas UNIQUE.")
    print("\n  VERDICT A : candidat de MECANISME pour la grappe connue du null")
    print("  (sqrt(2pi) alpha^8 a -0,05 %) -- pas de convergence independante")
    print("  nouvelle -> le mur v TIENT, v RESTE identifie/calibre.")


def loop_inductance_factor(c_geom):
    return math.log(8 * R_LINK_FM / A_TUBE_FM) - c_geom


def polygon_correction(n_sides):
    """L(polygone regulier a N cotes) / L(cercle), meme rayon circonscrit,
    fil de rayon a. Formule du polygone (Grover) : L = (mu0 p / 2pi) *
    [ln(p/a) - ln(N) + ln(2) - gamma_N] avec gamma_N -> corrections de forme.
    On chiffre l'ORDRE par la formule des segments droits mutuellement couples
    (approximation : perimetre p = N s, self de segment + mutuelles voisines).
    Retourne le ratio approximatif L_poly / L_cercle."""
    p = 2 * math.pi * R_LINK_FM
    s = p / n_sides
    # self d'un segment droit : (mu0 s/2pi)(ln(2s/a) - 1) ; somme = N selfs
    # + mutuelles ~ reconstruisent le ln global ; on encadre par la difference
    # de terme constant : cercle ln(8r/a)-2 vs polygone ln(2Ns/a)-~2.45 (carre)
    # -> correction relative ~ (delta_cste)/(ln(8r/a)-2), bornee par 1/N^2 pour
    # N grand (le polygone tend vers le cercle).
    l_circle = math.log(8 * R_LINK_FM / A_TUBE_FM) - 2
    delta = 1.0 / n_sides**2 * math.pi**2 / 6  # ordre de l'ecart geometrique
    return 1 - delta / l_circle


def route_b_beta():
    print()
    print("=" * 72)
    print("ROUTE B -- beta_otimes : ce que le socle ajoute au fluxoide")
    print("=" * 72)
    # rappel du mecanisme archive (H2, conventions derive_beta_otimes_test)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dbo", "simulation/experiments/derive_beta_otimes_test.py")
    dbo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dbo)
    e_surface = dbo.h2_fluxoid(2.0)
    e_uniform = dbo.h2_fluxoid(7.0 / 4.0)
    print("  Rappel archive (H2 fluxoide) : surface c=2 -> %.3f MeV (%+.1f %%) ;"
          % (e_surface, 100 * (e_surface / TARGET_DM - 1)))
    print("    uniforme c=7/4 -> %.3f MeV (%+.1f %%) ; cible %.3f encadree."
          % (e_uniform, 100 * (e_uniform / TARGET_DM - 1), TARGET_DM))

    print("\n  B1. d_tube = lambdabar_C(p) : LECTURE SOCLE -- le sommet amplifie")
    print("      par le nucleon a la taille Compton (un sommet par periode,")
    print("      E = hf identitaire). L'entree mesuree (1 %) devient une")
    print("      prediction du socle. Aucun nombre nouveau.")
    print("  B2. r_link = %.2f fm <-> E_K = hbar c / r = %.0f MeV : le circuit"
          % (R_LINK_FM, HBARC_MEVFM / R_LINK_FM))
    print("      est le mode fondamental d'anneau du quantum qu'il piege")
    print("      (lambda = 2 pi r). Lecture de coherence, pas un pin.")

    n_crests = 2 * math.pi * R_LINK_FM / A_TUBE_FM
    corr = polygon_correction(n_crests)
    e_corr = e_surface / corr
    print("\n  B3. Discretisation en sommets : N = 2 pi r / lambdabar_C = %.1f"
          % n_crests)
    print("      correction polygone/cercle sur L_ind ~ %.4f -> E = %.3f MeV"
          % (corr, e_corr))
    print("      (%+.2f %% vs cible) : ordre 1e-3, AUCUN levier sur le +8 %%."
          % (100 * (e_corr / TARGET_DM - 1)))

    print("\n  VERDICT B : le socle CONSOLIDE deux entrees (d_tube = sommet")
    print("  Compton ; r_link = mode d'anneau) mais n'apporte AUCUN levier")
    print("  numerique nouveau ; le residuel +8 % est geometrique (circuit")
    print("  non circulaire, route SONO) -> beta_otimes RESTE calibre.")


def run():
    print(__doc__)
    route_a_v()
    route_b_beta()


if __name__ == "__main__":
    run()
