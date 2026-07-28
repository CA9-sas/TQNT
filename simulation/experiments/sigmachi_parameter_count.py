"""Comptage propre des parametres du secteur de masse canonique m = sqrt(sigma_chi) * S.

Source de verite : l'engin valide spectrum_sigmachi_full.py (mediane 0.34%). On enumere
CHAQUE constante et on la classe : DERIVE de sqrt(sigma_chi) / CALIBRE (TQNT) / EMPRUNTE (QCD)
/ HINT / IDENTIFIE.

RESULTAT HONNETE :
  - sqrt(sigma_chi)=417.65 est la SEULE echelle du secteur LEGER/topologique (leptons Koide +
    baryons Regge+Z3+tenseur) : pente Regge, Lambda, Z3, tenseur Lambda/8 tous DERIVES.
  - Les secteurs LOURD/mesons/EW ne sont PAS predits depuis sqrt(sigma_chi) : ils portent des
    intrants propres (Cornell ; VEV) et des MASSES DE NOEUDS (TQNT, calibrees) : m_l = trefle 3_1
    (habille), m_s/m_c/m_b = noeuds etrange/charme/beaute, f_pi = echelle chirale Goldstone.
    PAS des quarks empruntes a QCD (il n'y a PAS de quarks fondamentaux en TQNT) -- meme si les
    VALEURS coincident (memes hadrons). Cible de derivation : M_topo = sqrt(sigma_chi)*ropelength.
  => Total masse = 9 calibres TQNT (4 coeur + 5 noeuds) + 2 hints (sqrt2, delta)
     [MAJ 21-22/07 : v DERIVE (croisement-ouvert) sort du coeur ; m_s DECLASSE
     (audit ACPR) rejoint les noeuds calibres].
  CADRE (audit 2026-07-13) : beta_otimes seul CALIBRE ; eta NON PINCE (~1, jamais calibre) ;
    alpha_cond DERIVEE (<0.1%, cinetique de condensation, commit bebfa51) ;
    E_0 reclasse COUCHE SOLVEUR chi (P4/P8, avec gamma_J et lambda_V : jamais ajuste aux
    observables ; echelle canonique de deconfinement E_K = hbar*c/r_link ~ 200 MeV, derivee).
  => TOTAL V6 : 10 calibres + 2 hints (eta ELIMINE 2026-07-14 : P22 en forme WKB
     Gamma_P exp(-2pi/xi), exposant DERIVE — Schwinger = -2pi/xi et Hawking = -4pi/xi
     exacts via xi = 2a*lambdabar_C/c^2 ; Gamma_0 elimine aussi).
  (C_plast archive : sa cible, le residu mu- d'Omega_ts, n'existe pas sous sqrt(sigma)*S).
"""

# (symbole, valeur, secteur, statut)
LEDGER = [
    ("sqrt(sigma_chi)", "417.65 MeV", "ECHELLE unique",            "CALIBRE (TQNT)"),
    ("2 pi sigma_chi",  "1.096 GeV^2","pente de Regge",            "DERIVE de sqsig"),
    ("Lambda_chi",      "208.8 MeV",  "= sqsig/2",                 "DERIVE de sqsig"),
    ("E_Z3 = 2/3 Lambda","139.2 MeV", "quantum Z3 (baryons)",      "DERIVE (2/3 = Koide)"),
    ("tenseur Lambda/8","26.1 MeV",   "correction bas-J baryons",  "DERIVE de sqsig"),
    ("delta_Koide",     "2/9 rad",    "phase leptons (=Q/3)",      "HINT (=Q/3)"),
    ("A",               "sqrt(2)",    "amplitude Koide (leptons)", "HINT (non derive)"),
    ("3/4",             "0.75",       "Casimir spin-1/2 (Koide)",  "DERIVE (Casimir)"),
    ("SL_rad/SL",       "1.38",       "pente radiale (excites)",   "CALIBRE (TQNT)"),
    ("tau_perp",        "~112 MeV",   "hyperfin N-Delta",          "CALIBRE (TQNT)"),
    ("hf_ccbar",        "J/psi-eta_c","hyperfin lourd (S-onde)",   "CALIBRE (TQNT)"),
    ("sigma_Cornell",   "= sigma_chi","quarkonia : MEME corde (catalogue) ; + alpha_s mesure", "DERIVE de sqsig"),
    ("v (VEV)",         "246.23 GeV", "EW : W,Z,H,top",            "DERIVE 21/07 (croisement-ouvert, +28 ppm ; hors comptage)"),
    ("f_pi / B chiral", "~92 MeV",    "mesons legers (GMOR)",      "NOEUD TQNT (calibre)"),
    ("m_l",             "310 MeV",    "trefle 3_1 (habille)",      "NOEUD TQNT (calibre)"),
    ("m_s",             "494.8 MeV",  "noeud etrange 5_2",         "NOEUD TQNT (calibre ; declasse 22/07, indice geometrique)"),
    ("m_c",             "1557 MeV",   "noeud charme (D, B_c)",     "NOEUD TQNT (calibre)"),
    ("m_b",             "4900 MeV",   "noeud beaute (B)",          "NOEUD TQNT (calibre)"),
]

FRAMEWORK = [  # parametres-cadre (hors masse), audit 2026-07-13
    ("beta_otimes", "0.210 MeV", "tension d'enlacement (n-p, durees fortes)", "CALIBRE"),
    ("eta",         "elimine",   "P22 reformule : Gamma_P exp(-2pi/xi)",       "ELIMINE (exposant derive, 2026-07-14)"),
    ("alpha_cond",  "1.20",      "penalite condensation (matiere noire)",     "DERIVEE (<0.1%, bebfa51)"),
    ("E_0",         "1.0 MeV",   "desserrage rho(K,E) de P4/P8",              "COUCHE SOLVEUR (avec gamma_J, lambda_V)"),
]


def run():
    print("=== Secteur de masse m = sqrt(sigma_chi) * S : ledger ===")
    print(f"{'symbole':18}{'valeur':13}{'secteur':30}{'statut'}")
    for s, v, sec, st in LEDGER:
        print(f"{s:18}{v:13}{sec:30}{st}")

    cal = [r for r in LEDGER if r[3].startswith("CALIBRE")]
    noeud = [r for r in LEDGER if r[3].startswith("NOEUD")]
    der = [r for r in LEDGER if r[3].startswith("DERIVE")]
    hint = [r for r in LEDGER if r[3].startswith("HINT")]
    idf = [r for r in LEDGER if r[3].startswith("IDENTIFIE")]
    print("\n=== Comptage du secteur de masse ===")
    print(f"  CALIBRE (TQNT natif)  : {len(cal)}  {[r[0] for r in cal]}")
    print(f"  IDENTIFIE (non derive): {len(idf)} {[r[0] for r in idf]}")
    print(f"  NOEUD TQNT (calibre)  : {len(noeud)}  {[r[0] for r in noeud]}")
    print(f"  HINT                  : {len(hint)} {[r[0] for r in hint]}")
    print(f"  DERIVE de sqrt(sigma) : {len(der)}  (ne comptent pas)")
    print(f"\n  => masse : {len(cal)+len(idf)+len(noeud)} calibres TQNT ({len(cal)+len(idf)} coeur + {len(noeud)} noeuds), tous topologiques.")
    print(f"     Secteur LEGER (leptons+baryons) : sqsig seul (+ hints sqrt2, delta).")
    print("\n=== Parametres-cadre (hors masse), audit 2026-07-13 ===")
    for s, v, role, st in FRAMEWORK:
        print(f"  {s:12}{v:11}{role:42}{st}")
    n_masse = len(cal) + len(idf) + len(noeud)
    n_cadre_cal = sum(1 for r in FRAMEWORK if r[3].startswith("CALIBRE"))
    n_nonpince = sum(1 for r in FRAMEWORK if r[3].startswith("NON PINCE"))
    print(f"\n  TOTAL V6 = {n_masse} (masse) + {n_cadre_cal} (cadre : beta_otimes) = "
          f"{n_masse + n_cadre_cal} calibres TQNT + {len(hint)} hints (sqrt2, delta). "
          f"Aucun quark. eta ELIMINE (P22 WKB, exposant derive).")
    print("  alpha_cond derivee (ne compte pas) ; E_0 couche solveur (hors comptage observable).")


if __name__ == "__main__":
    run()
