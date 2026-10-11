"""Le phonon dans la trame — vérifications (P1-P4).

P1 : la hiérarchie v_s << c est la hiérarchie contenu/trame —
     relation de Bohm-Staver v_s = v_F sqrt(Z m_e/3M) (le rappel est
     EM/framing au niveau alpha, l'inertie est le déficit des noyaux).
P2 : le canal troisième-corps — gap indirect du Si : le photon ne peut
     PAS fournir le recul (k_photon/k_X ~ 1/1800) -> le phonon est le
     troisième corps du dictionnaire P22-effectif.
P3 : le canal dipolaire — SiC : les phonons optiques portent un dipôle
     (courants de framing oscillants) -> polaritons, reststrahlen
     [w_TO, w_LO] : c'est le epsilon(i xi) qui nourrit W (D-1, x910).
P4 : le canal trame-effective — gravité analogique : un écoulement donne
     au champ de phonons une métrique effective (Unruh 1981) ; T_Hawking
     analogique d'un BEC (Steinhauer 2016) ~ nK.
"""
import numpy as np

ME = 9.1093837015e-31
U = 1.66053907e-27
HBAR = 1.054571817e-34
KB = 1.380649e-23
C = 2.99792458e8
E_CH = 1.602176634e-19

print("=" * 74)
print("LE PHONON DANS LA TRAME — P1-P4                     (2026-09-02)")
print("=" * 74)

# ---- P1 : Bohm-Staver -----------------------------------------------------
print("\nP1 — la hiérarchie v_s/c (Bohm-Staver : rappel alpha-EM, inertie"
      " = déficits)")
mats = [("Cu", 1, 1.57e6, 63.55, 4760), ("Al", 3, 2.03e6, 26.98, 6420),
        ("Pb", 4, 1.83e6, 207.2, 1960)]
for nom, Z, vF, A, vmes in mats:
    vs = vF * np.sqrt(Z * ME / (3 * A * U))
    print(f"   {nom} : v_s = {vs:5.0f} m/s (mesuré long. ~{vmes}) ; "
          f"v_s/c = {vs/C:.1e}")
print("   -> l'ordre sort de (v_F, m_e/M) SEULS : échelles du contenu ;")
print("      la petitesse v_s/c ~ 1e-5 EST la hiérarchie contenu/trame.")

# ---- P2 : troisième corps (gap indirect) ---------------------------------
a_si = 5.431e-10
k_X = 0.85 * 2 * np.pi / a_si
k_ph = 1.12 * E_CH / (HBAR * C)
print(f"\nP2 — le phonon = troisième corps (Si, gap indirect)")
print(f"   recul requis k_X = {k_X:.2e} /m ; photon 1,12 eV : "
      f"k = {k_ph:.2e} /m")
print(f"   -> rapport {k_X/k_ph:.0f} : le photon ne PEUT PAS fournir le "
      f"recul ;")
print("      l'absorption indirecte = photon + PHONON — le « réseau = 3e")
print("      corps » du dictionnaire P22-effectif est un échange de "
      "phonon.")

# ---- P3 : canal dipolaire (SiC) ------------------------------------------
wTO, wLO = 1.494e14, 1.827e14
print(f"\nP3 — le canal dipolaire : SiC, bande reststrahlen "
      f"[{HBAR*wTO/E_CH*1e3:.0f}, {HBAR*wLO/E_CH*1e3:.0f}] meV")
print(f"   hbar w_TO / k_B T(300K) = {HBAR*wTO/(KB*300):.1f} : DANS la "
      f"fenêtre thermique")
print("   -> les phonons optiques polaires = courants de framing "
      "oscillants :\n      ils entrent dans epsilon(i xi) et nourrissent "
      "les 3 branches de W\n      (le x910 de D-1 est le phonon vu par la "
      "trame). Acoustiques : sans\n      dipôle net -> sombres pour la "
      "trame (couplage d'ordre supérieur).")

# ---- P4 : trame effective (gravité analogique) ---------------------------
c_s, ell = 0.5e-3, 2e-6          # BEC : vitesse du son, échelle du saut
T_H = HBAR * (c_s / ell) / (2 * np.pi * KB)
print(f"\nP4 — le canal trame-effective (gravité analogique)")
print(f"   BEC (Steinhauer 2016) : c_s ~ {1e3*c_s} mm/s, gradient sur "
      f"~{1e6*ell:.0f} um\n   -> T_Hawking analogique = "
      f"hbar kappa/(2 pi k_B) ~ {1e9*T_H:.2f} nK (mesuré ~nK)")
print("""   -> un écoulement courbe la métrique EFFECTIVE du champ de phonons
      (Unruh 1981) : le phonon est l'excitation d'une trame effective
      qu'on construit et courbe à volonté — le prolongement du programme
      « construire le vide » : CONSTRUIRE LA TRAME.
   Contraste instructif (ce que la trame effective N'HÉRITE PAS) :
      h : IDENTIQUE (la dent est absolue — la graduation en action vaut
          pour tout milieu : la quantification du phonon vient de là) ;
      c -> v_s : renormalisée (hiérarchie P1) ;
      protections : PERDUES — dispersion au bord de zone, amortissement,
          « Lorentz » approximatif aux grandes longueurs d'onde seulement.
      Les protections de la vraie trame (dispersion 0, frottement 0,
      k entier) ne sont PAS des propriétés génériques de milieu : ce sont
      les théorèmes qui distinguent la trame d'un milieu.""")
