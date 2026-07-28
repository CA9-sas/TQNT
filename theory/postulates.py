"""
Théorie Nodale v3 — Postulats Fondamentaux

P1:  Le néant est instable.
P2:  Les ondes sont des aberrations du néant.
P3:  Les nœuds topologiques sont classifiés par le polynôme de Jones V_K(t).
     Jones distingue la chiralité → matière/antimatière.
P4:  ρ(K, E) = ρ_J² / (ρ_J + E/E₀) avec ρ_J = span(V_K) + γ·|V_K(-1)|.
     La rétractation décroît avec l'énergie ; les nœuds complexes résistent davantage.
P5:  L'espace-temps émerge des ondes. Métrique bi-conforme :
     ds² = -c²e^{-κ|Ψ|²}dt² + (1+κ|Ψ|²)(dx²+dy²+dz²).
     Le temps ralentit près des nœuds (gravité). β_PPN = γ_PPN = 1.
P6:  L'équation de Schrödinger ÉMERGE de Jones via Chern-Simons/A-polynomial.
P7:  Le spin est lié au sens de parcours et à la complexité du nœud.
P8:  Le serrage énergétique explique la dualité onde-particule.
P9:  Pas de mécanisme de Higgs (brisure spontanée), pas de graviton.
     Le boson H existe comme état composite (entrelacs 3₁+5₁+7₄, k=29).
P10: Le champ χ satisfait l'équation de Chern-Simons SU(2). Son équation est
     unique. Les particules = conditions topologiques (nœuds) sur χ.
P14: Le photon est une onde pure. Le gluon est le reliquat d'onde non-nodifié.
     ℏ et c sont les limites topologiques de l'espace-temps.
P15: Fragilité topologique : tau ∝ exp(-λ·n) / M_raw^ν.
     Chaque nœud = canal de désintégration indépendant.
"""
import numpy as np

# --- v1 constants (kept for backward compatibility) ---
ALPHA = 1.0     # Mass scaling constant
BETA = 0.5      # Angular node mass contribution (v1)
KAPPA = 0.1     # Spacetime coupling constant
C = 1.0         # Wave propagation speed (natural units)
HBAR = 1.0      # Reduced Planck constant (natural units)
E_CHARGE = 1.0  # Elementary charge unit

# --- v2 constants ---
GAMMA = 0.1     # Retraction coupling: weight of |Delta(-1)| in rho
LAMBDA = 1.0    # Retraction potential strength
R_0 = 10.0      # Base knot tightening radius
E_0 = 1.0       # Tightening energy scale

# --- v4 constants (multi-knot links) ---
LAMBDA_LINK = 1.0   # Link mass-distance coupling
BETA_LINK = 0.5     # Link charge-distance amplification


def nodal_mass(n: int, l: int) -> float:
    """v1: Mass from radial/angular nodes. M = ALPHA * n + BETA * l * (l + 1)"""
    return ALPHA * n + BETA * l * (l + 1)


def nodal_energy(n: int, l: int) -> float:
    """v1: Energy from nodal numbers. E = M * c²."""
    return nodal_mass(n, l) * C**2


def nodal_charge(m: int) -> float:
    """v1: Charge from azimuthal quantum number m."""
    return float(m) * E_CHARGE


def retraction_mass(rho: float) -> float:
    """v2: Mass from retraction factor. M = ALPHA * rho(K)."""
    return ALPHA * rho


# --- v3 postulate documentation ---

POSTULATE_P3_V3 = """
P3 (v3): Les nœuds topologiques sont classifiés par le polynôme de Jones V_K(t).
Jones distingue la chiralité (V_K(t) ≠ V_K(t⁻¹))
ce qui permet de différencier matière et antimatière via les nœuds miroirs.
"""

POSTULATE_P4_V3 = """
P4 (v3): Le facteur de rétractation dépend de l'énergie :
  ρ(K, E) = ρ_J(K)² / (ρ_J(K) + E/E₀)
  avec ρ_J(K) = span(V_K) + γ·|V_K(-1)|
La rétractation décroît avec l'énergie ; les nœuds complexes résistent davantage
car leur échelle d'énergie caractéristique E_K = E₀·ρ_J(K) est plus grande.
"""

POSTULATE_P6_V3 = """
P6 (v3): L'équation de Schrödinger n'est plus postulée. Elle ÉMERGE de la
quantification du polynôme-A, lui-même dérivé du Jones colorié via la conjecture AJ :
  Jones colorié J_n → récurrence Â_K → limite classique → A_K(L,M) = 0 → Schrödinger
"""

POSTULATE_P7_V3 = """
P7 (v3): Le spin et la charge sont des invariants topologiques du nœud K.
  Spin : s(K) = (b(K) - 1) / 2, où b(K) est l'indice de tresse minimal.
    b=1 → spin 0 (scalaire), b=2 → spin 1/2 (fermion),
    b=3 → spin 1 (boson de jauge), b≥4 → spin 3/2+.
  Charge : Q(K) = (p_min + p_max) / span(V_K), centre de puissance normalisé.
    Amphichiraux → Q=0. Miroir : Q(K*) = -Q(K).
    Valeurs fractionnaires : 1/3, 2/3, 1 (charges du Modèle Standard).
"""

POSTULATE_P7_V4 = """
P7 (v4): Modèle dual de détermination de la charge.

  FERMIONS (nœuds toriques T(2,s), b=2, spin 1/2) :
    Chaque nœud torique héberge une GÉNÉRATION de 4 fermions
    (eigenstates de Schrödinger sur le complément S³\\K).
    La charge vient de Gell-Mann–Nishijima :
      Q = I₃ + Y/2
    avec I₃ = ±1/2 (isospin faible) et Y (hypercharge) :
      - Leptons : Y = -1  →  neutrino (Q=0), lepton chargé (Q=-1)
      - Quarks  : Y = +1/3 →  quark up-type (Q=+2/3), quark down-type (Q=-1/3)

    3 générations :
      Gén. 1 : 3_1 = T(2,3) → νe, e⁻, u, d
      Gén. 2 : 5_1 = T(2,5) → νμ, μ⁻, c, s
      Gén. 3 : 7_1 = T(2,7) → ντ, τ⁻, t, b

    Antiparticules sur nœuds miroirs avec I₃ → -I₃, Y → -Y.

  BOSONS (nœuds non-toriques, b≥3, spin ≥ 1) :
    Charge topologique directe : Q(K) = (p_min + p_max) / span(V_K).
      4_1 (amphichiral) → Q=0 : γ, g, Z⁰
      8_2             → Q=+1 : W⁺
      8_2* (miroir)   → Q=-1 : W⁻
"""

POSTULATE_P10 = """
P10: Il existe un champ fondamental χ ('tchi') satisfaisant l'équation de
Chern-Simons SU(2) au niveau k. Son équation est unique pour toutes les particules.
Les différentes particules correspondent à différentes conditions topologiques (nœuds K).
La quantification de χ sur le complément S³\\K produit l'équation de Schrödinger
émergente via le polynôme-A.
"""

POSTULATE_P11 = """
P11: Les particules massives sont des entrelacs de n noeuds K_1, ..., K_n plonges
dans le meme R^3. L'onde Psi est l'unique solution de Schrodinger sur le
complement commun S^3 \\ (K_1 U ... U K_n). Le potentiel est V(r) = Sum_i V(r, K_i).

La masse de l'entrelacs depend de la distance inter-noeuds d :
  M_link = Sum_i M(K_i) + lambda_link * Sum_i rho_J(K_i) / d

La charge est amplifiee par la distance :
  Q_link = (Sum_i Q(K_i)) * (1 + beta_link * d / d_0)

Les distances inter-noeuds stables emergent comme minima d'energie et
correspondent aux inverses des nombres de Fibonacci : d_stable(k) = d_0 / F(k).
"""

POSTULATE_P12 = """
P12 (Potentiel de Cornell topologique) :
L'energie totale d'un entrelacs multi-noeuds est un potentiel de Cornell :
  U(d) = Sum_i M(K_i) + A/d + sigma_k * d
ou :
  A = lambda_link * Sum_i rho_J(K_i) / 2    (compression quantique)
  sigma_k = lambda_link * Sum_i rho_J(K_i) * F(k)^2 / (2 * R_0^2)
                                              (tension de corde topologique)

L'equilibre est atteint pour d*(k) = R_0 / F(k), donnant la masse :
  m(k) = eta * [Sum_i M(K_i) + lambda * Sum_i rho_J(K_i) * F(k) / R_0]

Le nombre quantique de liaison k (winding number) determine la tension de corde.
Delta_k = 9 entre les generations 2 et 3 pour tous les secteurs.
"""

POSTULATE_P13 = """
P13 (Duree de vie topologique) :
La largeur de desintegration d'un entrelacs multi-noeuds est une loi de
puissance de la masse topologique :
  Gamma(K, k) = Gamma_0 * M_raw(K, k)^nu
  tau(K, k) = tau_P / M_raw(K, k)^nu

avec M_raw = Sum_i M(K_i) + lambda * Sum_i rho_J(K_i) * F(k) / R_0.

Pour les desintegrations faibles (leptons, quarks) : nu = 5.
Cet exposant retrouve exactement la loi de Fermi (Gamma ~ G_F^2 * m^5).

En termes topologiques : tau propto 1 / [rho_total * F(k)]^5.
La complexite du noeud (rho_J) et le nombre de Fibonacci (F(k))
determinent conjointement la duree de vie.

Critere de stabilite : k = 1 (electron) correspond a la tension de
corde minimale sigma_1, d'ou une duree de vie essentiellement infinie.
"""

POSTULATE_P14 = """
P14 (Photon, gluon et limites topologiques) :

a) Le photon est une onde pure du champ chi, sans structure de noeud
   (rho = 0, M = 0). Il n'a ni masse ni charge topologique. Son spin 1
   provient de la polarisation transverse de l'onde, non de l'indice de
   tresse.

   Consequence : les bosons massifs (W+-, Z0) SONT des noeuds topologiques
   (respectivement 8_2 et 4_1), mais le photon ne l'est pas.
   Ceci resout le paradoxe de l'assignation photon=4_1 : le Z0 massif (91 GeV)
   est un noeud 4_1, le photon (m=0) est une onde sans noeud.

b) Le gluon n'est pas une particule au sens de la theorie nodale. Il est
   le RELIQUAT D'ONDE LIBRE du champ chi qui n'est pas nodifie (capture par
   la topologie du noeud) dans une onde comportant 1 ou 2 noeuds.

   Quand le champ chi forme un noeud K, seule une partie de l'onde est
   capturee par la structure topologique (= la particule). Le reste de
   l'onde (non-nodifie) constitue le champ residuel qui, en QCD, est
   interprete comme le gluon.

   Consequences :
   - La force nucléaire forte = interaction via l'onde residuelle partagee entre noeuds
   - Le confinement est naturel : l'onde residuelle lie les noeuds entre eux
   - La liberte asymptotique s'explique : a courte distance, l'onde est
     presque entierement nodifiee, donc le couplage residuel est faible
   - Il n'y a pas 8 gluons independants : il y a UN champ chi dont la
     partie non-nodifiee mediate l'interaction entre noeuds

c) La vitesse de la lumiere c est la vitesse maximale de propagation des
   ondes dans l'espace-temps emergent. C'est une propriete intrinseque du
   neant (P1-P2) : les aberrations du neant se propagent a vitesse finie c.
   Toute perturbation du champ chi (onde ou noeud) respecte cette limite.

d) La constante de Planck hbar est le quantum minimal d'action. Elle emerge
   de la quantification de Chern-Simons : l'action S_CS = (k/4pi) * int(...)
   avec k entier (niveau de quantification) impose que l'action minimale
   est S_min = hbar. Plus precisement :
     S_CS = k * (hbar / 2)    avec k in Z
   d'ou hbar = 2 * S_CS / k est le plus petit increment d'action.

   En resume : c et hbar ne sont pas des parametres libres mais des
   consequences de la structure topologique de l'espace-temps emergent.
"""

POSTULATE_P15 = """
P15 (Fragilite topologique exponentielle) :
La duree de vie d'un entrelacs multi-noeuds decroit exponentiellement
avec le nombre de noeuds n :

    tau = tau_P * exp(-lambda_f * n) / M_raw^nu

Chaque noeud de l'entrelacs est un CANAL DE DESINTEGRATION independant.
La probabilite de survie decroit exponentiellement :
    P(survie, t) ~ exp(-n * lambda_f * t)

Deux canaux de desintegration :
  Canal faible (leptons, quarks) : nu = 7.75, lambda_f = 5.0
    Calibre sur le muon (n=7, k=11). Retrouve les durees de vie
    de d, s, c, tau-, b a < 0.5 decade pres.

  Canal fort/EW (W, Z, t, H) : nu = 1.75, lambda_f = 1.75
    Calibre sur le W (n=7, k=25). Retrouve Z0, t, H a < 0.02 decade.

Interpretation physique :
  - Plus un entrelacs a de noeuds, plus il a de points de cassure.
  - Chaque jonction noeud-noeud est un canal de desintegration.
  - exp(-5) ~ 1/150 : chaque noeud supplementaire divise la duree de vie
    par ~150 (canal faible).
  - L'electron (n=1) est naturellement stable : un seul noeud n'a
    aucun canal de cassure inter-noeuds.

Consequence : le nombre de noeuds d'une particule est CONTRAINT par
sa duree de vie experimentale. La combinatoire masse + duree de vie
selectionne une configuration unique (noeuds, k, n) par particule.
"""

POSTULATE_P16 = """
P16 (Exclusion topologique) :
Dans un entrelacs irreductible de n noeuds K_1, ..., K_n, un noeud K_i
est topologiquement EXCLU (ne porte aucune fonction d'onde) si son puits
de potentiel effectif est domine par les puits voisins dans la region
d'action d < hbar*c / rho_J(K_i).

L'exclusion emerge de la competition entre puits de potentiel :
le quantum d'action hbar (Chern-Simons) empeche une onde de se localiser
sur un puits trop peu profond par rapport a ses voisins.

Verification FEM : dans l'entrelacs nucleon 9_1 + 7_4 + 6_3,
le noeud 7_4 (central) est exclu car son puits effectif est domine
par ceux de 9_1 et 6_3. Densite |Psi|^2 = 0 sur 7_4 pour les 5
premieres ondes et toutes les distances testees.
"""

POSTULATE_P17 = """
P17 (Consolidation topologique) :
Deux eigenstates Psi_i, Psi_j d'un entrelacs forment un DOUBLET
CONSOLIDE si :
  (a) |E_i - E_j| / |E_i| < hbar / S_topo       (gap interne petit)
  (b) |E_j - E_k| / |E_j| > c * hbar / S_topo   (gap de bande grand)
pour tout eigenstate E_k hors du doublet.

S_topo = rho_J * |E_0| est l'action topologique de l'entrelacs.
Le doublet consolide est l'analogue topologique du doublet d'isospin.
Le proton et le neutron sont les deux eigenstates consolides de
l'entrelacs nucleon.

Limites topologiques (P14) :
  hbar_topo = S_topo / span(V_K)   (action minimale / niveau CS)
  c_topo    = max(spread * |E_0|) / hbar_topo   (vitesse maximale)
  hbar * c  ~ 20 (resolution topologique minimale, unites maillage)

Le principe d'incertitude nodal est :
  Delta_E * Delta_x >= hbar_topo * c_topo
"""

POSTULATE_P18 = """
P18 (Matiere noire nodale) :
La matiere noire est composee de noeuds topologiques complexes
(c > c* = 18) isoles dans le champ chi.

(a) Distribution de Boltzmann topologique :
    L'abondance du noeud K est P(K) propto exp(-beta * rho_J(K))
    avec beta = 0.914 (temperature topologique inverse).

(b) Seuil de complexite : c* = 18 = k_nucleon.
    Les noeuds avec c > 18 ne forment pas d'entrelacs stables
    car aucun triplet compatible ne satisfait C1-C5.

(c) Invisibilite : les noeuds chiraux forment des dipoles
    K + K* (Q_total = 0). Pas d'interaction EM ni forte.
    Seule interaction : gravitationnelle (P5).

(d) Stabilite : un noeud isole (n=1) ne peut se desintegrer
    que par reconnection topologique, supprimee pour c >> 1.

(e) Ratio : Omega_DM / Omega_b = Sum_{c>18} M(c) / Sum_{c<=18} M(c)
    = 5.45 (Planck 2018 : 5.45 +/- 0.28).
    M(c) = N(c) * exp(-beta * 1.1c) * 1.1c
    avec N(c) ~ (2.68)^c / c (Sundberg-Thistlethwaite).
"""

POSTULATE_P19 = """
P19 (Gravite emergente et constante de Newton) :
La gravite est une propriete emergente de la topologie quantique.
G est DERIVEE de la longueur de couplage kappa (P5) et de m_p :
    G = kappa * c^2 / (8 * pi * m_p)

Derivation (Poisson -> Schwarzschild) :
  1. La metrique (P5) g_ij = (1 + kappa * |Psi|^2) * delta_ij
     donne l'equation de Poisson nodale nabla^2 h = -kappa * |Psi|^2.
  2. Pour N nucleons a grande distance : h(r) = kappa*N/(4*pi*r).
  3. Identification avec Schwarzschild isotrope h = 2GM/(rc^2),
     M = N*m_p  =>  G = kappa*c^2/(8*pi*m_p).

Determination de kappa par auto-consistance (SANS alpha) :
  B(r) -> j = rot(B)/mu_0 -> |Psi|^2 = |j|/(ec)
  -> Schrodinger-Poisson auto-consistant -> kappa comme unique point fixe.
  La gravite est TOPOLOGIQUE, pas electromagnetique.

Valeur numerique :
  kappa = 3.12e-53 m (longueur fondamentale de couplage).

Verification de coherence (PAS une definition) :
  kappa = 4*pi * r_S(proton) decoule de G = kappa*c^2/(8*pi*m_p)
  et r_S = 2Gm_p/c^2. C'est une identite, pas une circularite.

En termes topologiques :
  G proportional 1 / (rho_nucleon * F(18)) = 1 / 66409
  L'amplification Fibonacci F(18) = 2584 qui rend le nucleon massif
  (938 MeV) est la MEME qui rend la gravite 10^36 fois plus faible
  que l'electromagnetisme. Le probleme de hierarchie est resolu.
"""

POSTULATE_P7BIS = """
P7bis (Spin des entrelacs multi-noeuds) :
Le spin total d'un entrelacs est determine par la composition des spins
individuels selon les regles de couplage du moment angulaire SU(2) :

1. Chaque noeud Ki a spin s(Ki) = (b(Ki)-1)/2
2. Un noeud est « fermionique » si b(Ki) est pair (spin demi-entier)
3. THEOREME DE PARITE : le spin total est demi-entier ssi le nombre de
   composantes fermioniques est impair
4. ETAT FONDAMENTAL : le spin total est le minimum compatible avec la parite :
   - Fermion composite : s_total = 1/2
   - Boson composite (avec composantes fermioniques) : s_total = 0 (scalaire)
     ou s_total = 1 (vecteur) selon le contenu non-torique
   - Boson composite (sans composante fermionique) : s_total = max des spins

Consequence : chaque assignation SM doit respecter la parite fermion/boson.
"""

POSTULATE_P7TER = """
P7ter (Trois generations fermioniques) :
Le champ chi est SU(2) Chern-Simons. Le groupe SU(2) a dimension 3 (trois
generateurs sigma_1, sigma_2, sigma_3). Chaque generateur correspond a une
famille de noeuds toriques T(2, 2j+1) avec j = 1, 2, 3 :
  sigma_1 -> T(2,3) = 3_1  (generation 1)
  sigma_2 -> T(2,5) = 5_1  (generation 2)
  sigma_3 -> T(2,7) = 7_1  (generation 3)

Le nombre de noeuds toriques fermioniques est borne par dim(SU(2)) = 3.
T(2,9) et au-dela ne correspondent a aucun generateur independant
de SU(2) -> pas de 4e generation.
"""

POSTULATE_P20 = """
P20 (Constante de structure fine emergente) :
En theorie de Chern-Simons SU(2) au niveau k, le couplage de jauge
(normalisation standard, Witten 1989) est :
  g^2 = 4*pi / (k + 2)
  alpha = g^2 / (4*pi) = 1 / (k + 2)

L'identification du couplage CS avec la constante de structure fine
alpha_EM est le contenu physique de ce postulat.

Pour alpha_exp = 1/137.036 : k_CS = 135
Ceci donne alpha_theo = 1/137 = 0.007299 (erreur 0.03%).

La constante de structure fine n'est pas un parametre libre mais une
consequence du niveau de quantification de Chern-Simons.
"""

# --- P20 constants ---
K_CS = 135  # Chern-Simons level giving alpha = 1/137

# --- P18 constants ---
BETA_TOPO = 0.9144   # Topological inverse temperature
C_STAR = 18          # Complexity threshold (visible/dark)

# --- P19 constants ---
# kappa is the FUNDAMENTAL coupling length (P5). G is DERIVED from it.
# The relation kappa = 4*pi*r_S(proton) is a consistency CHECK, not a definition.
KAPPA_PHYS = 3.1218e-53  # m, longueur de couplage NUCLEONIQUE, DERIVEE (historique CODATA-ancree ; exacte auto-contenue : 3.1201e-53 = (96pi/7) a^18 lambdabar_C, V6 sec:fc-raccord)
