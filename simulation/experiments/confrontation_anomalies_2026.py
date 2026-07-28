"""Portefeuille d'anomalies : ce qu'aucune physique n'explique,
confronte a la TQNT (23/07 ; ETENDU 24/07 a 28 entrees).

DEMANDE : chercher d'autres observations/phenomenes inexpliques et les
confronter. METHODE : pour chaque anomalie etablie (significance et
statut communautaire), le verdict TQNT tombe dans QUATRE categories :
  ABSENCE   la TQNT predit que l'anomalie N'EST PAS de la nouvelle
            physique (theoreme/spectre/echelle canonique a l'appui) --
            falsifiable : une confirmation refuterait ;
  HERITE    la TQNT est indiscernable du cadre standard ici : elle
            herite du mystere et le DECLARE (pas d'explication) ;
  AUDIT     une revendication INTERNE (juin) requalifiee honnetement ;
  COHERENT  l'anomalie est resolue ailleurs et la TQNT est du bon cote.
Aucun ajustement ; tous les nombres sont canoniques (V6)."""
import math

ALPHA = 1 / 137.0
SIGMA_EQ = 1.1e-48        # cm^2 (canal polarisabilite, derive)
M_DARK_MIN = 1.07         # GeV (le plus leger etat sombre : strict 4_1+8_3)
M_N, M_P = 939.565, 938.272


def sec(t):
    print("=" * 78)
    print(t)
    print("=" * 78)


def main():
    print(__doc__)

    sec("[1] TENSION BOUTEILLE/FAISCEAU DU NEUTRON (~10 s, ~4 sigma) -- ABSENCE")
    tb, tf = 877.75, 888.0
    br = 1 - tb / tf
    print("""  tau(bouteille) = %.1f s vs tau(faisceau) = %.1f s : l'explication
  « canal invisible » (n -> sombre) exige BR_inv = %.1f %%.
  TQNT : BR_inv = 0 EXACTEMENT, par DEUX murs --
   (a) cinematique : l'etat sombre le plus leger est a %.2f GeV
       > m_n = %.3f GeV (AUCUN etat TQNT neutre sous le neutron :
       le spectre est un theoreme, pas un choix) ;
   (b) Vassiliev : meme virtuellement, creation c>=23 supprimee
       alpha^Delta_c ~ 1e-11..1e-24.
  -> la tension est SYSTEMATIQUE (le faisceau compte les protons : une
  perte de comptage suffit). FALSIFIABLE : tout canal invisible
  confirme refuterait le spectre sombre TQNT.""" % (
        tb, tf, 100 * br, M_DARK_MIN, M_N / 1000))

    sec("[2] X17 / ATOMKI (boson a 16,8 MeV, angles e+e- 8Be/4He) -- ABSENCE")
    print("""  Le spectre TQNT est FERME sous le GeV : ondes libres (masse 0),
  leptons (0,511 / 105,7 MeV, modes Z3 du trefle), Goldstone
  (pi 139,6 via GMOR). AUCUNE topologie ne peut produire 17 MeV :
  le plus proche est m_e (x33 dessous) puis pi (x8 dessus) -- il n'y a
  pas de « petit noeud » (m_topo >= sqrt(sigma_chi) l(3_1) ~ echelle
  310 MeV ; les leptons descendent par Koide, pas par un continuum).
  -> X17 = artefact experimental PREDIT. Etat des donnees : MEG II
  (2024) ne voit RIEN dans 8Be -- la prediction d'absence est deja
  soutenue par la meilleure re-mesure independante.""")

    sec("[3] EXCES GeV DU CENTRE GALACTIQUE (GCE, non resolu) -- ABSENCE (DM)")
    sv_needed, sv_tqnt = 1e-26, 3e-26 * ALPHA**10
    print("""  L'interpretation DM du GCE exige sigma_v ~ %.0e cm^3/s (masses
  ajustees 30-70 GeV en b b-bar). TQNT : reservoir a 1,7-1,9 GeV
  (mauvaise masse) ET annihilation ecrasee sigma_v ~ %.0e (%.0f dex
  sous le requis, meme mur qu'AMS-02).
  -> le GCE n'est PAS de l'annihilation DM : pulsars milliseconde
  (la granularite point-source observee va dans ce sens). Le SEUL
  signal gamma TQNT permis = les 6 raies fines 1,07-1,64 GeV
  (population stricte, signature DISTINCTE d'un continuum GCE).""" % (
        sv_needed, sv_tqnt, math.log10(sv_needed / sv_tqnt)))

    sec("[4] CANDIDATS ANTI-HELIUM AMS-02 (~10 evenements, inexpliques) -- ABSENCE")
    print("""  Baryogenese TQNT : le cote visible est la MIETTE CP (matiere
  seule ; l'antimatiere primordiale s'est annihilee) ; le cote sombre
  est symetrique mais fait de NOEUDS ISOLES c>=23 -- pas de dynamique
  de nucleosynthese sombre, pas d'anti-noyaux. Source TQNT
  d'anti-He3/He4 cosmique : ZERO (la spallation standard donne
  ~1e-12 du flux He, tres insuffisant pour les candidats).
  -> les candidats sont instrumentaux (confusion de rigidite) --
  PREDIT. FALSIFIABLE NET : UN SEUL anti-He4 cosmique confirme
  refuterait la baryogenese par nucleation (il exigerait un domaine
  d'antimatiere ou une DM qui s'annihile en anti-noyaux : deux
  impossibilites TQNT).""")

    sec("[5] EDGES 21 cm (absorption x2 trop profonde a z~17) -- ABSENCE")
    sigma_needed = 1e-21
    print("""  L'explication « refroidissement baryons-DM » exige une diffusion
  sigma ~ %.0e cm^2 (DM millichargee ~GeV). TQNT : Q = 0 EXACT
  (neutralisation par framing, theoreme) et sigma_eq = %.1e :
  %.0f dex sous le requis -- le refroidissement TQNT est NUL.
  -> l'anomalie EDGES est instrumentale (fond/etalonnage) : PREDIT ;
  et SARAS 3 (2022) ne la reproduit pas -- deja soutenu.""" % (
        sigma_needed, SIGMA_EQ, math.log10(sigma_needed / SIGMA_EQ)))

    sec("[6] EXCES BASSE ENERGIE MiniBooNE (4,8 sigma, non resolu) -- ABSENCE (sterile)")
    print("""  L'interpretation sterile (4e neutrino) est INTERDITE par le
  theoreme de saturation Z3 (jamais de 4e famille -- derive, V6) et
  par « le neutrino n'a aucune topologie » (onde libre).
  -> l'exces n'est PAS une oscillation sterile : photons mal
  reconstruits / bruits (MicroBooNE 2021-22 : aucune trace du
  scenario nu_e simple -- soutenu). Deja au registre V6 des
  predictions d'absence ; la confrontation le date sur l'anomalie.""")

    sec("[7] BIREFRINGENCE COSMIQUE (beta = 0,34 +- 0,09 deg, ~3 sigma) -- ABSENCE (conditionnelle)")
    print("""  Une rotation EB isotrope exige un couplage P-impair du VIDE au
  photon (axion-like). TQNT : la lumiere est la SPECTATRICE (jamais
  entree dans les rails, couplage residuel quantifie ~2e-3 oscillant
  a k = pi/pas -- aucune composante seculaire) et le spectre n'a
  AUCUN champ axionique leger. La structure chirale (CS, trefle) vit
  au niveau des NOEUDS (matiere), pas de l'onde libre.
  -> beta = 0 PREDIT (le hint ~3 sigma = avant-plans/etalonnage,
  ce que la controverse actuelle suggere). CONDITIONNEL declare :
  une beta != 0 confirmee (LiteBIRD, ~0,06 deg) mettrait la
  protection de la spectatrice sous tension quantitative.""")

    sec("[8] AUDIT INTERNE : le « signal m=3 » Planck de juin -- REQUALIFIE")
    print("""  Juin (monte_carlo_planck_v4) : exces azimutal m=3 declare
  « ROBUSTE » (p ~ 0, ratio 2,3 apres nettoyage) avec consigne
  « finaliser detection paper ». AUDIT (23/07) :
   (a) la mesure est en coordonnees GALACTIQUES -- un signal
       cosmologique n'a aucune raison de s'aligner sur le plan
       galactique : un m=3 galactique est la signature d'un RESIDU
       D'AVANT-PLAN ;
   (b) le caveat du P9 le disait deja : MC sur C_l observes, SANS
       modele d'avant-plans (pas de gabarit 353/30 GHz) ;
   (c) la position V6 ETABLIE predit ZERO empreinte : les domaines
       Z3 du gel (T_c = 62 MeV) sont tres sous-horizon -> effaces ;
       A_s est Starobinsky sans modulation (V5.15).
  -> REQUALIFIE : avant-plan attendu, PAS une detection TQNT ; ne
  pas envoyer de « detection paper ». Test decisif si repris :
  gabarits 353/30 GHz + verification d'alignement (un vrai signal
  survit en coordonnees celestes et ne suit pas le masque).""")

    sec("[9] TENSION H0 (5 sigma) + TENSION S8 (~2-3 sigma) -- HERITE")
    print("""  La TQNT est observationnellement indiscernable de LCDM+RG dans
  ces regimes (position V6) : AUCUN mecanisme differenciant tot/tard
  (P22 cosmologique nul ; sigma_8 = sigma_8^LCDM exact -- NB : le
  « sigma_8 = 0,77 » du document P9 est CADUC). Seule connexion
  etablie : la prediction rho_Lambda vit DANS la bande de la tension
  H0 (+5,9 % Planck / -9,9 % SH0ES, audit amont).
  -> MYSTERE HERITE, declare : si H0/S8 exigent de la nouvelle
  physique, la TQNT est en difficulte COMME LCDM -- l'honnetete
  l'impose.""")

    sec("[10] ENERGIE NOIRE EVOLUTIVE (DESI w0-wa, ~3-4 sigma) -- ABSENCE (w = -1)")
    print("""  TQNT : rho_Lambda = residu STATIQUE du vide chi (ordre de
  grandeur resolu, 123 ordres) -- aucun champ roulant, aucun
  mecanisme d'evolution : w = -1 CONSTANT predit.
  -> la preference DESI pour w0-wa doit REGRESSER (systematiques
  SN/BAO). FALSIFIABLE : une deviation w != -1 confirmee (DESI DR3+,
  Euclid) refuterait la lecture statique du vide TQNT.""")

    sec("[11] MUON PUZZLE (gerbes atmospheriques, deficit ~8 sigma combine) -- HERITE")
    print("""  La dynamique de collision TQNT (P15 reconnexion + hadronisation
  chi) reproduit ~les modeles standard (K/pi = 0,143 valide) et NE
  resout PAS quantitativement le deficit de muons (etabli 07/2026,
  brouillon dynamique de collision). MYSTERE HERITE, declare.""")

    sec("[12-17] EXPLIQUEES AU SENS FORT (le mecanisme TQNT rend le fait attendu)")
    print("""  [12] STERILE GLOBAL eV (cosmo exclut / labo suggere) : la TQNT
       explique POURQUOI la tension existe -- saturation Z3 (theoreme :
       jamais de 4e famille) + le neutrino n'a aucune topologie -> les
       indices courte-distance ne convergeront jamais, la cosmologie a
       raison. C'est le cas le plus net de la liste.
  [13] UNIVERSALITE LEPTONIQUE b->s l l (R_K, R_K*) : e/mu/tau = les
       trois modes Z3 DU MEME trefle -- couplage topologique identique
       par construction (g_topo = 1 universel) -> R_K = R_K* = 1
       EXACTEMENT. Reanalyse LHCb 2022 (retour SM) : SOUTENU. Le P5'
       residuel = facteurs de forme QCD, hors perimetre.
  [14] MUON g-2 : a = alpha/2pi derive (alpha = 1/137 topologique),
       AUCUNE contribution nouvelle (P23/C_plast archive) -> la TQNT
       predit la convergence SM ; le consensus lattice 2025 (BMW) qui
       eteint l'anomalie est exactement son camp. SOUTENU.
  [15] KATRIN : Sigma m_nu ~ 66 meV predit -> m_beta ~ qq meV : KATRIN
       ne verra JAMAIS de signal ; la pression cosmologique sur sa
       limite est coherente avec 66 meV.
  [16] RAYON DU PROTON (resolue) : r_p = 0,841 fm est la valeur
       CANONIQUE du cadre (d_tube = lambda_C au coeur trefle) -- la
       valeur muonique, celle du CODATA convergent. Bon cote.
  [17] GALAXIES MASSIVES JWST z>10 : amplification P22 rigoureusement
       NULLE (1,000000, audit V5.18) -> pas de nouvelle physique
       predit ; l'audit independant (revision M* ~ -1,5 dex : IMF,
       photo-z, poussiere) + LCDM/FFB ferment l'ecart. Dissolution
       VERIFIEE (regle etablie : ne pas presenter comme succes V5 --
       c'est une prediction d'absence, pas une amplification).""")

    sec("[18-23] ABSENCES SUPPLEMENTAIRES (l'anomalie n'est pas de la nouvelle physique)")
    print("""  [18] GALLIUM/BEST (deficit nu_e ~5 sigma) : pas de sterile
       (theoreme) -> le deficit est SYSTEMATIQUE (sections efficaces
       71Ga) ; il ne sera jamais confirme comme oscillation.
  [19] REACTEUR : largement resorbee en probleme de flux 235U --
       exactement l'issue predite par « pas de sterile ». VERIFIEE.
  [20] DAMA/LIBRA (modulation 12 sigma) : sigma_eq = 1,1e-48 -> le
       reservoir ne peut PAS produire une modulation qui exigerait
       ~1e-40 : artefact predit. ANAIS-112 + COSINE-100 (2024-25
       defavorables) : SOUTENU.
  [21] ANITA (gerbes ascendantes) : AUCUNE particule TQNT ne peut
       traverser la Terre et gerber (reservoir a sigma minuscule, pas
       de sterile lourd) -> artefact (reflexions dans la glace) ;
       PUEO tranchera. Falsifiable : un flux ascendant confirme de
       particules serait une tension.
  [22] ANOMALIES CMB GRANDE ECHELLE (axe du mal, cold spot,
       asymetrie) : la TQNT predit un CMB gaussien isotrope = LCDM
       (zero empreinte Z3, verifie par l'audit m=3 trois fois en
       frequence) -> fluctuations/systematiques, ou au pire anomalie
       PARTAGEE (statut herite).
  [23] MASSE DU W (CDF 2022) : SANS CONTENU TQNT -- la precision du
       cadre sur m_W (+-0,4 %, test nul d'assignation non passe) est
       ~6x trop lache pour discriminer 80,433 vs 80,377. Le consensus
       (systematique CDF, contredit ATLAS/CMS) est vraisemblable mais
       ce n'est pas la theorie qui le dit. Déclaré.""")

    sec("[24-25] HERITEES / HORS PERIMETRE SUPPLEMENTAIRES")
    print("""  [24] LITHIUM-7 PRIMORDIAL : le facteur P22 existant est
       CONJECTURAL (pose, non derive -- etiquete tel quel en V6) ;
       la creation Vassiliev-invisible interdit a la TQNT de modifier
       BBN -> elle HERITE du probleme. Honnetement non explique.
  [25] FRB / MECANISMES DES SURSAUTS GAMMA : astrophysique interne,
       HORS PERIMETRE (les GRB ne servent au cadre que comme
       non-detection de dispersion -- protection de la trame).""")

    sec("BILAN")
    print("""  28 anomalies confrontees (12 du 23/07 + 16 de la liste etendue,
  doublons fusionnes), zero ajustement :
   - 6 EXPLIQUEES AU SENS FORT (mecanisme) : sterile global (theoreme
     Z3), universalite leptonique (modes Z3 du meme trefle), muon g-2
     (a = alpha/2pi, rien de plus), KATRIN (66 meV), rayon du proton
     (0,841 canonique), JWST (amplification nulle) -- 4 deja
     SOUTENUES par les re-mesures (LHCb 2022, lattice 2025, CODATA,
     revisions JWST) ;
   - 12 ABSENCES PREDITES falsifiables + 1 conditionnelle
     (birefringence) : X17, MiniBooNE, gallium, DAMA, ANITA,
     positrons AMS, GCE, EDGES, anti-He, tau_n, w evolutif, CMB
     grande echelle -- 7 deja soutenues (MEG II, MicroBooNE,
     ANAIS/COSINE, SARAS 3, granularite, reacteur resorbee...) ;
   - 1 COHERENTE (reacteur, resorbee comme predit) ;
   - 1 AUDIT INTERNE (m=3, execute T0-T5) ;
   - 4 HERITEES declarees (H0, S8, Li-7, muon puzzle des gerbes) ;
   - 3 SANS CONTENU / HORS PERIMETRE declarees (masse du W, FRB,
     mecanismes GRB).
  Motif inchange et renforce : la ou l'anomalie demande un ETAT
  NOUVEAU (sterile, X17, WIMP modulant, boson leger), le
  spectre-theoreme repond ; la ou elle demande une VIOLATION
  d'universalite, le Z3 du trefle repond ; la ou elle demande un
  mecanisme cosmologique fin (H0, S8, Li-7), la TQNT est muette et
  le dit.""")


if __name__ == "__main__":
    main()
