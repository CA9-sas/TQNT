"""La distribution des elements lourds dans le plasma BBN peut-elle
sauver le mecanisme Li-7 ? Analyse honnete (24/07, question utilisateur).

L'IDEE : si les elements lourds (7Be) se rapprochent entre eux dans le
plasma, la probabilite locale d'etre frappe par un photon augmente --
le mur d'efficacite (x1e4) pourrait tomber.

QUATRE TESTS, TOUS CHIFFRES (epoque utile t ~ 3e4 s, T ~ 2 keV) :
 [D1] Y A-T-IL UNE FORCE DE REGROUPEMENT ? Couplage plasma, Debye,
      gravite, et les canaux TQNT (CGQ, forces chi).
 [D2] LE SIGNE : entre IONS (tous positifs), Coulomb est REPULSIF --
      la correlation est un TROU, pas un amas.
 [D3] MEME EN L'ACCORDANT : la destruction est LINEAIRE en n_Be et le
      photon moyenne sur son libre parcours Compton -- redistribuer
      le 7Be a total fixe ne change PAS la moyenne, sauf si les
      EMETTEURS (nos paires sombres) etaient co-localises avec les
      amas -- or ils sont distribues comme la matiere noire.
 [D4] OU L'INTUITION EST JUSTE : la sedimentation des lourds EXISTE --
      dans les ETOILES (diffusion atomique/gravitational settling sur
      des Gyr) : c'est precisement l'explication astrophysique
      dominante du plateau de Spite. La TQNT, standard-identique dans
      les etoiles, en herite naturellement.
"""
import math

# epoque t ~ 3e4 s
T_KEV = 2.0
T_K = T_KEV * 1.16e7
N_GAMMA = 0.244 * (T_KEV * 1e3 / 197.327e-7) ** 3   # cm^-3 (T/hbar c)
ETA_B = 6.1e-10
N_B = ETA_B * N_GAMMA
N_E = 0.88 * N_B
Z_BE = 4
E2 = 1.44e-7           # e^2 en eV cm
KT_EV = T_KEV * 1e3
SIG_KN = 1.63e-25      # cm^2 (2 MeV)


def main():
    print(__doc__)
    print("=" * 78)
    print("[D1] LES FORCES DISPONIBLES (t ~ 3e4 s, T = 2 keV)")
    print("=" * 78)
    a = N_B ** (-1 / 3)
    gam = Z_BE**2 * E2 / (a * KT_EV)
    lam_d = math.sqrt(KT_EV / (4 * math.pi * N_E * E2))
    print("""  n_b = %.1e cm^-3 ; distance inter-ions a = %.1e cm
  couplage plasma Gamma = Z^2 e^2/(a kT) = %.1e
     -> plasma ULTRA-faiblement couple : correlations ioniques
        negligeables (il faudrait Gamma >~ 1 pour structurer) ;
  longueur de Debye = %.1e cm : au-dela, AUCUNE interaction ;
  gravite : aucun gradient (univers homogene a 1e-5, domine par la
     radiation -- pas de puits ou « couler ») : sedimentation SANS
     OBJET a cette epoque ;
  canaux TQNT : le CGQ entre noyaux est l'empilement des deplacements
     (echelle kappa_g ~ 1e-39 : rien) ; les forces chi-chi sont
     supprimees (etabli) ; la danse thermalise mais n'attire pas.
  -> AUCUNE force de regroupement disponible, standard ou TQNT.""" % (
        N_B, a, gam, lam_d))

    print("=" * 78)
    print("[D2] LE SIGNE : les ions se REPOUSSENT")
    print("=" * 78)
    print("""  Le 7Be (Z = 4) et les protons sont tous POSITIFS : l'interaction
  ion-ion est repulsive (ecrantee a Debye). La correlation de paire
  g(r) presente un TROU de correlation (g < 1 a courte distance),
  pas un amas : l'effet disponible a le SIGNE CONTRAIRE de
  l'intuition -- les lourds s'EVITENT (d'un facteur ~Gamma ~ 1e-4,
  minuscule de surcroit). Le regroupement des lourds par gravite
  n'existe que la ou il y a un puits : dans les etoiles ([D4]).""")

    print("=" * 78)
    print("[D3] MEME EN ACCORDANT UN REGROUPEMENT : LA MOYENNE NE BOUGE PAS")
    print("=" * 78)
    lam_c = 1 / (SIG_KN * N_E)
    print("""  La probabilite de destruction est LINEAIRE en n_Be : pour un flux
  de photons uniforme, P ~ sigma_Be <n_Be>/(sigma_C n_e) -- et
  REDISTRIBUER le 7Be a total fixe ne change pas <n_Be>. Deux seules
  echappatoires, toutes deux fermees :
   (a) amas optiquement epais en destruction : il faudrait P_local
       ~ 1, soit une surdensite locale ~3e12 -- absurde ;
   (b) EMETTEURS co-localises avec les amas : nos paires metastables
       sont des objets SOMBRES, distribues comme la matiere noire,
       sans aucune correlation avec le 7Be ; et le photon moyenne de
       toute facon sur son libre parcours Compton
       lambda_C = %.1e cm (~%.0f millions de km !) : toute
       structure plus petite est integralement moyennee.
  -> le mur x1e4 est INSENSIBLE a la distribution spatiale.""" % (
        lam_c, lam_c / 1e11))

    print("=" * 78)
    print("[D4] OU L'INTUITION EST JUSTE : LES ETOILES")
    print("=" * 78)
    print("""  « Les elements les plus lourds se rapprochent » -- c'est vrai LA
  OU IL Y A UN PUITS : dans les etoiles du plateau de Spite. La
  sedimentation gravitationnelle (diffusion atomique) + le melange
  turbulent font COULER le 7Li sous la photosphere sur des Gyr :
  c'est l'explication astrophysique dominante du deficit observe
  (facteur ~2-3 possible, debattu sur l'uniformite du plateau). La
  TQNT etant standard-identique dans les etoiles, elle HERITE de ce
  mecanisme tel quel : le probleme du lithium n'est peut-etre pas
  cosmologique du tout -- et c'est la lecture la plus economique
  qui reste ouverte.""")

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("""  L'analyse honnete ferme la voie « distribution dans le plasma » :
   [D1] aucune force de regroupement (Gamma ~ 1e-4, pas de gradient,
        canaux TQNT nuls) ;
   [D2] le signe est CONTRAIRE (les ions s'evitent) ;
   [D3] meme en l'accordant, la destruction est lineaire en n_Be et
        le photon moyenne sur ~1e10 cm : la moyenne est invariante ;
   [D4] MAIS l'intuition physique (les lourds coulent) est la bonne
        -- une epoque plus tard : la sedimentation STELLAIRE est
        l'explication survivante du plateau, et la TQNT en herite.
  Le mur d'efficacite tient ; le Li-7 reste herite -- avec la
  redirection stellaire comme lecture la plus probable.""")


if __name__ == "__main__":
    main()
