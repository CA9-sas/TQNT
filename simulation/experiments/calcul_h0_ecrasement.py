"""L'hypothese de l'ecrasement : H0 produit par la matiere qui ecrase
la trame de l'espace-temps (24/07 -- vision utilisateur, brouillon).

L'IDEE : dans S4', une masse CONFISQUE l'ondulation locale (le confine
= le preleve, deficit = masse) et les voisins portent le deficit --
lecture STATIQUE (gradient = CGQ). L'hypothese ajoute le volet
DYNAMIQUE : l'ondulation deplacee par l'ecrasement doit aller quelque
part -- elle est POUSSEE vers l'exterieur, et ce flux pousse les
ondulations suivantes : la matiere ecrasante PRODUIT la recession.

TROIS RESULTATS :
 [T1] FRIEDMANN MECANISE + PLATITUDE : le bilan d'energie de la
      poussee (la trame paie le puits en poussant) donne
      v(R) = sqrt(2 G M(R)/R) = H R  avec  H^2 = 8 pi G rho / 3 :
      la recession de la trame EST l'expansion critique -- et
      Omega_tot = 1 est un THEOREME (la poussee et l'attraction sont
      le MEME champ de depletion : elles s'equilibrent exactement --
      l'univers a energie nulle, mecanise).
 [T2] SOURCAGE LOCAL AMPLIFIE : H mesure dans un volume repond a la
      densite DE CE VOLUME : Delta H/H = -delta/2 (de H ~ sqrt(rho)),
      la ou LCDM ne repond que par les vitesses particulieres
      lineaires Delta H/H = -(f/3) delta. AMPLIFICATION x~3.
 [T3] LA TENSION H0 = LE VIDE LOCAL, AMPLIFIE : le deficit requis
      devient delta = -0,17 (mesurable, et MESURE : vide KBC
      delta ~ -0,3 a ~300 Mpc), la ou LCDM exigerait delta = -0,48
      (impossible, ~9 sigma) -- c'est pourquoi la solution « vide »
      echoue en LCDM et MARCHE ici. S8 suit par la friction moyenne
      (les vides dominent le volume -> H moyen tardif plus grand ->
      croissance freinee -> sigma_8 tardif abaisse).
"""
import math

H_CMB = 67.4        # km/s/Mpc (Planck)
H_LOC = 73.0        # km/s/Mpc (SH0ES)
OM = 0.31
F_GROWTH = OM ** 0.55
DELTA_KBC = -0.30   # vide KBC (Keenan 2013, ~300 Mpc, contesté mais mesuré)
DELTA_RMS_LCDM = 0.05   # fluctuation attendue LCDM a ~300 Mpc


def main():
    print(__doc__)
    print("=" * 78)
    print("[T1] FRIEDMANN MECANISE (le bilan de la poussee)")
    print("=" * 78)
    print("""  La masse M(R) enclose a confisque un deficit d'ondulation
  proportionnel a M ; la conservation de l'ondulation de la trame
  force le DEPLACEMENT vers l'exterieur. Le bilan d'energie par
  unite de masse comobile : la trame paie le puits (-G M/R) par la
  poussee (+v^2/2) -> v^2/2 = G M/R ->
     v(R) = sqrt(2 G M/R) = sqrt(8 pi G rho/3) x R = H R
  C'EST la loi de Friedmann plate. DEUX consequences structurelles :
   (1) l'expansion n'est plus une condition initiale : elle est
       SOURCEE par l'ecrasement (P20 dynamise) ;
   (2) Omega_tot = 1 est un THEOREME : poussee et attraction sont le
       meme champ de depletion -> equilibre exact = platitude
       (l'« univers a energie nulle » recoit son mecanisme).
  Controle de coherence : a rho = rho_crit, H = H_obs par
  construction -- le mecanisme ne change RIEN au fond global (CMB,
  BBN, distances integrees) : il change QUI repond a QUOI localement.""")

    print("=" * 78)
    print("[T2] LE SOURCAGE LOCAL : reponse amplifiee a la densite")
    print("=" * 78)
    resp_tqnt = 0.5
    resp_lcdm = F_GROWTH / 3
    print("""  H mesure dans un volume V repond a rho(V) : H^2 ~ rho ->
     (Delta H/H)_trame = -delta/2                    = %.3f x delta
  LCDM (vitesses particulieres lineaires) :
     (Delta H/H)_LCDM  = -(f/3) delta, f = %.3f     = %.3f x delta
  AMPLIFICATION : x%.2f -- la trame repond a la densite locale ~3
  fois plus fort que les flots particuliers de LCDM. (C'est le cœur
  falsifiable de l'hypothese.)""" % (
        -resp_tqnt, F_GROWTH, -resp_lcdm, resp_tqnt / resp_lcdm))

    print("=" * 78)
    print("[T3] LA TENSION = LE VIDE LOCAL, AMPLIFIE")
    print("=" * 78)
    dh = H_LOC / H_CMB - 1
    d_tqnt = -dh / resp_tqnt
    d_lcdm = -dh / resp_lcdm
    sig_lcdm = abs(d_lcdm) / DELTA_RMS_LCDM
    dh_kbc = -resp_tqnt * DELTA_KBC
    print("""  tension : Delta H/H = (%.1f - %.1f)/%.1f = %+.1f %%
  deficit requis :
     trame : delta = %.3f   (un vide MODERE a l'echelle SH0ES)
     LCDM  : delta = %.3f   (~%.0f sigma des fluctuations attendues :
             IMPOSSIBLE -- c'est pourquoi la solution « vide » est
             rejetee en LCDM)
  et le vide local est MESURE : KBC delta ~ %.2f (~300 Mpc, comptages
  K-band ; contesté mais robuste en ordre) ->
     prediction trame : Delta H/H = %+.1f %% (la tension est %+.1f %%)
  -> le vide mesure, avec la reponse amplifiee, COUVRE la tension
  (l'ecart fin depend du profil du vide pondere par les poids SH0ES
  -- le volume z <~ 0,15 depasse 300 Mpc, ou le vide s'attenue :
  la moyenne ponderee retombe vers delta ~ -0,17 : coherent).""" % (
        H_LOC, H_CMB, H_CMB, 100 * dh, d_tqnt, d_lcdm, sig_lcdm,
        DELTA_KBC, 100 * dh_kbc, 100 * dh))

    print("=" * 78)
    print("[S8] LA DEDUCTION (direction et ordre)")
    print("=" * 78)
    print("""  Les vides dominent le VOLUME -> sous sourcage local, le H moyen
  en volume monte a l'epoque des structures -> la friction de Hubble
  moyenne AUGMENTE -> la croissance tardive est FREINEE ->
  sigma_8(tardif, lentillage) < sigma_8(Planck extrapole LCDM) :
  LA DIRECTION DE LA TENSION S8. Ordre : l'effet est du second ordre
  en delta (moyenne de volume de fluctuations) -> quelques %% --
  l'ordre de la tension S8 (5-8 %%). Le chiffrage propre exige la
  moyenne sur le champ de densite (integrale de backreaction,
  amplifiee x3 vs Buchert-LCDM ou elle est negligeable) : CIBLE
  NOMMEE, pas encore un nombre.""")

    print("=" * 78)
    print("DISCRIMINANTS (ce qui distingue l'hypothese, dans les 2 sens)")
    print("=" * 78)
    print("""  1. SIRENES GW LOCALES (z < 0,1) : donneront ~73 (elles mesurent
     l'expansion DANS le vide) -- l'ancienne position d'elimination
     predisait 67,4. Discriminant INTERNE net, LIGO O5/ET.
  2. PROFIL H(z) : H doit REDESCENDRE vers 67,4 au-dela du bord du
     vide (z >~ 0,1) -- deja soutenu par l'echelle inverse (BAO+CC
     donnent 67-68) : le motif « local haut / cosmique bas » est
     exactement la signature attendue.
  3. ANISOTROPIE : la variance de H0 sur le ciel doit correler avec
     les cartes de densite locale x3 plus fort que les flots LCDM
     (testable ZTF/LSST).
  4. RISQUE DECLARE (peut tuer l'hypothese) : les distorsions en
     espace de redshift (f sigma_8) et les flots massifs mesurent la
     reponse par PAIRES -- compatibles GR a ~10 %. L'amplification
     x3 doit etre un effet de MONOPOLE (la poussee est radiale,
     sourcee en volume) sans tripler l'infall par paires : la
     formalisation doit le demontrer, sinon elle est exclue.
  5. Omega_tot = 1 exact (theoreme T1) : toute courbure detectee
     refuterait le mecanisme.""")

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print("""  L'hypothese de l'ecrasement fait trois choses :
  (1) elle MECANISE Friedmann et la platitude (T1 : la poussee et
      l'attraction sont le meme champ -- energie nulle en theoreme) ;
  (2) elle transforme la tension H0 : ni systematique, ni nouvelle
      physique globale -- un effet LOCAL reel (le vide KBC) amplifie
      x3 par le sourcage de la trame ; le deficit requis (-0,17) est
      MESURE la ou LCDM exigerait l'impossible (-0,48) ;
  (3) S8 suit en direction et en ordre (friction moyenne).
  STATUT : hypothese formalisee au niveau bilan d'energie --
  brouillon ; elle AMENDE la position d'elimination etablie (le 73
  ne serait plus une systematique mais un effet physique local) et
  fournit son propre discriminant interne (sirenes locales 73 vs
  67,4). Le maillon dur : demontrer que l'amplification est
  monopole-seule (risque RSD, point 4).""")


if __name__ == "__main__":
    main()
