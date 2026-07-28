"""L'elasticite d'echelle de la trame (24/07 -- raffinement utilisateur
de l'hypothese de l'ecrasement).

L'IDEE : quand la masse pese sur la trame, l'ondulation est poussee ;
quand la masse se retire, elle REVIENT (reponse elastique). Localement
on ne voit quasiment pas d'ecart -- la trame change mais reste la meme
localement (nos regles sont faites de la meme trame : on ne peut pas
mesurer sa propre echelle -- la « relativite d'echelle »). Au lointain,
le cumul est la seule chose qu'on voit.

CE QUE CELA MECANISE (trois acquis reels) :
 [A1] LES SYSTEMES LIES NE S'ETENDENT PAS : la ou le motif
      d'ecrasement est STATIQUE (systeme lie : les masses ne se
      retirent pas), la trame tient sa forme comprimee -- pas de
      poussee seculaire locale. La vieille question « pourquoi
      l'atome/la Galaxie ne s'etend pas » recoit son mecanisme
      (au lieu de l'argument standard « les systemes lies decouplent
      du flot », qui est un resultat, pas un mecanisme).
 [A2] ZERO DERIVE LOCALE : l'elasticite + la relativite d'echelle
      (regles et horloges faites de la meme trame) => aucune
      observable locale ne derive -- c'est la LECTURE MECANISEE des
      protections etablies (Oklo, horloges, GPS, quasars).
 [A3] LE DECALAGE VERS LE ROUGE EST CUMULATIF : l'expansion n'est
      observable QU'AU LOINTAIN (l'integrale des poussees le long de
      la ligne de visee = l'etirement metrique de T1 pour les chemins
      non lies) -- le fait standard « l'expansion est invisible
      localement » devient un theoreme de l'elasticite.

LE TEST HONNETE POUR LA TENSION H0 (deux versions chiffrees) :
 [B1] TRANSITION D'ECHELLE (elastique -> cumulatif au rayon de
      deliaison, ~qq Mpc) : SH0ES mesure a 50-600 Mpc, tres au-dela
      -> plein regime cumulatif -> AUCUN effet sur la tension.
 [B2] POUSSEE SOURCEE PAR LA STRUCTURE (les puits s'approfondissent
      quand les structures se forment -> poussee tardive
      supplementaire -> H_tardif > H_extrapole : LE BON SIGNE !) --
      mais l'ordre est fixe par l'energie de liaison relachee :
      |W|/rho c^2 ~ f_coll (v/c)^2 -> chiffre ci-dessous.
"""
import math

F_COLL = 0.5          # fraction de matiere dans les halos effondres (z=0)
V_TYP = 300e3         # m/s, dispersion typique ponderee masse (halos)
V_CLUS = 1000e3       # m/s, amas massifs (borne haute)
C = 3e8
DH_NEEDED = 0.083     # la tension


def main():
    print(__doc__)
    print("=" * 78)
    print("[B2] LA POUSSEE STRUCTURELLE TARDIVE, CHIFFREE")
    print("=" * 78)
    for lab, v in (("halos typiques (300 km/s)", V_TYP),
                   ("borne haute amas (1000 km/s)", V_CLUS)):
        w = F_COLL * (v / C) ** 2
        dh = w / 2
        print("  %-30s |W|/rho c^2 ~ %.1e -> Delta H/H ~ %.1e"
              % (lab, w, dh))
    w_hi = F_COLL * (V_CLUS / C) ** 2
    print("""  tension requise : Delta H/H = %.3f
  -> deficit : ~%.0f ordres de grandeur MEME a la borne haute.
  La poussee sourcee par la formation des structures a le BON SIGNE
  (H tardif > H extrapole) mais elle est ~4-5 ordres TROP PETITE :
  l'energie de liaison relachee par l'effondrement est une fraction
  ~1e-6 de la densite -- rien ne peut en faire 8 %%.""" % (
        DH_NEEDED, math.log10(DH_NEEDED / (w_hi / 2))))

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("""  L'ELASTICITE D'ECHELLE EST UN ACQUIS DE MECANISATION, PAS UN
  LEVIER DE TENSION :
   - elle mecanise TROIS faits etablis (systemes lies fixes ; zero
     derive locale ; expansion visible seulement au lointain) et
     complete T1 en triptyque coherent : poussee globale (Friedmann),
     elasticite locale (systemes lies), relativite d'echelle
     (invisibilite locale) ;
   - pour la tension H0 : la version transition-d'echelle est sans
     effet (SH0ES est en plein regime cumulatif) ; la version
     poussee-structurelle a le bon signe mais est 4-5 ordres trop
     petite. La position du registre (H0 = 67,4, tension locale =
     systematique ou vide reel en RG) reste inchangee -- troisieme
     confirmation par une troisieme voie.
  A CONSIGNER : l'elasticite complete le volet (v) de S4' (la
  poussee est GLOBALE parce que seul l'univers entier ne « se
  retire » jamais ; partout ailleurs la trame revient) -- c'est
  la meme structure que la portee strictement globale de T1, vue
  du cote local.""")


if __name__ == "__main__":
    main()
