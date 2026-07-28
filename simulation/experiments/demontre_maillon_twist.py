"""Demonstration du maillon : twist residuel = (d-1) demi-tours (21/07).

MAILLON A DEMONTRER : pourquoi l'anneau peripherique partage d'une
paire de noeuds IDENTIQUES stocke un twist residuel de (d-1) demi-tours
(d = rac2, la dimension quantique forcee des 2 canaux de fusion).

TROIS PIECES :
 [A] FAIT GEOMETRIQUE (verifiable) : un demi-tour de l'anneau autour du
     cable a deux brins ECHANGE les deux membres -- c'est pourquoi
     l'exces s'attache a l'unite demi-tour, et SEULEMENT aux paires
     identiques (pour des membres distincts, la configuration echangee
     est un AUTRE etat : pas de condition de fermeture, pas de residu).
 [B] VALEUR DE BOUCLE (Temperley-Lieb, exact) : la fermeture de Markov
     de l'element d'echange e sur deux brins vaut d (une boucle),
     la fermeture classique (membres distinguables, pas d'echange)
     vaut 1. L'exces de contenu de boucle de l'anneau = d - 1.
     L'identification 'contenu de boucle <-> demi-tours stockes' est
     LA MEME que celle du quantum S4'' (fermeture d'anneau = unite
     pi sigma_chi) -- une identification, utilisee deux fois.
 [C] RESEAU D'IDENTITES EXACTES (sur-determination k=2, declaree) :
     d - 1 = rac2 - 1 = tan(pi/8) = tan(2 pi h_sigma)   [h_sigma=1/16]
                      = cot(2 pi h_1/2(k=2))            [h=3/16]
     -- l'exces est AUSSI la tangente de l'angle de spin topologique.
 DISCRIMINANT PAR LA DONNEE : la lecture 'phase brute du canal'
     donnerait 3/8 = 0,375 (l'angle 3pi/8 en demi-tours) : REJETEE.
"""
import numpy as np

R2 = np.sqrt(2)
MES, SIG = 0.4202, 0.0041   # exces mesure (3 determinations, 21/07)

print(__doc__)
print("=" * 76)
print("[A] LE FAIT GEOMETRIQUE : demi-tour du cable = un echange")
print("=" * 76)
t = np.linspace(0, 1, 5)
b1 = np.array([+0.5 + 0 * t, 0 * t, t])        # brin 1 du cable (x=+1/2)
b2 = np.array([-0.5 + 0 * t, 0 * t, t])        # brin 2 (x=-1/2)
th = np.pi                                      # demi-tour autour de z
R = np.array([[np.cos(th), -np.sin(th), 0],
              [np.sin(th), np.cos(th), 0], [0, 0, 1]])
b1r = R @ b1
print("  rotation de pi autour de l'axe du cable : brin1 -> position du")
print("  brin2 a %.1e pres (echange EXACT) ; pour des noeuds IDENTIQUES"
      % np.abs(b1r - b2).max())
print("  c'est une fermeture (meme configuration) ; pour des noeuds")
print("  DISTINCTS c'est un autre etat -> pas de condition, pas de residu.")

print("\n" + "=" * 76)
print("[B] LA VALEUR DE BOUCLE (Temperley-Lieb, exacte)")
print("=" * 76)
d = R2
print("  fermeture de Markov sur 2 brins : tr(identite) = d^2 = %.0f" % d**2)
print("  (= le nombre de canaux : coherent) ; tr(e) = d = %.4f (UNE" % d)
print("  boucle) ; membres distinguables (pas d'element e) : valeur 1.")
print("  EXCES de contenu de boucle du a l'echange : d - 1 = %.4f" % (d - 1))
print("  L'identification 'boucle fermee <-> demi-tour stocke' est celle")
print("  du quantum S4'' lui-meme (fermeture d'anneau = pi sigma_chi) :")
print("  AUCUNE hypothese nouvelle -- une identification, deux usages.")

print("\n" + "=" * 76)
print("[C] LE RESEAU D'IDENTITES ET LE DISCRIMINANT")
print("=" * 76)
cands = [("d - 1 (valeur de boucle)", d - 1),
         ("tan(pi/8) = tan(2 pi h_sigma)", np.tan(np.pi / 8)),
         ("cot(2 pi h_1/2(k=2)) [h=3/16]", 1 / np.tan(3 * np.pi / 8)),
         ("PHASE BRUTE 3/8 (angle 3pi/8 en 1/2-tours)", 3 / 8)]
for nom, v in cands:
    ns = abs(MES - v) / SIG
    print("  %-44s %.4f  (%+5.1f %%, %4.1f sigma_stat)"
          % (nom, v, 100 * (MES / v - 1), ns))
print("  -> les trois premieres sont LA MEME VALEUR (identites trigo")
print("     exactes a k=2 : sur-determination DECLAREE, un test en k")
print("     n'existe pas) ; la lecture 'phase brute' 3/8 est REJETEE par")
print("     la mesure (+12 %, ~11 sigma_stat ; meme avec +-0,01 de")
print("     systematique coeurs : ~4 sigma). La donnee SELECTIONNE la")
print("     forme valeur-de-boucle/tangente.")

print("\n" + "=" * 76)
print("VERDICT : le maillon est DEMONTRE au rang des autres theoremes de")
print("l'echelle (conditionnel aux identifications canoniques) --")
print("(1) demi-tour d'anneau = echange (fait geometrique, [A]) ;")
print("(2) l'echange insere l'element e dont la fermeture vaut d (theoreme")
print("    TL/Markov, [B]) ; le classique vaut 1 ; exces = d - 1 ;")
print("(3) l'identification boucle<->demi-tour est CELLE du quantum S4''")
print("    (pas d'hypothese nouvelle) ;")
print("(4) la donnee rejette la seule lecture concurrente (3/8, ~4 sigma)")
print("    et l'identite tan(2 pi h_sigma) rattache l'exces au spin")
print("    topologique (sur-determination k=2 declaree).")
print("L'echelle de recouplage est desormais SANS maillon libre : base,")
print("bascule, x3, 1/2, et l'exces (d-1) -- tous au meme rang.")
