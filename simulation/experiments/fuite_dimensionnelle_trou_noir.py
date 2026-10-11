"""Fuite dimensionnelle : le trou noir écrase une dimension ; les objets de dimension n−1 ressortent (16/09/2026).

Proposition (hors du cadre 3+1 actuel — postulat nouveau) : dans notre espace-temps le trou noir reste clos ;
dans une dimension supérieure il écrase une dimension ; la gravité n'agirait pas sur les objets de dimension
inférieure, qui fuient et ressortent dans notre espace. On chiffre ce qui la soutient, ce qui la contraint,
et ce qu'il faudrait pour alimenter un point rouge.
"""
import math
c=2.998e8; Msun=1.989e30; yr=3.156e7
print("=== 1. Ce qui soutient l'idée (physique connue) ===")
print("  (a) en 2+1 dimensions, la relativité générale n'a PAS d'attraction locale : une masse ponctuelle crée un cône")
print("      (déficit angulaire 8πGm), deux masses au repos ne s'attirent pas. « Écraser une dimension » supprime bien l'attraction.")
print("  (b) dans le cadre, la matière est faite de NŒUDS (courbes fermées dans 3 dimensions). Un nœud n'existe qu'en dimension 3 :")
print("      en 2D une courbe fermée ne peut pas se croiser (pas de croisement → pas de nœud) ; en 4D tout nœud se dénoue.")
print("      → changer la dimension DÉNOUE la matière : ce qui « fuit » ne peut être que des ondes χ non nouées —")
print("        lumière, neutrinos, ondes de la trame — et l'énergie de masse des nœuds (m = √σ_χ·𝒮) est libérée.")
print()
print("=== 2. Ce qui la contraint dans NOTRE espace (n = 3) ===")
print("  les objets « moins structurés » y subissent la gravité exactement comme la matière :")
print("   - GW170817 : ondes gravitationnelles et photons, même délai de Shapiro à |Δγ| ≤ 1e-6 près ;")
print("   - SN 1987A : neutrinos (ondes χ libres du cadre, sans nœud) et photons arrivés ensemble malgré ~mois de délai de Shapiro → égalité ~0,1 % ;")
print("   - lentilles, décalage vers le rouge : la lumière tombe.")
print("  → l'insensibilité à la gravité ne peut valoir QUE dans la région où la dimension est écrasée (l'intérieur) — cohérent avec la proposition.")
print()
print("=== 3. Le bilan d'énergie d'un point rouge alimenté par la fuite ===")
for L_erg in (1e43, 1e44, 1e45):
    L = L_erg*1e-7
    for eta in (1.0, 0.1):
        print(f"  L = {L_erg:.0e} erg/s, rendement de fuite η = {eta:<4} : matière avalée {L/(eta*c**2)*yr/Msun:.1e} Msun/an")
print("  → avec η ~ 1 (masse des nœuds entièrement dénouée), 10× moins de matière que l'accrétion ordinaire (η = 0,1).")
print()
print("=== 4. La contrainte qui tranche : les trous noirs proches ===")
print("  si TOUT trou noir laisse fuir une fraction η_fuite de l'énergie de la matière avalée sous forme VISIBLE :")
for Mdot in (2e-9, 1e-8, 2e-7):
    Mdc2 = Mdot*Msun/yr*c**2*1e7
    for Lobs in (1e35, 1e36):
        print(f"   Sgr A* : Ṁ = {Mdot:.0e} Msun/an → Ṁc² = {Mdc2:.1e} erg/s ; L_obs ≈ {Lobs:.0e} erg/s → η_fuite ≲ {Lobs/Mdc2:.0e}")
print("   binaires X à trou noir : pas de sursauts de surface (Narayan & Heyl 2002) ; EHT (M87*, Sgr A*) : ombre sans émission de surface.")
print("  → une fuite VISIBLE est bornée à ≲ 1e-5–1e-2 de Ṁc² dans les trous noirs proches.")
print("  → une fuite INVISIBLE (neutrinos, ondes χ de fond) échappe à la borne… mais ne peut pas éclairer un point rouge.")
print()
print("=== 5. Ce que la proposition doit fixer pour devenir testable ===")
print("  (i) la loi η_fuite(M, Ṁ, âge) : pourquoi ~1 dans un point rouge jeune et ≲ 1e-3 dans Sgr A* (régime super-Eddington ?) ;")
print("  (ii) en quoi ressort l'énergie dénouée (photons → visible et borné ; neutrinos → invisible) ;")
print("  (iii) où elle ressort : près de l'horizon (spectre chaud, X) ou plus loin (enveloppe froide) — les points rouges sont")
print("       FAIBLES en X, ce qu'une fuite déposée dans l'enveloppe plutôt qu'un disque chaud pourrait expliquer ;")
print("  (iv) le postulat dimensionnel lui-même : le cadre actuel est 3+1 (Chern–Simons en 3D, nœuds en 3D) ; ajouter une dimension")
print("       où le trou noir « écrase » la nôtre est un postulat nouveau, qui touche P1–P5.")
