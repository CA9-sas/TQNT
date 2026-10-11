"""Fonctionnelle de trame E(a) -> kappa_lambda — tâche 4 du brouillon Higgs.

Directive utilisateur (2026-08-27) : « c'est la trame de l'espace-temps qui
va définir le potentiel scalaire ». Programme : écrire l'énergie E(a) de la
trame pour un déplacement local d'amplitude a autour de l'équilibre v,
contrôler E''(v) <-> m_H, prédire kappa_lambda = E'''(v)/E'''_SM(v).

BANC DES FONCTIONNELLES (chacune = une lecture ontologique de la trame,
normalisée par le contrôle E''(v) = m_H^2 ; kappa_lambda est alors un
nombre PUR, indépendant de la normalisation) :

  F1  polynôme fondamental (le MS) : V = (m_H^2/8v^2)(a^2-v^2)^2
      -> kappa_lambda = 1. Exige un potentiel posé — la trame n'en a pas.
  F2  SANS ÉCHELLE PROPRE / TRANSMUTATION : E = B a^4 [ln(a/v) - 1/4].
      Lecture canonique : la trame n'a AUCUNE échelle scalaire propre
      (relativité d'échelle, volet (vi)) ; v est TRANSMUTÉ (l'échelle des
      croisements : v = sqrt(2pi) alpha^8 M_Pl — chaque croisement
      multiplie par alpha, une échelle multiplicative = logarithmique).
      La seule dépendance permise : a^4 x f(ln a). Forme minimale
      analytique à minimum -> kappa_lambda = 5/3 EXACT.
      [Antécédent déclaré : c'est la forme Coleman-Weinberg / Gildener-
      Weinberg de la brisure radiative — le 5/3 y est connu. Différence
      TQNT : on ne réclame PAS la normalisation par la somme de boucles
      (contrôle C2 : elle a le MAUVAIS SIGNE, cf. infra) — seule la FORME
      est sans échelle ; la normalisation vient de l'asthénion.]
  F3  graduation périodique : E = L^4 [1 - cos((a-v)/f)]
      -> kappa_lambda = 0 (cubique nul au minimum symétrique).
      REJETÉE par l'ontologie : la denture de la trame est graduée en
      ACTION (pas = h), pas en amplitude — aucune périodicité en a.
  F4  plancher de déplétion : E = A (a^2 - v^2)^2 / a^2 (diverge à a->0)
      -> kappa_lambda = -1.
      REJETÉE : le plancher d'amplitude de la trame est à ~M_Pl du point
      de travail (17 ordres) — aucun mécanisme ne le place à l'échelle EW.

CONTRÔLES :
  C1  E''(v) = m_H^2 : utilisé comme normalisation (cohérent : la lecture
      canonique donne m_H = masse de l'asthénion 7_1+8_3+10_1 = 125,26,
      +0,05 % du mesuré) — la version profonde (DÉRIVER B et donc m_H de
      la trame) est la cible nommée.
  C2  la normalisation « somme de boucles du quartet » (CW standard) :
      calculée ci-dessous — le top (signe fermionique = parité de tresse)
      la rend NÉGATIVE -> la normalisation n'est PAS une somme de boucles
      de matière ; c'est la raison historique de la mort du CW dans le MS,
      et la raison pour laquelle la TQNT n'en réclame que la FORME.
  C3  robustesse : E = B a^4 [ln - 1/4 + eps ln^2] (l'ordre suivant de
      l'échelle) -> d(kappa_lambda)/d(eps) mesuré numériquement ; avec
      eps ~ alpha (le pas de l'échelle), correction au pour-mille.

Toutes les dérivées analytiques sont VÉRIFIÉES par différences finies.
"""
import numpy as np

m_H, v = 125.25, 246.22          # GeV
E3_SM = 3 * m_H**2 / v           # E'''_SM(v) — référence du kappa

# ------------------------------------------------ outillage : dérivées FD
# pas ADAPTÉS à l'échelle v (un pas absolu ~1e-3 sur E ~ 1e8 GeV^4 est
# détruit par l'annulation catastrophique en double précision)
def d2(f, x, h=0.5):
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2

def d3(f, x, h=0.5):
    return (f(x + 2*h) - 2*f(x + h) + 2*f(x - h) - f(x - 2*h)) / (2 * h**3)

def d4(f, x, h=3.0):
    return (f(x + 2*h) - 4*f(x + h) + 6*f(x) - 4*f(x - h) + f(x - 2*h)) / h**4

# ------------------------------------------------ banc des fonctionnelles
lam = m_H**2 / (2 * v**2)
B_cw = m_H**2 / (4 * v**2)
A_dep = m_H**2 / 8
f_per = 30.0                      # échelle arbitraire, se simplifie
L4 = m_H**2 * f_per**2

BANC = {
    'F1 polynome (MS)':
        lambda a: (lam / 4) * (a**2 - v**2)**2,
    'F2 sans echelle propre (transmutation)':
        lambda a: B_cw * a**4 * (np.log(a / v) - 0.25),
    'F3 graduation periodique':
        lambda a: L4 * (1 - np.cos((a - v) / f_per)),
    'F4 plancher de depletion':
        lambda a: A_dep * (a**2 - v**2)**2 / a**2,
}

print("=" * 74)
print("FONCTIONNELLE DE TRAME E(a) -> kappa_lambda   (tâche 4, 2026-08-27)")
print("=" * 74)
print(f"\nContrôle C1 (normalisation) : E''(v) = m_H^2 avec m_H = {m_H} GeV")
print(f"Référence MS : E'''_SM(v) = 3 m_H^2/v = {E3_SM:.1f} GeV\n")
print(f"{'fonctionnelle':42s} {'E\"(v)/m_H^2':>12s} {'k_lambda':>9s} {'k_4':>7s}")
for name, E in BANC.items():
    c2 = d2(E, v) / m_H**2
    kl = d3(E, v) / E3_SM
    k4 = d4(E, v) / (3 * m_H**2 / v**2)
    print(f"{name:42s} {c2:12.4f} {kl:9.3f} {k4:7.2f}")

print(f"""
-> le banc DISCRIMINE : kappa_lambda ∈ {{1, 5/3, 0, -1}} selon la lecture.
   Sélection ontologique (cf. brouillon) : F2 — la trame est sans échelle
   scalaire propre, v est transmuté. PRÉDICTION :
      kappa_lambda = 5/3 = {5/3:.4f}   (E'''(v) = 5 m_H^2/v = {5*m_H**2/v:.1f} GeV)
      kappa_4      = 11/3 (quartique, hors de portée avant FCC)""")

# ------------------------------------------------ C2 : la somme de boucles
masses = {'W': (80.68, 6), 'Z': (90.56, 3), 'H': (125.26, 1),
          'top': (170.41, -12)}
S = sum(n * (m / v)**4 for m, n in masses.values())
print(f"""
Contrôle C2 — normalisation « somme de boucles du quartet » (CW standard) :
   Somme n_i (m_i/v)^4 = {S:+.3f}  (top : {-12*(170.41/v)**4:+.3f} domine)
   -> NÉGATIVE : une normalisation par boucles de matière est EXCLUE
   (c'est la mort historique du CW dans le MS). La TQNT n'en réclame que
   la FORME (sans échelle) ; la normalisation B = m_H^2/(4v^2) vient de
   l'asthénion (contrôle C1). DÉRIVER B de la trame = cible nommée.""")

# ------------------------------------------------ C3 : robustesse en eps
# E = B a^4 [L - 1/4 + eps L^2], L = ln(a/v0) : le minimum RESTE à v0
# (E' ∝ a^3 L (4 + 2 eps + 4 eps L) = 0 en L=0) ; développement exact :
#   E/Bv^4 = (2+eps) x^2 + (10/3 + 3 eps) x^3 + O(x^4),  x = a/v - 1
#   -> kappa_lambda(eps) = (20 + 18 eps)/(12 + 6 eps)  [= 5/3 + (2/3) eps]
print("\nContrôle C3 — ordre suivant de l'échelle : "
      "E = B a^4 [ln(a/v0) - 1/4 + eps ln^2]\n"
      "   analytique : kappa_lambda(eps) = (20+18 eps)/(12+6 eps) ; "
      "vérif. FD :")
for eps in (0.01, -0.01, 0.1, -0.1):
    E = lambda a: B_cw * a**4 * (np.log(a / v) - 0.25
                                 + eps * np.log(a / v)**2)
    kl_ana = (20 + 18 * eps) / (12 + 6 * eps)
    kl_fd = d3(E, v) / (3 * d2(E, v) / v)
    print(f"   eps = {eps:+.2f} : kappa_lambda = {kl_ana:.4f} "
          f"(FD {kl_fd:.4f}) ({100*(kl_ana/(5/3)-1):+.2f} %)")
print("   -> correction relative = 0,4 x eps : avec eps ~ alpha (le pas\n"
      "      de l'échelle des croisements), ~0,3 % — le 5/3 est ROBUSTE.")

# ------------------------------------------------ phénoménologie HH
# parabole grossière par 3 points publiés (ggF 13-14 TeV) :
# sigma/sigma_SM ~ 2,2 (k=0), 1 (k=1), ~0,35 au minimum k~2,4. DÉCLARÉ
# grossier (interférence triangle/boîte) — ordre de grandeur seulement.
import numpy.polynomial.polynomial as P
M = np.array([[1, 0, 0], [1, 1, 1], [1, 2.4, 2.4**2]])
c = np.linalg.solve(M, np.array([2.2, 1.0, 0.35]))
sig = lambda k: c[0] + c[1] * k + c[2] * k**2
print(f"""
Phénoménologie HH (parabole grossière déclarée, ggF) :
   sigma(HH; kappa_lambda=5/3)/sigma_SM ~ {sig(5/3):.2f}
   -> côté DESTRUCTIF de l'interférence : di-Higgs ~{100*(1-sig(5/3)):.0f} %
   SOUS le MS. Limites actuelles kappa_lambda ~ [-0,6, 6,6] : 5/3 est
   loin d'être testé ; HL-LHC (sensibilité ~SM) le frôlera, FCC tranchera.
   SIGNATURE JOINTE : kappa=1 partout (arbre) + kappa_lambda=5/3 + taux
   HH réduit ~moitié — motif que ni le MS (kappa_lambda=1) ni les 2HDM
   (kappas d'arbre déviés) ne reproduisent.""")

print("""Statuts : forme F2 = identification CONDITIONNELLE (« la
fonctionnelle de trame est sans échelle propre ») au rang canonique
(relativité d'échelle + v transmuté) ; 5/3 = EXACT sous la forme ;
normalisation B (donc m_H depuis la trame) = OUVERTE (cible) ; F3/F4
écartées par mécanisme (graduation en action, plancher à 17 ordres) ;
antécédent CW/Gildener-Weinberg déclaré.""")
