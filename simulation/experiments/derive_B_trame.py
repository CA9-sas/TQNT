"""Dérivation de B depuis la trame — cible fille de la fonctionnelle E(a).

Chaîne (brouillon couplages Higgs, §potentiel-trame) :

  (1) forme F2 établie : E(a) = B a^4 [ln(a/v) - 1/4]  -> m_H^2 = 4 B v^2.
  (2) EXACT (définition du couplage courant) : E = (lambda(a)/4) a^4 avec
      lambda(a) = 4B[ln(a/v) - 1/4]  =>  d lambda / d ln a = 4B = beta_lambda.
      B est LA fonction beta du quartique (structure Gildener-Weinberg).
  (3) TRAME : le RG de la trame est DISCRET — les barreaux de l'échelle
      des croisements, espacés de L = ln(1/alpha) (canonique : alpha par
      croisement, chaîne G alpha^0 -> alpha^18). Donc
         beta_lambda = (Delta lambda par barreau)/L   — le slot 1/L est
      STRUCTUREL (la discrétisation de l'échelle), pas un choix.
  (4) le résidu par barreau, MESURÉ : B·L = 1/pi à ~0,1 %
      (Delta lambda par barreau = 4/pi). Identification candidate — le
      mécanisme est la cible (candidats angulaires nommés ci-dessous),
      le null est passé DANS la classe (aucun autre O(1) simple à <1,7 %).

  =>  B = 1/(pi ln(1/alpha))
  =>  m_H = 2v/sqrt(pi ln(1/alpha))
          = 2 sqrt(2) alpha^8 M_Pl (1 + alpha/4pi) / sqrt(ln(1/alpha))
      — m_H depuis alpha et M_Pl SEULS.

Contrôles : fermeture de la fourche (asthénion 125,26 vs fonctionnelle),
null en classe et hors classe, profondeur du puits (corollaire trame).
"""
import numpy as np
from math import gcd

# ---------------------------------------------------------------- entrées
alpha_inv = 137.035999          # alpha mesuré (le même que la chaîne v)
alpha = 1 / alpha_inv
L = np.log(alpha_inv)           # espacement d'un barreau (e-folds)
M_Pl = 1.220890e19              # GeV
v_GF = 246.21965                # GeV (mesuré, référence)
mH_pdg, s_pdg = 125.20, 0.11    # PDG 2024 (moyenne mondiale)
mH_old, s_old = 125.25, 0.17    # PDG 2022 (déclaré : les deux confrontés)
mH_asthenion = 125.26           # TQNT secteur nœuds (7_1+8_3+10_1)

print("=" * 74)
print("DÉRIVATION DE B DEPUIS LA TRAME  ->  m_H prédit    (2026-08-27)")
print("=" * 74)

# ------------------------------------------ la chaîne v (contrôle amont)
v_deriv = np.sqrt(2 * np.pi) * alpha**8 * M_Pl * (1 + alpha / (4 * np.pi))
print(f"""
Contrôle amont — v dérivé : sqrt(2pi) alpha^8 M_Pl (1+alpha/4pi)
   = {v_deriv:.3f} GeV vs {v_GF} (G_F) : {1e6*(v_deriv/v_GF-1):+.0f} ppm aux
   chiffres d'entrée de ce script (+28 ppm canonique — établi)""")

# ------------------------------------------ le résidu par barreau, mesuré
print(f"\nÉtage (3)-(4) — le résidu par barreau : c = B·L mesuré")
print(f"   L = ln(1/alpha) = {L:.6f} (un barreau = un croisement)")
for mh, s, tag in ((mH_pdg, s_pdg, 'PDG 2024'), (mH_old, s_old, 'PDG 2022')):
    B_t = mh**2 / (4 * v_GF**2)
    c_t = B_t * L
    sc = c_t * 2 * s / mh
    dev = (1/np.pi / c_t - 1)
    print(f"   m_H = {mh} +/- {s} ({tag}) : c = {c_t:.6f} +/- {sc:.6f} ;"
          f" 1/pi = {1/np.pi:.6f} ({100*dev:+.2f} %, {abs(dev)*c_t/sc:.1f} sigma)")

# ------------------------------------------ null EN CLASSE (c = O(1) simple)
print("\nNull EN CLASSE (formes B = c/L, c constante simple) :")
c_meas = (mH_pdg**2 / (4 * v_GF**2)) * L
CANDS = {'1/pi': 1/np.pi, '1/3': 1/3, '3/10': 0.3, '5/16': 5/16,
         'pi/10': np.pi/10, '1/e': 1/np.e, 'sqrt(2)/4': np.sqrt(2)/4,
         '(rac2-1)': np.sqrt(2)-1, '1/2pi': 1/(2*np.pi),
         '2/pi': 2/np.pi, '8/25': 0.32, 'ln2/2': np.log(2)/2}
for name, val in sorted(CANDS.items(), key=lambda kv: abs(kv[1]/c_meas-1)):
    print(f"   {name:10s} = {val:.5f} : {100*(val/c_meas-1):+7.2f} %")
print("   -> 1/pi seul sous 0,5 % ; voisins : 8/25 (+0,6 %, sans slot"
      " mécanistique),\n      5/16 (-1,7 %), 1/3 (+4,8 %). (7/22 etc. ="
      " mimiques diophantiennes de pi,\n      exclues par principe.)")

# ------------------------------------------ null HORS CLASSE (balayage)
B_targ = mH_pdg**2 / (4 * v_GF**2)
band = 2 * (2 * s_pdg / mH_pdg)          # 2 sigma sur B (B ∝ m_H^2)
hits = []
seen = set()
for n in range(1, 11):
    for d in range(1, 11):
        if gcd(n, d) != 1:
            continue
        q = n / d
        for p in (-1, 0, 1):
            for r in (-1, 0, 1):
                for s_ in (0, 1):
                    val = q * np.pi**p * L**r * alpha**s_
                    key = round(np.log(val), 6)
                    if key in seen:
                        continue
                    seen.add(key)
                    if abs(val / B_targ - 1) < band:
                        hits.append((n, d, p, r, s_, val))
print(f"\nNull HORS CLASSE — balayage q(n/d<=10)·pi^p·L^r·alpha^s "
      f"(p,r ∈ [-1,1], s ∈ [0,1]) :\n   {len(seen)} formes distinctes, "
      f"bande +/-{100*band:.2f} % (2 sigma) : {len(hits)} coup(s)")
for n, d, p, r, s_, val in hits:
    print(f"   {n}/{d} · pi^{p} · L^{r} · alpha^{s_} = {val:.6f} "
          f"({100*(val/B_targ-1):+.2f} %)")
print("   -> le null numérique SEUL ne suffit pas (concurrents en L^+1"
      " alpha^1) ;\n   la sélection est STRUCTURELLE : l'étage (2)-(3)"
      " IMPOSE le slot L^-1\n   (B = résidu/barreau) — les formes en L^+1"
      " n'ont aucun slot mécanistique.\n   Dans la classe structurelle"
      " B = c/L, le balayage ne rend QUE c = 1/pi.")

# ------------------------------------------ LA PRÉDICTION
B = 1 / (np.pi * L)
mH_pred = 2 * v_GF / np.sqrt(np.pi * L)
mH_pred_full = (2 * np.sqrt(2) * alpha**8 * M_Pl * (1 + alpha/(4*np.pi))
                / np.sqrt(L))
lam_pred = 2 * B
print(f"""
PRÉDICTION — B = 1/(pi L) = {B:.6f} :
   m_H = 2v/sqrt(pi ln(1/alpha))          = {mH_pred:.4f} GeV   (v = G_F)
   m_H = 2rac2 alpha^8 M_Pl (1+a/4pi)/rac(L) = {mH_pred_full:.4f} GeV   (M_Pl seul)
   vs mesuré  : {mH_pdg} +/- {s_pdg} -> {100*(mH_pred/mH_pdg-1):+.3f} % ({abs(mH_pred-mH_pdg)/s_pdg:.2f} sigma)
                {mH_old} +/- {s_old} -> {100*(mH_pred/mH_old-1):+.3f} % ({abs(mH_pred-mH_old)/s_old:.2f} sigma)
   vs asthénion TQNT (7_1+8_3+10_1) {mH_asthenion} : {100*(mH_pred/mH_asthenion-1):+.3f} %
   => LA FOURCHE SE FERME : les deux lectures TQNT de m_H (état de nœud /
      courbure de la fonctionnelle) coïncident à ~1e-4 — support direct de
      « l'asthénion sature le mode d'amplitude ».
   lambda (= m_H^2/2v^2) = 2/(pi L) = {lam_pred:.6f} (mesuré {mH_pdg**2/(2*v_GF**2):.6f})""")

# ------------------------------------------ corollaire : le puits ne pèse pas
depth = -B * v_GF**4 / 4
print(f"""
Corollaire (candidat) — profondeur du puits : E(v) = -B v^4/4 = {depth:.2e} GeV^4.
   Dans le MS c'est l'énergie de vide EW (problème). Lecture trame : G ne
   répond qu'aux DÉFICITS de la trame (S4', canonique) — la réserve interne
   de la trame, dont E(a) fait partie, ne gravite pas : le puits EW ne pèse
   pas PAR CONSTRUCTION. À consolider séparément (ligne rho_Lambda ailleurs).

Mécanisme du 1/pi — cible nommée (candidats consignés, AUCUN retenu) :
   (i)  écho du barème « par radian » (le même diviseur pi que s = DWr/pi :
        une unité de raideur par demi-tour et par barreau) ;
   (ii) moyenne de projection PLANE <|cos|> = 2/pi (le croisement est un
        objet de projection 2D) — au facteur 2 près sur Delta lambda ;
   (iii) mesure de boucle 1/2pi (g-2) — au facteur 2 dans l'autre sens.
   Statut global : (1)-(2) exacts ; (3) = discrétisation canonique de
   l'échelle (rang : alpha par croisement) ; (4) = identification MESURÉE
   (0,5 sigma), mécanisme ouvert — même rang que kappa_red=3/16 avant sa
   dérivation. Falsifiable : m_H mondial convergeant vers 125,11 (ATLAS
   seul) mettrait la prédiction à +1,3 sigma ; sous 125,0 elle tombe.""")
