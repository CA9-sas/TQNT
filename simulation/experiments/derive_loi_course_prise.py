"""Loi de course de la prise — tâche 3 du brouillon couplages Higgs.

Question : quelle « prise sur la trame » le vertex P15 du H lit-il pour
une espèce CONFINÉE (stoikhion) ? Les données (kappa_b ~ 1 avec la masse
COURANTE m_b(m_H) = 2,80 GeV) montrent que ce n'est PAS la masse de nœud
du catalogue (m_b = 4952 MeV).

Résultat en trois étages :

  E1  THÉORÈME DE REGISTRE (dérivé conditionnel) : le vertex crée
      l'IDENTITÉ de l'objet — ce que l'objet EST à l'état libre.
        - nœud LIBRE (leptons) / entrelacs LIBRE (asthénions W, Z, top) :
          identité = le nœud entier -> prise = masse pleine (pôle) ;
        - stoikhion CONFINÉ (b, c, s...) : n'existe jamais libre ;
          l'identité créée au vertex = le MODE NU ; le tube/habillage est
          une propriété du condensat (assemblé à l'hadronisation, sans
          coût de prise). Support canonique : demi-substitution (le tube
          des mésons est PARTAGÉ -> pas additif par stoikhion -> pas dans
          l'identité) ; architecture nu/habillé déjà établie (k=135,
          m_s nu 468 vs habillé 494,8).

  E2  PINCEMENT PAR LES DONNÉES (les deux lectures fausses sont exclues) :
        - « les quarks lisent le catalogue » : kappa_b = 4952/2800 = 1,77
          -> exclu à ~8 sigma (mesuré 0,89 +/- 0,11) ;
        - « tout le monde est cisaillé à la résolution du vertex »
          (prise = 2 sigma_chi hbar c / m_H) : kappa_tau = 0,0016
          -> exclu à ~13 sigma (mesuré 0,93 +/- 0,07).
      La règle de registre est FORCÉE des deux côtés.

  E3  L'ÉTAGE DE COURSE (forme dérivée, coefficient emprunté) : la prise
      nue dépend de l'échelle. Forme FORCÉE par le même théorème qu'au
      vertex : il n'existe aucune autre échelle d'espèce que m lui-même
      -> le délestage par e-fold de résolution est proportionnel à m,
      au taux du couplage de secteur : dm/dln(mu) = -c (alpha_s/pi) m
      -> course en PUISSANCE de alpha_s (la forme RG exacte). Le
      coefficient c = 2 (gamma_0 = 6 C_F = 8) = comptage non abélien,
      EMPRUNTÉ (l'ouvert déclaré du cadre). Vérification LO autonome
      ci-dessous ; valeurs NLO standard pour la confrontation.
"""
import numpy as np

# ---------------------------------------------------------------- entrées
hbarc = 0.19733          # GeV.fm
sigma_chi = 0.41847**2   # GeV^2 ; en GeV/fm : sigma_chi/hbarc
m_H = 125.25             # GeV

# registres (GeV) : catalogue (nœud/constituant) vs courant
REG = {
    #        catalogue  m(m_q) MSbar   m(m_H) standard (NLO)
    'b':    (4.952,     4.18,          2.79),
    'c':    (1.557,     1.27,          0.62),
    's':    (0.4948,    0.093,         0.052),   # m_s(2 GeV)=93 MeV
}
m_tau = 1.77686
KAPPA_MEAS = {'b': (0.89, 0.11), 'tau': (0.93, 0.07), 't': (0.94, 0.11)}

print("=" * 74)
print("LOI DE COURSE DE LA PRISE — dérivation en 3 étages (E1 registre,")
print("E2 pincement, E3 course) — script compagnon du brouillon Higgs")
print("=" * 74)

# ------------------------------------------ E2a : lecture catalogue exclue
k_b, s_b = KAPPA_MEAS['b']
kappa_cat = REG['b'][0] / REG['b'][2]
print(f"""
E2a — si le vertex lisait le CATALOGUE pour le b :
   kappa_b = m_b(cat)/m_b(m_H) = {REG['b'][0]}/{REG['b'][2]} = {kappa_cat:.2f}
   mesuré : {k_b} +/- {s_b}  ->  exclu à {(kappa_cat-k_b)/s_b:.1f} sigma""")
kappa_cat_c = REG['c'][0] / REG['c'][2]
print(f"   (même lecture pour le c : kappa_c = {kappa_cat_c:.1f} — au bord des"
      f" bornes combinées actuelles kappa_c <~ 3 ; le b tranche seul)")

# ------------------------------------------ E2b : cisaillement géométrique exclu
k_t_, s_t_ = KAPPA_MEAS['tau']
prise_cisaillee = 2 * (sigma_chi / hbarc) * (hbarc / m_H)   # GeV
kappa_tau_cis = prise_cisaillee / m_tau
print(f"""
E2b — si TOUTE espèce était cisaillée à la résolution du vertex
      (prise = 2 sigma_chi hbar c / m_H = {1e3*prise_cisaillee:.2f} MeV) :
   kappa_tau = {kappa_tau_cis:.4f}
   mesuré : {k_t_} +/- {s_t_}  ->  exclu à {(k_t_-kappa_tau_cis)/s_t_:.1f} sigma
   => le tau (nœud LIBRE) est lu à sa masse PLEINE : la course n'est pas
      une résolution géométrique, c'est une règle de REGISTRE.
   [coïncidence consignée SANS mécanisme, non retenue (règle du null) :
    2 sigma_chi hbar c / m_H = {1e3*prise_cisaillee:.2f} MeV ~ m_d(m_H) ~ 2,6 MeV]""")

# ------------------------------------------ E3 : la course LO autonome
# alpha_s LO, seuils m_b ; alpha_s(MZ) = 0,1179
MZ, a_MZ, mb = 91.19, 0.1179, 4.18
b0 = lambda nf: 11 - 2 * nf / 3

def alpha_lo(mu):
    inv = 1 / a_MZ + b0(5) / (2 * np.pi) * np.log(mu / MZ)
    if mu >= mb:
        return 1 / inv
    inv_mb = 1 / a_MZ + b0(5) / (2 * np.pi) * np.log(mb / MZ)
    inv = inv_mb + b0(4) / (2 * np.pi) * np.log(mu / mb)
    return 1 / inv

def run_mass_lo(m0, mu0, mu1):
    """m(mu1) depuis m(mu0), LO, gamma_0 = 8, seuil m_b géré."""
    def leg(m, a, b, nf):
        expo = 8 / (2 * b0(nf))
        return m * (alpha_lo(b) / alpha_lo(a)) ** expo
    if mu0 < mb <= mu1:
        return leg(leg(m0, mu0, mb, 4), mb, mu1, 5)
    nf = 5 if mu0 >= mb else 4
    return leg(m0, mu0, mu1, nf)

print(f"""
E3 — la course (forme dérivée : dm/dlnmu = -c (alpha_s/pi) m ;
     c = gamma_0/4 = 2 EMPRUNTÉ au comptage non abélien) :
   alpha_s LO : {alpha_lo(mb):.4f} (m_b), {alpha_lo(m_H):.4f} (m_H)""")
mb_mH_lo = run_mass_lo(REG['b'][1], mb, m_H)
mc_mH_lo = run_mass_lo(REG['c'][1], 1.27, m_H)
print(f"   m_b(m_H) : LO autonome {mb_mH_lo:.2f} GeV vs NLO standard "
      f"{REG['b'][2]} ({100*(mb_mH_lo/REG['b'][2]-1):+.0f} %)")
print(f"   m_c(m_H) : LO autonome {mc_mH_lo:.2f} GeV vs NLO standard "
      f"{REG['c'][2]} ({100*(mc_mH_lo/REG['c'][2]-1):+.0f} %)")

# ------------------------------------------ synthèse : les kappa prédits
print(f"""
SYNTHÈSE — kappa prédits sous la loi de course (registre + secteur NLO) :
   kappa_b   = 1,00   mesuré {k_b} +/- {s_b}   ({abs(1-k_b)/s_b:.1f} sigma)
   kappa_tau = 1,00   mesuré {k_t_} +/- {s_t_}   ({abs(1-k_t_)/s_t_:.1f} sigma)
   kappa_t   = 1,00 (asthénion : JAMAIS confiné, se désintègre en ~5e-25 s
                     avant toute formation de tube -> masse pleine, pas de
                     course hadronique)   mesuré {KAPPA_MEAS['t'][0]} +/- {KAPPA_MEAS['t'][1]}
   PRÉDICTION HL-LHC : kappa_c = 1,00 AVEC m_c(m_H) = 0,62 GeV (toute
   extraction convergeant vers kappa_c ~ 2,5 signerait la lecture
   catalogue — réfutée ici par le b : la cohérence b/c est un test).

L'écart catalogue - courant à l'échelle propre (l'habillage) :""")
for q, (cat, own, at_mH) in REG.items():
    print(f"   {q} : {cat:.3f} -> {own:.3f} GeV  (habillage {1e3*(cat-own):.0f} MeV)")
print("""   -> PAS de forme close commune (287/772/402 MeV) : l'habillage par
   saveur reste MESURÉ, aucune formule retenue (règle du null).

Statuts : E1 registre = dérivé CONDITIONNEL (identification : identité =
état libre ; support demi-substitution + architecture nu/habillé) ;
E2 = pincement par les données (8 et 13 sigma) ; E3 forme = dérivée (même
théorème « aucune autre échelle d'espèce » qu'au vertex), coefficient
gamma_0 = 8 = EMPRUNTÉ (cible nommée : le comptage non abélien du secteur).""")
