"""Test d'alignement directionnel — tâche 1 du brouillon sablier.

Question : les directions « anormales » publiées (anomalies CMB, dipôle
quasars, dipôle alpha de Webb, dipôle de chiralité des galaxies)
partagent-elles un axe commun, comme le prédirait l'axe du sablier —
ou leur concentration est-elle compatible avec l'isotropie ?

Méthode : les directions sont traitées comme des AXES (vecteurs sans
tête, ±n équivalents — conservatif, car les signes des différents
traceurs ne sont pas comparables). Statistique de concentration :
lambda_1, la plus grande valeur propre de la matrice d'orientation
T = (1/N) somme n n^T (test de Watson/Bingham). Hypothèse nulle :
N axes isotropes indépendants ; p-value par Monte-Carlo.

Variantes déclarées (l'indépendance des traceurs est LE point délicat) :
  V1  tous les axes (naïf — surestime : dipôles cinématiques inclus) ;
  V2  sans les dipôles à direction attendue (CMB cinématique, CatWISE :
      leur direction n'est PAS une anomalie, seule l'amplitude l'est) ;
  V3  anomalies internes au CMB seulement (même carte : non indépendantes,
      mais statistiques différentes) ;
  V4  UN axe par physique indépendante : bas-l CMB (moyenne quad/oct),
      asymétrie de puissance, point froid, alpha-Webb, spins-Shamir.
La variante qui répond à la question du sablier est V4.

Réserves : sélection a posteriori (ces directions sont publiées PARCE
QUE anormales — le p-value est indicatif, pas confirmatoire) ; erreurs
angulaires publiées ~5-20 deg non propagées (jackknife à la place) ;
Webb et Shamir sont contestés (variantes avec/sans).
"""
import numpy as np

rng = np.random.default_rng(137)

# ---------------------------------------------------------------- outillage
def gal_to_vec(l_deg, b_deg):
    l, b = np.radians(l_deg), np.radians(b_deg)
    return np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])

# conversion équatoriale J2000 -> galactique (matrice standard IAU)
_R = np.array([
    [-0.0548755604, -0.8734370902, -0.4838350155],
    [+0.4941094279, -0.4448296300, +0.7469822445],
    [-0.8676661490, -0.1980763734, +0.4559837762],
])

def radec_to_gal(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    v_eq = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    v = _R @ v_eq
    b = np.degrees(np.arcsin(v[2]))
    l = np.degrees(np.arctan2(v[1], v[0])) % 360.0
    return l, b, v

def lambda1(vecs):
    """Plus grande valeur propre de la matrice d'orientation (axes)."""
    V = np.array(vecs)
    T = V.T @ V / len(V)
    w, e = np.linalg.eigh(T)
    return w[-1], e[:, -1]

def mc_pvalue(n_axes, lam_obs, n_mc=200_000):
    """P(lambda_1 >= lam_obs) sous N axes isotropes."""
    count = 0
    batch = 20_000
    done = 0
    while done < n_mc:
        m = min(batch, n_mc - done)
        X = rng.normal(size=(m, n_axes, 3))
        X /= np.linalg.norm(X, axis=2, keepdims=True)
        T = np.einsum('mni,mnj->mij', X, X) / n_axes
        w = np.linalg.eigvalsh(T)
        count += int(np.sum(w[:, -1] >= lam_obs))
        done += m
    return count / n_mc

def vec_to_gal(v):
    v = v / np.linalg.norm(v)
    return (np.degrees(np.arctan2(v[1], v[0])) % 360.0,
            np.degrees(np.arcsin(np.clip(v[2], -1, 1))))

# ---------------------------------------------------------------- directions
# Coordonnées galactiques (l, b) en degrés — sources vérifiées 2026-08-27.
shamir_l, shamir_b, _ = radec_to_gal(47.0, -22.0)   # (alpha, delta) = (47, -22)

DIRS = {
    # clef : (l, b, étiquette, classe)
    'quad':    (228.0, 60.0,  'Quadrupôle CMB (normale du plan, Planck)', 'cmb'),
    'oct':     (246.0, 66.0,  'Octopôle CMB (normale du plan, Planck)',   'cmb'),
    'coldspot':(209.0, -57.0, 'Point froid CMB',                          'cmb'),
    'pwasym':  (227.0, -27.0, 'Asymétrie de puissance CMB (dipôle de modulation)', 'cmb'),
    'cmbdip':  (264.0, 48.25, 'Dipôle cinématique CMB (référence, non anormal)',  'kin'),
    'catwise': (237.2, 41.8,  'Dipôle quasars CatWISE (direction ~cinématique)',  'kin'),
    'webb':    (320.0, -11.0, 'Dipôle alpha de Webb (contesté)',          'indep'),
    'shamir':  (round(shamir_l, 1), round(shamir_b, 1),
                'Dipôle de chiralité des spins (Shamir, contesté)',       'indep'),
}

print("=" * 76)
print("TEST D'ALIGNEMENT DIRECTIONNEL — axe du sablier (tâche 1 du brouillon)")
print("=" * 76)
print(f"\nDirections utilisées (galactiques) ; Shamir converti de "
      f"(RA,dec)=(47,-22) -> (l,b)=({shamir_l:.1f},{shamir_b:.1f})\n")
for k, (l, b, lab, cl) in DIRS.items():
    print(f"  {k:9s} (l,b)=({l:6.1f},{b:6.1f})  [{cl:5s}]  {lab}")

vecs = {k: gal_to_vec(l, b) for k, (l, b, _, _) in DIRS.items()}

# axe bas-l CMB = moyenne (axiale) quadrupôle/octopôle
lowl = vecs['quad'] + vecs['oct'] * np.sign(vecs['quad'] @ vecs['oct'])
lowl /= np.linalg.norm(lowl)

VARIANTES = {
    'V1 tous (naïf, 8 axes)': list(vecs),
    'V2 sans dipôles cinématiques (6 axes)':
        ['quad', 'oct', 'coldspot', 'pwasym', 'webb', 'shamir'],
    'V3 CMB interne (4 axes, même carte)':
        ['quad', 'oct', 'coldspot', 'pwasym'],
    'V4 un axe par physique (5 axes) — LA variante du sablier':
        ['LOWL', 'pwasym', 'coldspot', 'webb', 'shamir'],
    'V4a V4 sans Webb (4 axes)': ['LOWL', 'pwasym', 'coldspot', 'shamir'],
    'V4b V4 sans Shamir (4 axes)': ['LOWL', 'pwasym', 'coldspot', 'webb'],
}

def get(k):
    return lowl if k == 'LOWL' else vecs[k]

print("\n--- Séparations angulaires axiales (deg ; 0 = alignés, 90 = orthogonaux) ---")
keys = list(vecs) + ['LOWL']
for i, a in enumerate(keys):
    for b_ in keys[i + 1:]:
        ang = np.degrees(np.arccos(np.clip(abs(get(a) @ get(b_)), 0, 1)))
        if ang < 30:
            print(f"  {a:9s} - {b_:9s} : {ang:5.1f}  <-- proche")

print("\n--- Concentration axiale (lambda_1) et p-value Monte-Carlo (200 000 tirages) ---")
print(f"{'variante':52s} {'lam1':>6s} {'attendu iso':>11s} {'p-value':>9s}")
results = {}
for name, ks in VARIANTES.items():
    V = [get(k) for k in ks]
    lam, axis = lambda1(V)
    # espérance isotrope approximative de lambda_1 pour N axes
    Xs = rng.normal(size=(4000, len(V), 3))
    Xs /= np.linalg.norm(Xs, axis=2, keepdims=True)
    Ts = np.einsum('mni,mnj->mij', Xs, Xs) / len(V)
    lam_iso = np.mean(np.linalg.eigvalsh(Ts)[:, -1])
    p = mc_pvalue(len(V), lam)
    l_ax, b_ax = vec_to_gal(axis)
    results[name] = (lam, p, (l_ax, b_ax))
    print(f"{name:52s} {lam:6.3f} {lam_iso:11.3f} {p:9.4f}"
          f"   axe commun (l,b)=({l_ax:.0f},{b_ax:.0f})")

print("\n--- Jackknife sur V4 (retirer un axe à la fois) ---")
ks4 = VARIANTES['V4 un axe par physique (5 axes) — LA variante du sablier']
for drop in ks4:
    sub = [k for k in ks4 if k != drop]
    lam, _ = lambda1([get(k) for k in sub])
    p = mc_pvalue(len(sub), lam, n_mc=50_000)
    print(f"  sans {drop:9s} : lam1 = {lam:.3f}, p = {p:.4f}")

print("\n--- Angles au meilleur axe commun de V4 ---")
_, axis4 = lambda1([get(k) for k in ks4])
for k in ks4:
    ang = np.degrees(np.arccos(np.clip(abs(get(k) @ axis4), 0, 1)))
    print(f"  {k:9s} : {ang:5.1f} deg")

print("""
Rappels d'interprétation :
 - sélection a posteriori : ces directions sont publiées PARCE QUE
   anormales ; le p-value mesure la concentration DES directions
   publiées, pas la probabilité globale de l'hypothèse ;
 - V1/V3 sur-comptent (dipôles cinématiques / même carte CMB) ;
 - la prédiction canonique de la trame (isotropie structurelle) et la
   prédiction du sablier (axe réel) seront départagées proprement par
   la RÉPLICATION dans des sondes indépendantes (modes E LiteBIRD,
   lentillage, 21 cm), pas par ce test rétrospectif.""")
