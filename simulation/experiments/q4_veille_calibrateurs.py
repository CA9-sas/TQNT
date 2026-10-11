# -*- coding: utf-8 -*-
"""Q-4 — Veille calibrateurs H₀ (état au 03/09/2026).

Prédiction à surveiller (H1, falsifiable daté) : à mesure que les
calibrateurs convergent, H₀ local doit descendre vers 67,4±1 (au plus ~69
avec le vide KBC) ; un >72 consolidé avec vide exclu réfuterait.

Sources (vérifiées 03/09/2026) :
- CCHP (Freedman+, ApJ 985, 203, 2025) : JWST seul — TRGB 68,81±1,79±1,32 ;
  JAGB 67,80±2,17±1,64 ; Céphéides 72,05±1,86±3,10 ; TRGB+HST 24 SN :
  70,39±1,22±1,33±0,70. TRGB/JAGB s'accordent <1 % entre eux, diffèrent
  des Céphéides de 2,5-4 % en distance. Conclusion CCHP : « consistent
  with ΛCDM, no new physics needed ».
- SH0ES : maintient ~73,0 ; le crowding des Céphéides est EXCLU à 8σ par
  JWST (Riess+ 2024) — la systématique n'est PAS le crowding ; la
  différence CCHP/SH0ES tracée à la SÉLECTION du sous-échantillon SN
  (arXiv:2408.11770).
- Sirènes O4a (dark) : H0 = 81,6 (+21,5/−15,9) ; combiné GW170817 : 76,6
  (précision 5-10 %) — PAS ENCORE arbitre.
- Gaia DR4 : attendu fin 2026 (parallaxes ×√2) — le prochain jalon.
"""
import numpy as np

H_reg = 67.4

def sig(*errs):
    return np.sqrt(sum(e * e for e in errs))

mesures = [
    ("JAGB (JWST seul)",     67.80, sig(2.17, 1.64)),
    ("TRGB (JWST seul)",     68.81, sig(1.79, 1.32)),
    ("TRGB+HST 24 SN",       70.39, sig(1.22, 1.33, 0.70)),
    ("Céphéides (CCHP)",     72.05, sig(1.86, 3.10)),
    ("SH0ES (Céphéides)",    73.04, 1.04),
    ("Sirène GW170817+O4a",  76.60, 7.5),
]
print(f"{'méthode':24s} {'H0':>6} {'±':>5}  écart au registre 67,4")
for nom, h, e in mesures:
    print(f"{nom:24s} {h:6.2f} {e:5.2f}  {(h-H_reg)/e:+5.1f}σ")

# synthèse des DEUX méthodes JWST neuves (indépendantes des Céphéides)
w = [(67.80, sig(2.17, 1.64)), (68.81, sig(1.79, 1.32))]
wts = [1.0 / e**2 for _, e in w]
m = sum(h * t for (h, _), t in zip(w, wts)) / sum(wts)
s = 1.0 / np.sqrt(sum(wts))
print(f"\ncombiné JWST (TRGB+JAGB, hors Céphéides) : {m:.1f} ± {s:.1f}")
print(f"  → à {(m-H_reg)/s:+.1f}σ du registre 67,4 ; à {(m-73.04)/s:+.1f}σ de SH0ES 73,0")

print("""
VERDICT DE VEILLE (03/09/2026) :
1. Les deux calibrateurs NEUFS (JWST, hors Céphéides) sont DÉJÀ dans la
   fenêtre prédite [67,4±1, ≲69] : JAGB 67,8 (+0,4 du registre !), TRGB
   68,8 (le bord « vide » ~69). Combiné 68,4±1,7 : +0,6σ du registre,
   −2,7σ de SH0ES.
2. La tension est devenue une DISPUTE INTER-CALIBRATEURS (TRGB/JAGB vs
   Céphéides, 2,5-4 % en distance) — la « systématique existe » de H1
   est un fait publié, localisé sur UN étage (Céphéides/sélection SN),
   crowding exclu 8σ (la systématique est ailleurs — sélection SN nommée).
3. Sirènes : pas encore arbitres (5-10 %) ; centraux hauts, sans poids.
4. Prochain jalon : Gaia DR4 (fin 2026, parallaxes ×√2) — si les
   Céphéides recalibrées descendent, convergence ; si elles tiennent 73
   avec vide exclu, la chaîne rivière-élasticité est sous pression.
STATUT DE LA PRÉDICTION : EN COURS DE VÉRIFICATION, sens favorable.""")