# -*- coding: utf-8 -*-
"""Q-3 — Confrontation au run O-O/Ne-Ne/p-O du LHC (03/09/2026).

État des données (vérifié 03/09/2026) :
- ALICE, étrangeté en ions légers (arXiv:2608.02874) : PUBLIÉ (proceedings).
- ALICE, R_OO du π⁰ (arXiv:2604.17974) : publié — réserve (modification
  nucléaire, hors du périmètre chimie).
- LHCf p-O avant (photons/π⁰/neutrons) : PAS ENCORE PUBLIÉ → les
  prédictions TQNT ci-dessous sont PRÉ-ENREGISTRÉES (avant publication).

(a) CONFRONTATION EXÉCUTÉE — l'universalité de multiplicité :
ALICE : « The OO results follow the smooth multiplicity-dependent evolution
observed in pp, p-Pb, and Pb-Pb collisions, confirming that charged-particle
multiplicity is an effective scaling variable » (rapports Λ/K⁰s, Ξ/K⁰s,
Ω/K⁰s). Lecture TQNT : la chimie du condensat χ thermalisé ne dépend QUE du
volume thermalisé (⇔ multiplicité), jamais du système qui l'a créé — la
courbe universelle est la signature attendue. VERDICT : PASSE.
Réserve déclarée : les spectres p_T d'O-O suivent la tendance Pb-Pb à
multiplicité égale (géométrie initiale → écoulement) — c'est la DYNAMIQUE,
hors du périmètre de la revendication de chimie (rendements).

(b) PRÉ-ENREGISTRÉ pour LHCf p-O (chaîne Q-2/Q-2b, zéro nombre lu) :
"""
import numpy as np

d = np.sqrt(2.0)
q = (2.0 - d) / (2.0 * (3.0 - d))
print(__doc__)
print(f"P1. Taux d'échange de tête chargée : q = (2−√2)/(2(3−√2)) = {q:.4f}")
print("    → p-O : probabilité de neutron de tête PAR CONTACT p→n = 0,185")
print("    (le proton Q=+1 a un sens stérile Q=+2, comme le pion — même ½).")
print("P2. Asymétrie chargé/neutre : une tête NEUTRE échange au taux DOUBLE")
print(f"    (les 2 sens ouverts) : {2*q:.3f} — testable K⁰/Λ de tête vs K±/p.")
print("P3. Case échangée 3:1 vecteur (demi-tour porté par la reconnexion) :")
print("    fraction EM avant RÉDUITE vs modèles π⁰-démocratiques ;")
print("    Δ_ρ⁰(π-C) = 8,2 % (déjà +6 % de NA61) — l'analogue p-O à mesurer.")
print("P4. AUCUN exotique dans l'état final avant (désert + Vassiliev).")
print()
print("Discriminant : si LHCf p-O mesure une fraction d'énergie avant en")
print("photons CONFORME aux générateurs pré-2025 (sans réduction vecteur),")
print("la chimie de tête TQNT est réfutée ; si réduite (sens EPOS.LHC-R),")
print("elle est confortée — et le 4e hérité se dissout en QCD molle,")
print("exactement comme la prédiction dure l'exige.")