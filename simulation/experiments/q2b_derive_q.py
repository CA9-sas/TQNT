# -*- coding: utf-8 -*-
"""Q-2b — Dériver q, le taux d'échange de charge de tête (03/09/2026).

Chaîne en trois pièces, zéro nombre nouveau :

(1) PARTITION DE LA FENÊTRE DE CAPTURE (théorème canonique, celle-là même
    qui donne α_cond = ln(2+√2)) : à un contact P15, les issues se
    partagent en PASSAGE ΔLk=±1 (transfert d'enlacement) et SCISSION
    (fragmentation), dans le rapport 2(d−1)/d = 2−√2 (d=√2, TL k=2 exact).
    → P(passage) = (2−√2)/(3−√2) = 0,3694 — la valeur mesurée de la
    cascade branchée (0,372) la confirme.
(2) TRANSFERT au contact périphérique d'une collision : « α par croisement
    est UNIVERSEL » — la partition est une propriété locale du croisement
    P15, pas du gel. CONDITIONNEL déclaré (même rang que la forme close
    d'α_cond : identification boucle↔demi-tour).
(3) SÉLECTION DE SENS PAR LA CHARGE : les 2 directions de capture = les 2
    sens ΔLk (canonique). Pour une tête chargée (Q=±1), UN seul sens mène
    à un multiplet léger existant (Q=0) ; l'autre (Q=±2) n'a pas d'état —
    stérile. → q(chargé) = ½·P(passage).

    q = (2−√2)/(2(3−√2)) = 0,1847

COROLLAIRE falsifiable : une tête NEUTRE (π⁰/ρ⁰) a les DEUX sens ouverts
→ taux d'échange DOUBLE (0,369) — asymétrie chargé/neutre testable.
Règle du null : q = 1/6 = 0,167 colle mieux à l'inversion centrale mais
n'a AUCUN mécanisme (comptage de directions de rails inapplicable à un
contact de tête) — non retenu, consigné.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import kn

d = np.sqrt(2.0)
P_passage = (2.0 - d) / (3.0 - d)
q_th = 0.5 * P_passage
print(f"P(passage)  = (2−√2)/(3−√2) = {P_passage:.4f}  (cascade mesurée : 0,372)")
print(f"q dérivé    = ½·P(passage)  = {q_th:.4f}")

# ----- confrontation à l'inversion NA61 (bande x_L) -----
share_rho0 = 0.051          # part ρ⁰ du condensat thermalisé (Q-2)
DELTA_NA61 = 0.077
print(f"\ninversion NA61 (Δ_ρ⁰ = 7,7 %) selon x_L :")
for xl in (0.30, 0.35, 0.40):
    q_inv = (DELTA_NA61 - (1.0 - xl) * share_rho0) / (xl * 0.75)
    print(f"  x_L = {xl:.2f} : q_inv = {q_inv:.3f}  (dérivé {q_th:.3f} : "
          f"{(q_th/q_inv-1)*100:+.0f} %)")
print("→ le dérivé tombe DANS la bande d'inversion (exact à x_L=0,30,")
print("  +11 % au central x_L=0,35) — compatible aux incertitudes du jouet.")

# ----- boucle refermée : Δ_ρ⁰ et muons PRÉDITS avec q dérivé -----
X_L, N_C, F_C = 0.35, 12, 0.289
x_c = (1.0 - X_L) / N_C
E0_XI = 5e8

pred_drho = (1.0 - X_L) * share_rho0 + q_th * 0.75 * X_L
print(f"\nΔ_ρ⁰ PRÉDIT (q dérivé, x_L=0,35) = {pred_drho:.3f} "
      f"vs NA61 0,077 ({(pred_drho/DELTA_NA61-1)*100:+.0f} %)")

def beta_of(frags):
    f = lambda b: sum(w * n * x**b for w, x, n in frags) - 1.0
    return brentq(f, 0.5, 1.0)

def frags_tqnt(q):
    return [(1.0 - F_C, x_c, N_C), (1.0 - q, X_L, 1), (q * 0.75, X_L / 2, 2)]

def frags_gen(q, f_gen):
    return [(1.0 - F_C, x_c, N_C), (1.0 - q, X_L, 1),
            (q * 0.75 * f_gen, X_L / 2, 2)]

b1 = beta_of(frags_tqnt(q_th))
for f_gen, lab in ((0.40, "QGSJet/EPOS (−60 % ρ⁰)"), (0.90, "Sibyll 2.3 (−10 %)")):
    r = np.exp((b1 - beta_of(frags_gen(q_th, f_gen))) * np.log(E0_XI))
    print(f"muons TQNT / {lab:24s} : ×{r:.2f}")
print("(requis ×1,42 vs QGSJet/EPOS ; ~×1,2 vs Sibyll — inchangé au q dérivé)")

print("\nSTATUT : q = (2−√2)/(2(3−√2)) — dérivation CONDITIONNELLE (2 pièces :")
print("universalité de la partition au croisement ; sens stérile Q=±2).")
print("La chaîne du muon puzzle n'a plus de nombre LU : q dérivé, V:P=3:1")
print("(demi-tour porté par la reconnexion), centre = condensat thermalisé.")
print("Falsifiables : asymétrie chargé/neutre ×2 du taux d'échange ;")
print("Δ_ρ⁰ = 8,2 % (+6 % de NA61) ; ρ⁻ avant supprimé.")