# -*- coding: utf-8 -*-
"""Q-2 — La fraction EM de tête (le chaînon avant du muon puzzle), 2026-09-02.

Mécanisme TQNT : dans une interaction π-air, la paire K+K̄ SURVIVANTE (la
particule de tête) n'est pas thermalisée — la reconnexion P15 périphérique y
dépose des DEMI-TOURS (le quantum S4″, πσ_χ, celui du V−P universel 0,7 %).
La dégénérescence 2J+1 donne alors V:P = 3:1 sur chaque case de charge.

Théorème de protection (démontré ici) : sous démocratie d'isospin, TOUT
multiplet moyenne ⟨w_EM⟩ = 1/3 (π : (0+1+0)/3 ; ρ : (½+0+½)/3) — le levier
du déficit de muons n'est PAS le comptage linéaire, c'est (i) la case
NEUTRE de tête (π⁰ coûte 1, ρ⁰ coûte 0) et (ii) la non-linéarité de la
cascade (moment-Z, β<1).

Calcul : moment-Z de Heitler-Matthews généralisé — β résout
Z(β) = Σ_fragments hadroniques ⟨x_i^β⟩ = 1 ; N_μ ∝ (E/ξ_c)^β.
Chimies comparées à q (taux d'échange de charge de tête) FIXÉ :
  baseline : tête retenue π⁻ / échangée π⁰ ; centre pions 2/3 hadronique.
  TQNT     : tête {3/4 V, 1/4 P} sur chaque case ; centre = condensat χ
             thermalisé (f_EM = 0,289, calculé dans quatre_herites_trame).
q n'est PAS dérivé (cible nommée : taux d'échange ΔLk périphérique P15) —
il est INVERSÉ de la mesure NA61 Δ_ρ⁰ = 7,7 % (2105.06148, 1705.08206).
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import kn

# ----- entrées canoniques -----
X_L = 0.35          # fraction d'énergie de la particule de tête
N_C = 12            # multiplicité centrale effective
F_C_TQNT = 0.289    # fraction EM du condensat χ thermalisé (calculée H4)
F_C_NAIF = 1.0 / 3.0
E0_XI = 5e8         # E/ξ_c à 1e19 eV (ξ_c^π = 20 GeV)
Z_DEFICIT = 0.35    # ln(data/MC) requis (WHISP)

x_c = (1.0 - X_L) / N_C

# ----- part centrale du rho0 dans le gaz thermique (pour l'inversion de q) -----
T = 156.5
species = [
    ("pi0",   135.0,  1), ("pi+-",  139.6,  2), ("K+-",  493.7,  2),
    ("K0",    497.6,  2), ("eta",   547.9,  1), ("rho0", 775.3,  3),
    ("rho+-", 775.3,  6), ("omega", 782.7,  3), ("K*",   893.6, 12),
    ("eta'",  957.8,  1), ("phi",  1019.5,  3), ("N",    938.9,  8),
    ("Delta", 1232.0, 16),
]
en = {}
for name, m, g in species:
    x = m / T
    en[name] = g * m**2 * T * kn(2, x) * (m * kn(1, x) / kn(2, x) + 3.0 * T)
share_rho0 = en["rho0"] / sum(en.values())
print(f"part d'énergie du ρ⁰ dans le condensat thermalisé : {share_rho0:.3f}")

# ----- inversion de q sur la mesure NA61 -----
DELTA_RHO0_NA61 = 0.077     # ±0,002, π⁻+C 158 GeV/c
central_rho0 = (1.0 - X_L) * share_rho0
q = (DELTA_RHO0_NA61 - central_rho0) / (X_L * 0.75)
print(f"Δ_ρ⁰ : centre {central_rho0:.3f} + tête q·(3/4)·x_L → "
      f"q inversé = {q:.3f} (NA61 7,7 %)")

# ----- moment-Z : Z(β) = somme des x^β hadroniques = 1 -----
def beta_of(frags):
    """frags = [(poids, x, mult)] : Z = Σ poids·mult·x^β."""
    f = lambda b: sum(w * n * x**b for w, x, n in frags) - 1.0
    return brentq(f, 0.5, 1.0)

def chemistry(mode, q):
    """mode : 'naif' | 'tqnt_brut' (3:1 partout) | 'tqnt' (demi-tour PORTÉ
    par la reconnexion : case retenue = pas de contact P15 = pas de dépôt =
    reste π ; case échangée = a reconnecté = 2J+1 → 3:1)."""
    frags = []
    w_had = (1.0 - F_C_NAIF) if mode == "naif" else (1.0 - F_C_TQNT)
    frags.append((w_had, x_c, N_C))
    if mode == "naif":
        frags.append((1.0 - q, X_L, 1))          # π⁻ retenu ; π⁰ échangé → 0
    elif mode == "tqnt_brut":
        frags.append(((1 - q) * 0.25, X_L, 1))       # π⁻
        frags.append(((1 - q) * 0.75, X_L / 2, 1))   # ρ⁻ → π⁻ (π⁰ meurt)
        frags.append((q * 0.75, X_L / 2, 2))         # ρ⁰ → π⁺π⁻ (2 fragments)
    else:  # 'tqnt' — l'excitation voyage avec la reconnexion
        frags.append((1.0 - q, X_L, 1))              # π⁻ retenu, froid
        frags.append((q * 0.75, X_L / 2, 2))         # ρ⁰ → π⁺π⁻
        # π⁰ échangé (q/4) → 0
    return frags

print()
print(f"{'q':>5} {'β_naïf':>8} {'brut':>8} {'TQNT':>8} {'×N_μ brut':>10} {'×N_μ TQNT':>10}  (requis 1,42)")
for qq in (0.0, 0.10, q, 0.30):
    b0 = beta_of(chemistry("naif", qq))
    bb = beta_of(chemistry("tqnt_brut", qq))
    b1 = beta_of(chemistry("tqnt", qq))
    rb = np.exp((bb - b0) * np.log(E0_XI))
    r1 = np.exp((b1 - b0) * np.log(E0_XI))
    tag = " ← q(NA61)" if abs(qq - q) < 1e-9 else ""
    print(f"{qq:5.2f} {b0:8.4f} {bb:8.4f} {b1:8.4f} {rb:10.2f} {r1:10.2f}{tag}")

# décomposition au q inversé
b0 = beta_of(chemistry("naif", q))
frags_tete = [(1.0 - F_C_NAIF, x_c, N_C), (1.0 - q, X_L, 1), (q * 0.75, X_L / 2, 2)]
frags_centre = [(1.0 - F_C_TQNT, x_c, N_C), (1.0 - q, X_L, 1)]
r_tete = np.exp((beta_of(frags_tete) - b0) * np.log(E0_XI))
r_centre = np.exp((beta_of(frags_centre) - b0) * np.log(E0_XI))
print(f"\ndécomposition à q={q:.2f} : tête seule ×{r_tete:.2f}, centre seul ×{r_centre:.2f}")

b1 = beta_of(chemistry("tqnt", q))
gain = (b1 - b0) * np.log(E0_XI)
print(f"\nvs cascade naïve : ln(gain) = {gain:+.2f} (requis {Z_DEFICIT}) — "
      f"la chimie TQNT SURCOUVRE (×{np.exp(gain):.2f})")

# ----- la comparaison pertinente : vs GÉNÉRATEUR moderne -----
# les générateurs ont le centre à résonances MAIS sous-estiment le ρ⁰ avant
# de ~60 % dans 0,4<x_F<0,7 (NA61) → case échangée : ~40 % du 3:1 TQNT.
for f_gen, lab in ((0.40, "générateur −60 % (QGSJet/EPOS)"),
                   (0.90, "générateur −10 % (Sibyll 2.3)")):
    frags_gen = [(1.0 - F_C_TQNT, x_c, N_C), (1.0 - q, X_L, 1),
                 (q * 0.75 * f_gen, X_L / 2, 2)]
    b_gen = beta_of(frags_gen)
    r = np.exp((b1 - b_gen) * np.log(E0_XI))
    print(f"TQNT / {lab:32s} : ×{r:.2f}")
print(f"(requis data/générateur : ×{np.exp(Z_DEFICIT):.2f} vs QGSJet/EPOS ;"
      f" Sibyll 2.3d ~×1,2 résiduel)")
print("VERDICT DE MÉCANISME : la variante brute (3:1 aussi sur la case")
print("retenue) donne ×0,89 < 1 — RÉFUTÉE par le signe même du déficit ;")
print("le demi-tour est PORTÉ PAR LA RECONNEXION (pas de contact P15 → pas")
print("de dépôt → le survivant retenu reste π). Falsifiables : ρ⁰:π⁰ = 3")
print("sur la case échangée à grand x_F ; ρ⁻ avant SUPPRIMÉ vs comptage")
print("de spin naïf. Cible nommée : dériver q (échange ΔLk périphérique).")
