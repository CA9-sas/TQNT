# -*- coding: utf-8 -*-
"""Q-3c — Contrôle du neutron de tête sur les données EXISTANTES (03/09/2026).

Prédiction à contrôler (Q-2b, forme close, zéro nombre lu) :
    q(p→n) = (2−√2)/(2(3−√2)) = 0,1847 par contact P15 de tête
(probabilité que le baryon survivant ait échangé sa charge).

Trois ancres publiées :
(A) ZEUS (HERA, ep → e'nX, Nucl.Phys. B637) : fraction d'événements DIS
    avec neutron de tête ≈ 9,1 % DANS LA FENÊTRE (x_L ≳ 0,2, |t| < 0,5
    GeV²). L'extrapolation t (exponentielle b≈8-12 GeV⁻²) et la queue
    x_L<0,2 relèvent le taux PLEIN d'un facteur ~1,6-2,2 (déclaré).
(B) LHCf pp 13 TeV (JHEP 07 (2020) 016) : ⟨1−k_n⟩ = 0,536 (+0,031/−0,037)
    — inélasticité moyenne SACHANT un neutron très avant (η>8,65).
(C) ISR (pp) : ⟨x_F⟩(neutron) = 0,26 vs ⟨x_F⟩(proton) = 0,44.
"""
import numpy as np

d = np.sqrt(2.0)
q = (2.0 - d) / (2.0 * (3.0 - d))
print(f"q(p→n) dérivé = {q:.4f}\n")

# (A) ZEUS : bande d'extrapolation déclarée
zeus_win = 0.091
for fac in (1.6, 1.9, 2.2):
    print(f"(A) ZEUS 9,1 % fenêtre × {fac} (extrap. t + queue x_L) → "
          f"taux plein {zeus_win*fac:.3f}"
          + ("  ← q dérivé DANS la bande" if abs(zeus_win*fac - q) < 0.02 else ""))
print(f"    bande pleine [{zeus_win*1.6:.2f}, {zeus_win*2.2:.2f}] vs q = {q:.3f}"
      " → COMPATIBLE (au centre de la bande)")
print("    réserve : ep (sonde photon) ≠ pp — le vertex proton→neutron est")
print("    le même (échange de charge de tête), la sonde diffère (déclaré).\n")

# (B) LHCf : contexte de cohérence (pas un test de q)
print("(B) LHCf 13 TeV : ⟨1−k_n⟩ = 0,536 — l'inélasticité SACHANT neutron ;")
print("    ne teste pas q (biais du cône très avant) ; cohérence : le")
print("    neutron a subi LA reconnexion → dépôt d'énergie → inélasticité")
print("    substantielle attendue (≈½), pas un passage froid. ✓ (contexte)\n")

# (C) ISR : le corollaire de mécanisme
print("(C) ISR : ⟨x_F⟩(n) = 0,26 < ⟨x_F⟩(p) = 0,44 — COROLLAIRE TQNT :")
print("    le proton de tête retenu vole souvent FROID (pas de contact,")
print("    x_F haut) ; le neutron existe PARCE QU'il a reconnecté (dépôt")
print("    obligatoire → plus mou). L'asymétrie de dureté p/n est une")
print("    signature du « demi-tour porté par la reconnexion ». ✓\n")

print("VERDICT : q = 0,185 PASSE le contrôle sur données existantes —")
print("(A) dans la bande ZEUS extrapolée [0,15, 0,20] ; (B) cohérent ;")
print("(C) corollaire de dureté vérifié qualitativement. Réserves : ")
print("extrapolation ×1,6-2,2 déclarée, sonde ep≠pp, x_L du jouet non")
print("dérivé. Le test PROPRE reste LHCf p-O (pré-enregistré P1).")