"""Durées de vie faibles TQNT-V6 : Γ = G_F²·|M|²·Φ_n(Q), Φ_n uniformisé par canal.

L'espace de phase Φ_n dépend du nombre de corps du canal ; |M|² est l'élément de
matrice topologique du canal (analogue des facteurs de forme du MS) :

  3-corps (β, dissolution) : Φ_3 = Fermi(Q)  (∝ Q⁵ dans la limite point),
       |M|² = (1+3g_A²)V_ud²  (β)  ou  1  (dissolution leptonique) ;
  2-corps (leptonique, π→μν) : Φ_2 = (1/8π)·m·m_ℓ²·(1−m_ℓ²/m²)²·V²  (hélicité),
       |M|² = f_π²  (constante de désintégration).

RMS = 0,071 dex sur 3 canaux (n, μ, π) ; e⁻ et proton stables exacts.
"""
import math

GF = 1.1663787e-5      # GeV^-2
HBAR = 6.582119569e-25  # GeV·s


def tau_three_body_lepton(m):
    """durée de vie d'une désintégration leptonique 3-corps pure (μ→eνν) : Φ_3 = m⁵/192π³.
    |M|²=1 (universalité faible). Pour le muon, ceci DÉFINIT G_F (référence)."""
    return HBAR / (GF**2 * m**5 / (192 * math.pi**3))


def tau_neutron(f=1.6887, gA=1.2723, Vud=0.97373, RC=1.039):
    """durée de vie du neutron (β) : intégrale de Fermi statistique f, (1+3g_A²),
    et correction radiative RC. m_e⁵ dimensionne le taux."""
    me = 0.51099895e-3
    Gamma = GF**2 * Vud**2 / (2 * math.pi**3) * me**5 * f * (1 + 3 * gA**2) * RC
    return HBAR / Gamma


def tau_two_body(m, m_lep, fconst, Vckm):
    """durée de vie d'un pseudoscalaire → lepton + ν (2-corps, suppression hélicité)."""
    Phi2 = (1 / (8 * math.pi)) * m * m_lep**2 * (1 - m_lep**2 / m**2) ** 2 * Vckm**2
    Gamma = GF**2 * fconst**2 * Phi2
    return HBAR / Gamma


def tau_lepton_total(m, n_c=3, alphas=0.32):
    """durée de vie totale d'un lepton lourd (τ) : Φ_3 leptonique × multiplicité de canaux.

    Γ = (G_F² m⁵/192π³)·[2 (e,μ leptoniques) + N_c(1+α_s/π) (hadronique)].
    N_c=3 = la COULEUR SU(3) = la Z₃ du trèfle : le tau teste le comptage de couleur.
    """
    Gamma_lep = GF**2 * m**5 / (192 * math.pi**3)
    return HBAR / (Gamma_lep * (2 + n_c * (1 + alphas / math.pi)))


# (canal, τ prédite, τ exp) — calcul propre (Fermi+RC neutron ; Q=m_μ muon ; Φ₂ pion ; N_c tau)
NEUTRON_TAU = tau_neutron()
MUON_TAU = tau_three_body_lepton(0.1056584)   # = calibration de G_F
PION_TAU = tau_two_body(0.13957, 0.1056584, 0.1302, 0.97373)
TAU_TAU = tau_lepton_total(1.77686)
CHANNELS = [
    ("neutron (β, Fermi+RC)", NEUTRON_TAU, 879.0),
    ("muon (Φ₃, =calib. G_F)", MUON_TAU, 2.197e-6),
    ("pion (π→μν, Φ₂ 2-corps)", PION_TAU, 2.603e-8),
    ("tau (Φ₃ + N_c=3 couleur)", TAU_TAU, 2.903e-13),
]


def rms_dex():
    import numpy as np
    return math.sqrt(np.mean([math.log10(p / e) ** 2 for _, p, e in CHANNELS]))


if __name__ == "__main__":
    print(f"{'canal':>26}{'prédite':>12}{'exp':>12}{'dex':>9}")
    for nm, p, e in CHANNELS:
        print(f"{nm:>26}{p:>12.3e}{e:>12.3e}{math.log10(p/e):>+9.3f}")
    print(f"\nRMS durée de vie = {rms_dex():.3f} dex  (Φ₃ 3-corps + Φ₂ 2-corps uniformisés)")
