r"""CKM comme distance de reconnexion topologique (Règle 0 : CKM=reconnexions P15).

Voir docs/brouillon_CKM_reconnexion.tex. La HIÉRARCHIE CKM = distance de reconnexion :
|V_ij| ~ λ^(distance), pas adjacent gén-n↔n+1 ~ λ^n (complexité croissante) → structure
de Wolfenstein depuis un seul λ. λ=√(m_d/m_s) (GST, masses=Z₃) à 0.6%. 8/9 magnitudes
<9% avec (λ, A). V_ub exception (+153%, porte la phase CP ρ,η). DÉRIVÉ = la structure
(hiérarchie=puissances de λ=distance de reconnexion) ; NON dérivé = valeurs (puzzle de saveur),
A, phase CP. La TQNT recadre CKM topologiquement sans fermer le puzzle de la saveur.
"""
import numpy as np

V_EXP = {'ud': 0.97435, 'us': 0.22500, 'ub': 0.00369,
         'cd': 0.22486, 'cs': 0.97349, 'cb': 0.04182,
         'td': 0.00857, 'ts': 0.04110, 'tb': 0.999118}
M_D, M_S, M_B = 4.7, 93.4, 4180.0   # MeV, masses courantes (= Z₃/complexité en TQNT)


def lambda_cabibbo():
    return np.sqrt(M_D / M_S)          # Gatto-Sartori-Tonin


def wolfenstein(L, A=0.826):
    return {'ud': 1 - L**2/2, 'us': L, 'ub': A*L**3,
            'cd': L, 'cs': 1 - L**2/2, 'cb': A*L**2,
            'td': A*L**3, 'ts': A*L**2, 'tb': 1 - A**2*L**4/2}


if __name__ == '__main__':
    L = lambda_cabibbo()
    print(f"λ = √(m_d/m_s) = {L:.4f}  (exp Cabibbo 0.2257, {(L-0.2257)/0.2257*100:+.1f}%)")
    mod = wolfenstein(L)
    d = []
    for k in V_EXP:
        dd = (mod[k] - V_EXP[k]) / V_EXP[k] * 100
        d.append(abs(dd))
        tag = '  <-- exception (phase CP)' if k == 'ub' else ''
        print(f"  V_{k}: {mod[k]:.5f} vs {V_EXP[k]:.5f}  Δ={dd:+.1f}%{tag}")
    d8 = [x for k, x in zip(V_EXP, d) if k != 'ub']
    print(f"RMS 8/9 (sans V_ub) = {np.sqrt(np.mean(np.square(d8))):.1f}% ; V_ub seul = +153%")


def cp_phase_from_chirality(k=135):
    """δ_CP depuis la chiralité du trèfle : framing (writhe 3) + base Z₃ π/3.
    Chiralité = 2π·h_{1/2}·writhe = 5.9° (dérivé) ; base π/3=60° (heuristique).
    δ = 65.9° vs γ_CKM 65.7° (0.3%). Amphichiral writhe=0 → 0° → CP-sym (falsifiable)."""
    import numpy as np
    h = 0.75 / (k + 2)
    phi_chir = 3 * 2 * np.pi * h          # writhe 3
    delta = np.pi / 3 + phi_chir
    return np.degrees(phi_chir), np.degrees(delta)


if __name__ == '__main__':
    pc, d = cp_phase_from_chirality()
    print(f"chiralité trèfle = {pc:.2f}° (framing writhe 3) ; δ = π/3 + chir = {d:.2f}° "
          f"(exp γ 65.7°, {(d-65.7)/65.7*100:+.1f}%)")


def pi_over_3_justification():
    """π/3 = angle intérieur de l'équilatéral des 3 modes Z₃ (0,120,240°).
    Lien CP : Jarlskog = 2×aire du triangle d'unitarité → CP = aire de triangle.
    Z₃-symétrique → triangle équilatéral → phase = π/3. La hiérarchie de masse
    déforme les CÔTÉS (α,β≠60°) mais pas la PHASE (γ=δ~60°). Motivé, non théorème."""
    import numpy as np
    modes = [np.degrees(np.angle(np.exp(2j*np.pi*k/3))) for k in range(3)]
    return modes, 60.0  # positions, angle équilatéral


def delta_eq_pi3_REFUTED():
    """RÉFUTATION de δ=π/3 : matrice de masse Z₃-symétrique = circulante = diagonalisée
    par la base de Fourier F. Si up ET down circulants → CKM=F†F=1 → AUCUNE phase CP.
    Z₃ EXACT → CKM=identité → δ non défini. δ = effet de BRISURE (pattern-dépendant),
    NON dérivable de Z₃ non brisé. De plus, 2 triangles distincts (modes Z₃ vs unitarité) ;
    Jarlskog = aire du triangle d'UNITARITÉ, pas des modes. δ=π/3 RÉTRACTÉ (coïncidence)."""
    import numpy as np
    w = np.exp(2j * np.pi / 3)
    P = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    circ = lambda a, b, c: a * np.eye(3) + b * P + c * P @ P
    Mu, Md = circ(1.0, 0.3, 0.1), circ(0.8, 0.2, 0.15)
    _, Uu = np.linalg.eigh(Mu.conj().T @ Mu)
    _, Ud = np.linalg.eigh(Md.conj().T @ Md)
    return np.abs(Uu.conj().T @ Ud)   # ≈ bloc trivial : pas de mélange 3-génération/CP


if __name__ == '__main__':
    print("REFUTATION δ=π/3 : |CKM| pour 2 secteurs Z₃-circulants =")
    print(delta_eq_pi3_REFUTED(), "\n→ CKM trivial (Z₃ exact = pas de CP). δ=π/3 rétracté.")


def cp_tqnt_native(k=135):
    """CP TQNT-native : phase de framing d'une reconnexion P15, SANS matrice de masse.
    θ_f = 2π·h_(1/2) = 2π·(3/4)/(k+2) = 1.97°/writhe (DÉRIVÉ). A_CP ∝ sin(θ_f·ΔW).
    Amphichiral (W=0) → CP conservé ; chiral → CP violé (falsifiable, DUNE).
    HONNÊTE : framing du trèfle = 3θ_f = 5.9°, mais δ_CKM~66°, δ_PMNS~90° (10-15× trop grand)
    → la chiralité donne le SIGNE + un petit quantum ; la grande magnitude NON expliquée.
    Aucune matrice de masse, aucun δ_SM, aucun π/3 (rétracté : Z₃ exact → CKM=1)."""
    import numpy as np
    theta_f = np.degrees(2 * np.pi * (0.75 / (k + 2)))
    return theta_f, 3 * theta_f   # quantum/writhe, framing du trèfle (W=3)


if __name__ == '__main__':
    tf, tref = cp_tqnt_native()
    print(f"θ_f = {tf:.2f}°/writhe (dérivé) ; framing trèfle = 3θ_f = {tref:.1f}° "
          f"(vs δ_CKM~66°, δ_PMNS~90° : 10-15× trop petit → magnitude ouverte, SIGNE ok)")


def cp_magnitude_from_Z3():
    """Magnitude CP = phase de génération Z₃ (TQNT-native, sans matrice de masse).
    3 générations = 3 modes Z₃ ; reconnexion j↔k → sin(2π·Δg/3) : phase 120°(Δg=1)/60°(Δg=2).
    ÉCHELLE 60-120° = bonne magnitude pour CP observé (vs framing 6°).
    Lepton = Z₃ PUR (non coloré) → CP MAXIMALE ~90° (cohérent δ_PMNS~-90°, falsifiable DUNE).
    Quark = Z₃ habillé QCD → réduit à δ_CKM~66° (facteur ~0.73 NON dérivé).
    Rôles séparés : chiralité(framing 6°)=SIGNE ; Z₃ génération=MAGNITUDE. Valeurs exactes ouvertes."""
    import numpy as np
    return {'Z3_adjacent_deg': 120.0, 'Z3_half_deg': 60.0,
            'lepton_maximal_deg': 90.0, 'quark_ckm_deg': 65.7}


if __name__ == '__main__':
    r = cp_magnitude_from_Z3()
    print(f"phase Z₃ adjacent=120°/60° = échelle CP (vs framing 6°); lepton~90° (Z₃ pur), "
          f"quark~66° (Z₃ habillé QCD). Magnitude=Z₃, signe=chiralité. Valeurs exactes ouvertes.")


def qcd_dressing_mechanism():
    """Dressing QCD du quark : la couleur est CP-CONSERVANTE (θ_QCD≈0, problème CP fort)
    → ajoute une amplitude CP-paire qui DILUE le CP-impair du Z₃ → δ_quark<δ_lepton (66<90).
    MÉCANISME DÉRIVÉ (relie la diff CP lepton/quark au problème CP fort). FACTEUR non dérivé :
    rapport CP-pair/impair=1/tan(66°)=0.45 ~ C_F/N_c=4/9=0.44 (suggestif, NON revendiqué :
    look-elsewhere + cible δ_PMNS incertaine ±30-40°). Nombre = amplitude QCD non-perturbative."""
    import numpy as np
    return 1/np.tan(np.radians(65.7)), (4/3)/3   # ratio observé, C_F/N_c


def qcd_dressing_ising():
    """Dressing QCD TQNT-NATIVE : couleur = vortex d'Ising SU(2)_2 (k=2), PAS SU(3).
    Correction de cadre : C_F/N_c=4/9 était SM (Casimir SU(3)) ; le 5π/12 de G_F était
    SM (espace de phase). En TQNT l'anyon couleur = σ d'Ising (h_σ=1/16), spin topologique
    θ_σ=e^(iπ/8)=π/8=22.5° (DÉRIVÉ, k=2). δ_quark ≈ δ_lepton(90°) − θ_σ = 67.5° vs 66° (+2.7%).
    MOTIVÉ TQNT (θ_σ dérivé), NON dérivé (signe+exactitude via tressage d'Ising, non fermé)."""
    import numpy as np
    theta_sigma = np.degrees(2 * np.pi * (1/16))   # spin topo anyon σ d'Ising
    return theta_sigma, 90.0 - theta_sigma          # π/8, δ_quark prédit


if __name__ == '__main__':
    ts, dq = qcd_dressing_ising()
    print(f"θ_σ(Ising)=π/8={ts:.1f}° (dérivé k=2) ; δ_quark=90-θ_σ={dq:.1f}° vs 66° (+2.7%)"
          f" — TQNT-native (Ising), pas SU(3)/phase-space SM")


def ising_braiding_cp():
    """Tressage d'Ising SU(2)_2 dans la reconnexion couleur (CALCUL FAIT).
    R^σσ_ψ = e^(i3π/8), vérifié par la relation de ruban R_c²=θ_c/θ_σ² (θ_σ=e^(iπ/8), θ_ψ=-1).
    Quark = fermion → canal ψ → δ_quark = arg(R^σσ_ψ) = 3π/8 = 67.5° vs γ_CKM 65.7°±3° (DANS 1σ).
    3π/8 = π/2−π/8 : le 'maximal−θ_σ' EST la demi-tresse d'Ising, DÉRIVÉE de k=2 (pas un hack).
    Motivé non prouvé : canal ψ + observable=arg(R) demi-tresse. Lepton (sans σ) = Z₃, à part."""
    import numpy as np
    R_ss_psi = np.exp(1j * 3 * np.pi / 8)
    # vérif ruban
    theta_s, theta_psi = np.exp(1j*np.pi/8), np.exp(1j*np.pi)
    ribbon_ok = np.isclose(R_ss_psi**2, theta_psi/theta_s**2)
    return np.degrees(np.angle(R_ss_psi)), ribbon_ok


if __name__ == '__main__':
    d, ok = ising_braiding_cp()
    print(f"R^σσ_ψ (Ising k=2, ruban {'✓' if ok else 'NON'}) → δ_quark=arg=3π/8={d:.1f}° "
          f"vs γ_CKM 65.7°±3° (dans 1σ). DÉRIVÉ, pas ajusté.")


def prove_psi_channel_REFUTED():
    """Preuve du canal ψ / arg(R) → RÉFUTÉE (3 obstacles fatals) :
    (1) COLOR-BLIND : δ_CKM=phase faible, or le faible ne distingue pas les couleurs
        → σ spectateur, ne se tresse pas dans le vertex CP faible.
    (2) SINGULET : CP mesurée dans mésons = singulets couleur (σ×σ̄→canal 1 = -22.5°), pas ψ.
    (3) SECTEUR : Ising = phase FORTE ; δ_CKM = FAIBLE → identification = erreur de catégorie.
    ⇒ δ_quark=R^σσ_ψ=67.5° NON prouvé ; match à 66° probablement COÏNCIDENT (comme π/3).
    PATTERN saveur : quantités TQNT dérivées coïncident mais AUCUN mécanisme ne survit au
    calcul explicite. Puzzle de saveur NON résolu. Seul survit : CP=chiralité qualitatif (DUNE)."""
    return {'color_blind': True, 'meson_singlet_channel': 1, 'weak_vs_strong': True,
            'psi_channel_proven': False, 'likely_coincidence': True}


if __name__ == '__main__':
    print("Preuve canal ψ:", prove_psi_channel_REFUTED())
