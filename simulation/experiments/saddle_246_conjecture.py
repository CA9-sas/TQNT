r"""
Conjecture des trois visages de l'échelle 246 GeV (k=28).

L'entrelacs-selle de la reconnexion faible (9_1²+10_1³ ou 7_2+10_1⁴, k=28,
≈246 GeV) est :
  (i)  couplage faible : selle virtuelle -> G_F (neutron_equilibration.py) ;
  (ii) matière noire   : nœud réel, dénouement supprimé -> stable, invisible ;
  (iii) Higgs          : 246 GeV = énergie de déconfinement (E=mc²) = le Higgs.

Usage :  python3 -m simulation.experiments.saddle_246_conjecture
"""
import numpy as np
from theory.postulates import K_CS

def main():
    print("="*70)
    print("  246 GeV (k=28) — trois visages")
    print("="*70)
    # (ii) matière noire : 9_1²+10_1³, croisements et dénouement
    cross = 2*9 + 3*10
    u_tot = 2*4 + 3*1            # u(9_1=T(2,9))=4 ; u(10_1)=1
    print(f"\n  (ii) MATIÈRE NOIRE — 9₁²+10₁³ :")
    print(f"       croisements = {cross}  (seuil c*=18) -> régime mat. noire")
    print(f"       dénouement u = 2·4 + 3·1 = {u_tot}")
    S = 2*u_tot*K_CS*np.pi
    print(f"       barrière exp(2u·k_CS·π) = e^{S:.0f} ≈ 10^{S/np.log(10):.0f}")
    print(f"       τ ~ 10^{S/np.log(10):.0f} s ≫ âge univers (10^17,6 s) -> STABLE/invisible")
    # (iii) Higgs : déconfinement
    print(f"\n  (iii) HIGGS — déconfinement :")
    print(f"       E_déconf = 246 GeV (E=mc², coût de déconfinement topologique)")
    print(f"       => le boson de Higgs (excitation de déconfinement de l'échelle faible)")

    # CRUCIAL : budget d'énergie — la selle est-elle produite en β ?
    HBARC=197.327; m_n=939.565; m_p=938.272; m_e=0.511
    Q=m_n-m_p-m_e; E_s=246000.
    print(f"\n  CRUCIAL — budget d'énergie en β (n -> p e ν̄) :")
    print(f"       neutron {m_n/1000:.2f} GeV vs selle {E_s/1000:.0f} GeV "
          f"-> {E_s/m_n:.0f}× trop léger")
    print(f"       λ dispo (ħc/Q={HBARC/Q:.0f} fm) vs requise (ħc/E_s={HBARC/E_s:.1e} fm)"
          f" -> {HBARC/Q/(HBARC/E_s):.0e}× trop longue")
    print(f"       => selle VIRTUELLE (hors couche) : β produit SEULEMENT e⁻,ν̄,p.")
    print(f"          mat. noire / Higgs uniquement SUR COUCHE (246 GeV réels).")
    print("="*70)

if __name__ == "__main__":
    main()
