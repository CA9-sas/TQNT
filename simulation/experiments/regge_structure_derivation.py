r"""
La STRUCTURE de Regge (M² ∝ J) se dérive du tube χ relativiste en rotation,
SANS passer par la valeur de σ_χ.

Tube (corde Nambu-Goto) de tension σ, mi-longueur R, pointes tournant à c :
vitesse v(r)=r/R (unités c), facteur relativiste γ=1/√(1−v²).
  Énergie          M = 2∫₀ᴿ σ γ dr        = 2σR·(π/2) = π σ R
  Moment cinétique J = 2∫₀ᴿ σ γ v r dr    = 2σR²·(π/4) = π σ R²/2
  ⇒ M²/J = 2πσ  ⇒  M² = 2πσ·J   (trajectoire de Regge LINÉAIRE).

POINT CLÉ : la LINÉARITÉ M²∝J est une conséquence KINÉMATIQUE de la rotation
relativiste (pointes à c). σ n'apparaît que comme facteur d'ÉCHELLE (la pente
1/α'=2πσ), PAS dans la structure. Donc la STRUCTURE se dérive SANS connaître σ_χ.

CONSÉQUENCE : la pente est UN paramètre mesuré (= la tension), comme en QCD :
pente = 1,096 GeV²/J (5 trajectoires, r²≥0,997). L'exprimer comme 8πΛ_χ²
reposait sur la coïncidence σ_χ=4Λ_χ² (rejetée) — on l'abandonne. Ce qui est
DÉRIVÉ (sans σ_χ) est la linéarité M²∝J ; la pente est mesurée.
"""
from __future__ import annotations
import sympy as sp


def derive():
    """Retourne (M, J, M²/J) pour le tube relativiste en rotation."""
    u, sig, R = sp.symbols("u sigma R", positive=True)
    I_M = sp.integrate(1 / sp.sqrt(1 - u ** 2), (u, 0, 1))      # = π/2
    I_J = sp.integrate(u ** 2 / sp.sqrt(1 - u ** 2), (u, 0, 1))  # = π/4
    M = sp.simplify(2 * sig * R * I_M)          # π σ R
    J = sp.simplify(2 * sig * R ** 2 * I_J)     # π σ R²/2
    slope = sp.simplify(M ** 2 / J)             # 2πσ
    return M, J, slope


def main():
    M, J, slope = derive()
    print(f"  M = {M}   ;   J = {J}")
    print(f"  M²/J = {slope}   ⇒   M² = 2πσ·J  (Regge linéaire)\n")
    print("  La linéarité M²∝J est KINÉMATIQUE (rotation relativiste), σ-INDÉPENDANTE.")
    print("  σ = facteur d'échelle (pente 1/α'=2πσ), pas dans la structure.")
    print("\n  ⇒ pente = UN nombre mesuré = 1,096 GeV²/J (r²≥0,997), pas 8πΛ_χ².")
    print("     La STRUCTURE se dérive sans σ_χ ; la pente est mesurée (comme en QCD).")
    assert slope == 2 * sp.pi * sp.Symbol("sigma", positive=True)


if __name__ == "__main__":
    main()
