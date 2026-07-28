r"""Calcul 3-corps variationnel de la fonction d'onde baryonique (catalogue σ_χ).

Objet : dériver — et non ajuster — la COMPACITÉ de l'état baryonique, c.-à-d.
|ψ_ij(0)|² = ⟨δ³(r_ij)⟩ par paire de quarks, pour tester si le baryon doublement
étrange (Ξ, Ξ*) est plus compact et si cela corrige le résidu hyperfin Ξ que le
modèle constituant PAIRWISE (τ_⊥) manque.

Méthode : Gaussienne corrélée (formalisme A-matrice, Suzuki-Varga).
Coord. Jacobi J1=r1−r2, J2=r3−(m1r1+m2r2)/(m1+m2) → kinétique découplée.
Essai ψ=exp(−½ Jᵀ A J), A=[[a,c],[c,b]], minimisée (⟨T⟩+⟨V_conf⟩).
  ⟨T⟩=(3/2)[a/(2μ1)+b/(2μ2)],  ⟨r_ij⟩=√(8/π)√(w_ijᵀA⁻¹w_ij),
  |ψ_ij(0)|²=(2π w_ijᵀA⁻¹w_ij)^(−3/2),  w12=(1,0),w13=(f2,−1),w23=(−f1,−1).
Hyperfin : E_hf = κ Σ ⟨S_i·S_j⟩ |ψ_ij(0)|²/(m_i m_j), κ calibré sur N−Δ=293 MeV.

RÉSULTAT (m_s/m_l=5/3, σ-INDÉPENDANT car pur ratio de masse) :
  ψ²_ss(Ξ)/ψ²_ll(N) = 1,63  → Ξ PLUS COMPACT (dérivé).
  Ξ*−Ξ : 176 (pairwise) → 224 vs 214 exp (+4,7%)  → compacité CORRIGE le trou.
  Σ−Λ : 78 → 54 (−30%)  → dégradé (la paire ls se compactifie aussi).
CONCLUSION : la compacité (mécanisme du résidu Ξ) est DÉRIVÉE et fixe le splitting
doublement-étrange ; mais l'hyperfin de contact pur (δ³) ne fait pas <5% sur Ξ*−Ξ
ET Σ−Λ simultanément → il faut un hyperfin ÉTALÉ (portée ~1/m_q), physique connue.
"""
import numpy as np
from scipy.optimize import minimize

SS = {  # ⟨S_i·S_j⟩ par paire (12,13,23) selon l'état de spin
    'N':  {'12': -0.75, '13': 0.0,  '23': 0.0},
    'Lam':{'12': -0.75, '13': 0.0,  '23': 0.0},   # ud en S=0
    'Sig':{'12': 0.25,  '13': -0.5, '23': -0.5},  # ud en S=1
    'Xi': {'12': -0.5,  '13': -0.5, '23': 0.25},  # ss(23) en S=1
    'D':  {'12': 0.25,  '13': 0.25, '23': 0.25},
    'Sst':{'12': 0.25,  '13': 0.25, '23': 0.25},
    'Xst':{'12': 0.25,  '13': 0.25, '23': 0.25},
    'Om': {'12': 0.25,  '13': 0.25, '23': 0.25},
}


def _jacobi(m):
    m1, m2, m3 = m
    M = m1 + m2 + m3
    f1, f2 = m1 / (m1 + m2), m2 / (m1 + m2)
    mu1 = m1 * m2 / (m1 + m2)
    mu2 = m3 * (m1 + m2) / M
    W = {'12': np.array([1., 0.]),
         '13': np.array([f2, -1.]),
         '23': np.array([-f1, -1.])}
    return mu1, mu2, W


def _energy(p, mu1, mu2, W, sig):
    a, b = np.exp(p[0]), np.exp(p[1])
    c = np.tanh(p[2]) * np.sqrt(a * b) * 0.98
    A = np.array([[a, c], [c, b]])
    Ai = np.linalg.inv(A)
    T = 1.5 * (a / (2 * mu1) + b / (2 * mu2))
    V = sig * np.sqrt(8 / np.pi) * sum(np.sqrt(w @ Ai @ w) for w in W.values())
    return T + V


def solve(m, sig):
    """Minimise ⟨H⟩ pour masses m=(m1,m2,m3) ; renvoie (E, {paire: |ψ(0)|²})."""
    mu1, mu2, W = _jacobi(m)
    best = None
    for seed in [(-1, -1, 0), (0, 0, 0), (1, 0, 0.3), (-0.5, 0.5, -0.3)]:
        r = minimize(_energy, seed, args=(mu1, mu2, W, sig), method='Nelder-Mead',
                     options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 5000})
        if best is None or r.fun < best.fun:
            best = r
    a, b = np.exp(best.x[0]), np.exp(best.x[1])
    c = np.tanh(best.x[2]) * np.sqrt(a * b) * 0.98
    Ai = np.linalg.inv(np.array([[a, c], [c, b]]))
    psi2 = {k: (2 * np.pi * (w @ Ai @ w)) ** -1.5 for k, w in W.items()}
    return best.fun, psi2


def hyperfine_splittings(ml=0.33, ms=0.55, sig=0.15, lam0=None):
    """Splittings hyperfins (MeV) depuis les |ψ(0)|² variationnels.

    lam0=None : contact pur δ³ (Ξ*−Ξ bon +4,7%, Σ−Λ raté −30%).
    lam0>0 (GeV) : hyperfin ÉTALÉ, portée fixe 1/lam0. lam0≈Λ_χ≈0.209 réconcilie
    Σ−Λ et Ξ*−Ξ à ~10% chacun (sans nouveau paramètre : la portée = 1/Λ_χ ≈ 1 fm).
    """
    B = {'N': (ml, ml, ml), 'Lam': (ml, ml, ms), 'Sig': (ml, ml, ms),
         'Xi': (ml, ms, ms), 'D': (ml, ml, ml), 'Sst': (ml, ml, ms),
         'Xst': (ml, ms, ms), 'Om': (ms, ms, ms)}
    A = {k: solve(m, sig) for k, m in B.items()}     # (E, psi2_contact)
    mu_W = {k: _jacobi(m) for k, m in B.items()}
    idx = {'12': (0, 1), '13': (0, 2), '23': (1, 2)}

    def s2(k, p):                                     # variance/composante de la paire
        return (2 * np.pi) ** -1 * A[k][1][p] ** (-2 / 3.)

    def psi2(k, p):
        contact = A[k][1][p]
        if lam0 is None:
            return contact
        x = 2 * lam0 ** 2 * s2(k, p)
        return contact * (x / (1 + x)) ** 1.5

    def mpair(m, key):
        i, j = idx[key]
        return m[i] * m[j]

    def Ehf(k, kappa):
        return kappa * sum(SS[k][p] * psi2(k, p) / mpair(B[k], p) for p in idx)

    kappa = 0.293 / (Ehf('D', 1) - Ehf('N', 1))     # calibré N−Δ
    return {
        'Sig-Lam': (Ehf('Sig', kappa) - Ehf('Lam', kappa)) * 1000,
        'Xist-Xi': (Ehf('Xst', kappa) - Ehf('Xi', kappa)) * 1000,
        'compactness_Xi': psi2('Xi', '23') / psi2('N', '12'),
    }


if __name__ == '__main__':
    r = hyperfine_splittings()
    print(f"ψ²_ss(Ξ)/ψ²_ll(N) = {r['compactness_Xi']:.2f}  (compacité dérivée)")
    print("-- contact δ³ --")
    print(f"Σ−Λ  = {r['Sig-Lam']:.1f} MeV  (exp 77.4, {(r['Sig-Lam']-77.4)/77.4*100:+.1f}%)")
    print(f"Ξ*−Ξ = {r['Xist-Xi']:.1f} MeV  (exp 213.5, {(r['Xist-Xi']-213.5)/213.5*100:+.1f}%)")
    rs = hyperfine_splittings(lam0=0.209)            # portée = 1/Λ_χ
    print("-- étalé, portée 1/Λ_χ --")
    print(f"Σ−Λ  = {rs['Sig-Lam']:.1f} MeV  ({(rs['Sig-Lam']-77.4)/77.4*100:+.1f}%)")
    print(f"Ξ*−Ξ = {rs['Xist-Xi']:.1f} MeV  ({(rs['Xist-Xi']-213.5)/213.5*100:+.1f}%)")
