r"""
Chantier 1 → D3 — origine algébrique du facteur d'écrantage f(n).

Valeurs empiriques V4 (theory/mass_formula.py) :
  f(1) = 1.00    (singles, électron, muon...)
  f(2) = 0.86    (mésons : 2 composantes)
  f(3) = 0.77    (baryons : 3 composantes, p, n)

Hypothèse : f(n) provient d'un mécanisme topologique du confinement
de Wilson loop dans Chern-Simons SU(2)_k. On teste un éventail de
formes algébriques candidates et on retient celles qui reproduisent
f(2)=0.86 et f(3)=0.77 SIMULTANÉMENT avec 0-1 paramètre.

Familles testées :
  A. Lois de puissance  : f(n) = β^(n-1)
  B. q-integers SU(2)_k : [n]_q / [2]_q^(n-1), etc., balayage sur k
  C. Dimensions quantiques : d_j(k) / d_n(k)
  D. Ratios simples : (n+a)/(n+b), exp(-α·H_n), cos(θ_n)
  E. HOMFLY/Kauffman bracket : facteurs (a+a^-1)^(n-1)

On classe par RMS d'erreur sur les 2 points (f(2), f(3)) et on prédit
f(4), f(5) pour comparaison croisée avec d'éventuels entrelacs à
4-5 composantes (atomes nucléaires liés, états exotiques).
"""
from __future__ import annotations
import numpy as np

F_EMPIRICAL = {1: 1.00, 2: 0.86, 3: 0.77}


def rms_error(predict_fn, params=()):
    err = [(predict_fn(n, *params) - F_EMPIRICAL[n]) ** 2 for n in (1, 2, 3)]
    return float(np.sqrt(sum(err) / 3))


def fit_1param(predict_fn, theta_range):
    """Recherche 1-paramètre sur grille."""
    best = (None, np.inf)
    for theta in theta_range:
        e = rms_error(lambda n: predict_fn(n, theta))
        if e < best[1]:
            best = (theta, e)
    return best


def fit_2params(predict_fn, t1_range, t2_range):
    best = (None, None, np.inf)
    for t1 in t1_range:
        for t2 in t2_range:
            e = rms_error(lambda n: predict_fn(n, t1, t2))
            if e < best[2]:
                best = (t1, t2, e)
    return best


def section(title):
    print(f"\n  {'═' * 80}")
    print(f"  ▶ {title}")
    print(f"  {'═' * 80}")


def report(name, pred, params=(), extra=""):
    f1 = float(np.real(pred(1, *params)))
    f2 = float(np.real(pred(2, *params)))
    f3 = float(np.real(pred(3, *params)))
    f4 = float(np.real(pred(4, *params)))
    f5 = float(np.real(pred(5, *params)))
    e2 = (f2 - 0.86) / 0.86 * 100
    e3 = (f3 - 0.77) / 0.77 * 100
    err = rms_error(lambda n: float(np.real(pred(n, *params))))
    flag = "★" if err < 0.005 else ("✓" if err < 0.02 else " ")
    print(f"  {flag} {name:<48} f(1)={f1:.3f}  f(2)={f2:.3f} "
          f"({e2:+.1f}%)  f(3)={f3:.3f} ({e3:+.1f}%)  RMS={err:.4f}")
    if -1 < f4 < 2 and -1 < f5 < 2:
        print(f"     prédictions :  f(4)={f4:.3f}  f(5)={f5:.3f}  {extra}")


def main():
    print("=" * 96)
    print("  ORIGINE ALGÉBRIQUE DU FACTEUR f(n) — survey de formes candidates")
    print("=" * 96)
    print(f"  Cible empirique : f(1)=1.00, f(2)=0.86, f(3)=0.77")

    # ═══════════════════════════════════════════════════════════════════════
    section("A. Lois de puissance simples")

    # A1 : f(n) = β^(n-1)
    def A1(n, beta):
        return float(beta) ** (n - 1)
    beta, err = fit_1param(A1, np.linspace(0.7, 1.0, 301))
    report(f"A1: β^(n-1), β fit = {beta:.4f}", A1, (beta,),
           extra=f"(géométrique pur)")

    # A2 : f(n) = 1 - α(n-1)
    def A2(n, alpha):
        return max(1 - alpha * (n - 1), 0.0)
    alpha, err = fit_1param(A2, np.linspace(0.05, 0.20, 301))
    report(f"A2: 1-α(n-1), α fit = {alpha:.4f}", A2, (alpha,))

    # A3 : f(n) = 1 - α(n-1)^p (2 params)
    def A3(n, alpha, p):
        return max(1 - alpha * (n - 1) ** p, 0.0)
    alpha, p, err = fit_2params(A3, np.linspace(0.05, 0.20, 50),
                                np.linspace(0.5, 2.5, 50))
    report(f"A3: 1-α(n-1)^p, α={alpha:.3f}, p={p:.2f}", A3, (alpha, p))

    # ═══════════════════════════════════════════════════════════════════════
    section("B. q-integers Chern-Simons SU(2)_k")

    # B1 : f(n) = [n]_q / n, balayage k
    def cs_q_real(k):
        # On utilise q réel (|q|=1 dans CS, mais on teste les deux)
        return np.exp(2 * np.pi / (k + 2))  # forme "réelle"
    def cs_q_complex(k):
        return np.exp(2j * np.pi / (k + 2))
    def q_int(n, q):
        if abs(q - 1) < 1e-12:
            return float(n)
        return float(np.real((q ** (n / 2) - q ** (-n / 2))
                             / (q ** 0.5 - q ** (-0.5))))

    # B1 : [n]_q / n, q = exp(2πi/(k+2))
    def B1(n, k):
        q = cs_q_complex(k)
        return float(np.real(q_int(n, q) / n)) if n > 0 else 1.0
    k_grid = list(range(2, 50))
    best_k, err = None, np.inf
    for k in k_grid:
        e = rms_error(lambda n: B1(n, k))
        if e < err:
            best_k, err = k, e
    report(f"B1: Re([n]_q / n), q=e^(2πi/(k+2)), k={best_k}", B1, (best_k,))

    # B2 : 2·cos(π/(k+2))^(n-1) — quantum dimension d_{1/2} à la puissance n-1
    def B2(n, k):
        return (2 * np.cos(np.pi / (k + 2)) / 2) ** (n - 1)
    best_k, err = None, np.inf
    for k in range(2, 200):
        e = rms_error(lambda n: B2(n, k))
        if e < err:
            best_k, err = k, e
    report(f"B2: cos(π/(k+2))^(n-1), k fit = {best_k}", B2, (best_k,))

    # B3 : Re([2]_q^(n-1)) avec q = exp(2πi/(k+2)), k = 135 (V4 standard)
    def B3(n, k):
        q = cs_q_complex(k)
        d_half = float(np.real(q + q ** -1)) / 2  # 2·cos(2π/(k+2))/2
        return d_half ** (n - 1)
    report(f"B3: ((q+q^-1)/2)^(n-1), k = 135 (V4)", B3, (135,))

    # ═══════════════════════════════════════════════════════════════════════
    section("C. Ratios algébriques simples")

    # C1 : (a-n)/(a) — fit a
    def C1(n, a):
        return max((a - n + 1) / a, 0.0)
    a, err = fit_1param(C1, np.linspace(2, 20, 1000))
    report(f"C1: (a-n+1)/a, a fit = {a:.3f}", C1, (a,))

    # C2 : 1 / (1 + α·(n-1))
    def C2(n, alpha):
        return 1 / (1 + alpha * (n - 1))
    alpha, err = fit_1param(C2, np.linspace(0.05, 0.5, 301))
    report(f"C2: 1/(1+α(n-1)), α fit = {alpha:.4f}", C2, (alpha,))

    # C3 : exp(-α(n-1)^p)
    def C3(n, alpha, p):
        return float(np.exp(-alpha * (n - 1) ** p))
    alpha, p, err = fit_2params(C3, np.linspace(0.05, 0.3, 50),
                                np.linspace(0.5, 2.0, 50))
    report(f"C3: exp(-α(n-1)^p), α={alpha:.3f}, p={p:.2f}", C3, (alpha, p))

    # C4 : cos(θ(n-1)) — phase géométrique
    def C4(n, theta):
        return float(np.cos(theta * (n - 1)))
    theta, err = fit_1param(C4, np.linspace(0.05, 1.0, 301))
    report(f"C4: cos(θ(n-1)), θ fit = {theta:.4f} rad = {theta*180/np.pi:.2f}°",
           C4, (theta,))

    # C5 : exp(-α·H_n) où H_n = 1+1/2+...+1/n (nombre harmonique)
    H = {1: 1.0, 2: 1.5, 3: 11/6, 4: 25/12, 5: 137/60}
    def C5(n, alpha):
        return float(np.exp(-alpha * H[n]))
    alpha, err = fit_1param(C5, np.linspace(0.05, 1.0, 301))
    report(f"C5: exp(-α·H_n), α fit = {alpha:.4f}  (H_n: 1, 3/2, 11/6, ...)",
           C5, (alpha,))

    # ═══════════════════════════════════════════════════════════════════════
    section("D. Constantes physiques fondamentales (sans fit)")

    # D1 : α_EM = 1/137
    def D1(n):
        return (1 - 1/137) ** (n - 1)
    report(f"D1: (1 - α_EM)^(n-1) = (136/137)^(n-1)",
           lambda n: D1(n))

    # D2 : 1/φ (nombre d'or) = 0.618
    def D2(n):
        return 0.618 ** (n - 1) / (n - 0.5) ** 0 if n > 1 else 1.0
    report(f"D2: 1/φ^(n-1), φ = 1.618", lambda n: D2(n))

    # D3 : (1 - 1/e)^(n-1)
    def D3(n):
        return (1 - 1/np.e) ** (n - 1)
    report(f"D3: (1 - 1/e)^(n-1)", lambda n: D3(n))

    # D4 : 1 - log(n)/log(N) pour N à fit
    def D4(n, N):
        if n == 1:
            return 1.0
        return 1 - np.log(n) / np.log(N)
    N, err = fit_1param(D4, np.linspace(10, 1000, 1000))
    report(f"D4: 1 - log(n)/log(N), N fit = {N:.1f}", D4, (N,))

    # ═══════════════════════════════════════════════════════════════════════
    section("E. Wilson loops / HOMFLY")

    # E1 : facteur de braid : sin((n+1)π/(k+2))/sin(π/(k+2))
    def E1(n, k):
        return float(np.sin((n + 1) * np.pi / (k + 2))
                     / np.sin(np.pi / (k + 2)) / (n + 1))
    best_k, err = None, np.inf
    for k in range(2, 100):
        e = rms_error(lambda n: E1(n, k))
        if e < err:
            best_k, err = k, e
    report(f"E1: sin((n+1)π/(k+2))/((n+1)·sin(π/(k+2))), k = {best_k}",
           E1, (best_k,))

    # E2 : (n+1)^(-α)
    def E2(n, alpha):
        return (n + 1) ** (-alpha) * 2 ** alpha
    alpha, err = fit_1param(E2, np.linspace(0.1, 1.0, 301))
    report(f"E2: ((n+1)/2)^(-α), α fit = {alpha:.4f}", E2, (alpha,))

    # E3 : (1 + a·(n-1))^p
    def E3(n, a, p):
        return (1 + a * (n - 1)) ** p
    a, p, err = fit_2params(E3, np.linspace(-0.5, 0.5, 50),
                            np.linspace(-3, 3, 50))
    report(f"E3: (1+a(n-1))^p, a={a:.3f}, p={p:.2f}", E3, (a, p))

    # ═══════════════════════════════════════════════════════════════════════
    section("F. Quantique : SU(2)_k spin-1 colored Wilson, k=2,...,200")

    # F1 : ratio quantum dim d_n/d_1 normalisé
    def F1(n, k):
        q = cs_q_complex(k)
        d_n = float(np.real(q_int(n, q)))
        d_1 = float(np.real(q_int(1, q)))
        if d_1 == 0:
            return 0.0
        return (d_n / d_1) ** (1 / (n - 1) if n > 1 else 1) if n > 1 else 1.0
    best_k, err = None, np.inf
    for k in range(2, 200):
        e = rms_error(lambda n: F1(n, k))
        if e < err:
            best_k, err = k, e
    report(f"F1: ([n]_q/n)^(1/(n-1)), k = {best_k}", F1, (best_k,))

    # F2 : un mécanisme géométrique direct : f(n) = (1 - α_s·H_{n-1})
    # avec H_{n-1} nombre harmonique, α_s ≈ α_QCD à courte distance
    def F2(n, alpha_s):
        H_nm1 = sum(1.0 / k for k in range(1, n))
        return 1 - alpha_s * H_nm1
    alpha, err = fit_1param(F2, np.linspace(0.05, 0.5, 301))
    report(f"F2: 1 - α_s · H_{{n-1}}, α fit = {alpha:.4f}  "
           f"(α_s à courte dist?)", F2, (alpha,))

    # ═══════════════════════════════════════════════════════════════════════
    section("G. Forme inspirée du χ field (TQNT-spécifique)")

    # G1 : f(n) = 1 / (1 + γ(n-1)) où γ pourrait être le coefficient GAMMA = 0.10
    # de l'éq. c_J = span + γ·det
    def G1(n, gamma):
        return 1 / (1 + gamma * (n - 1))
    gamma, err = fit_1param(G1, np.linspace(0.05, 0.5, 301))
    report(f"G1: 1/(1+γ(n-1)), γ fit = {gamma:.4f}", G1, (gamma,))
    print(f"     Note : γ_V4 = 0.10 (theory/mass_formula.py). Test direct :")
    f_G1_010 = [G1(n, 0.10) for n in (1, 2, 3)]
    print(f"     f(γ=0.10) = {f_G1_010[0]:.3f}, {f_G1_010[1]:.3f}, "
          f"{f_G1_010[2]:.3f}  (vs cible 1.00, 0.86, 0.77)")

    # G2 : f(n) = (1 + γ)^(1-n)  — équivalent à puissance de 1/(1+γ)
    def G2(n, gamma):
        return (1 + gamma) ** (1 - n)
    gamma, err = fit_1param(G2, np.linspace(0.05, 0.5, 301))
    report(f"G2: (1+γ)^(1-n), γ fit = {gamma:.4f}", G2, (gamma,))

    # G3 : f(n) = exp(-γ(n-1)) avec γ à fit
    def G3(n, gamma):
        return float(np.exp(-gamma * (n - 1)))
    gamma, err = fit_1param(G3, np.linspace(0.05, 0.5, 301))
    report(f"G3: exp(-γ(n-1)), γ fit = {gamma:.4f}  "
           f"(comparable α de C5 ?)", G3, (gamma,))

    # ═══════════════════════════════════════════════════════════════════════
    section("H. FORMES « RATIONNELLES » sans aucun paramètre libre")

    # H1 : f(n) = ((n+1)/2)^(-3/8) — découverte du fit E2 avec α=0.375=3/8
    def H1(n):
        return ((n + 1) / 2) ** (-3 / 8)
    report(f"H1: ((n+1)/2)^(-3/8)  ← exposant EXACT 3/8 = h_1 WZW SU(2)_2 ?",
           lambda n: H1(n))
    print(f"     log f(n) / log((n+1)/2) :")
    for n in (2, 3, 4):
        f_emp = F_EMPIRICAL.get(n, None)
        if f_emp:
            ratio = np.log(f_emp) / np.log((n + 1) / 2)
            print(f"        n={n}: log({f_emp})/log({(n+1)/2:.2f}) = {ratio:+.4f}")

    # H2 : cos(π/6)^(n-1) — la forme CS k=4 trouvée en B2
    def H2(n):
        return (np.sqrt(3) / 2) ** (n - 1)
    report(f"H2: (√3/2)^(n-1) = cos(π/6)^(n-1)  ← CS SU(2)_4 ?",
           lambda n: H2(n))

    # H3 : (n+1)^(-α) avec α = α_EM·a_quelque_chose
    # Test α = log(0.86)/log(3/2) ≈ -0.372
    alpha = -np.log(0.86) / np.log(1.5)
    print(f"     Calcul direct α via f(2) seul : α = {alpha:.5f}")
    print(f"     Comparer à 3/8 = {3/8:.5f}, écart = {abs(alpha-3/8)*100/(3/8):.2f}%")

    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n  {'═' * 80}")
    print(f"  ▶ DIAGNOSTIC")
    print(f"  {'═' * 80}")
    print(f"  ★ = RMS < 0.005 (excellent)   ✓ = RMS < 0.02 (acceptable)")
    print(f"  ")
    print(f"  Hypothèse à privilégier : la forme la plus simple avec le moins")
    print(f"  de paramètres et RMS < 0.005 sur les 2 points (f(2), f(3)).")
    print(f"  ")
    print(f"  Si une forme à 0 ou 1 paramètre PHYSIQUE (constante connue)")
    print(f"  marche, c'est probablement le bon mécanisme.")
    print(f"  ")
    print(f"  Si SEUL un fit 2-paramètres marche, f(n) est probablement une")
    print(f"  paramétrisation phénoménologique sans origine fondamentale —")
    print(f"  ou alors une combinaison de plusieurs effets (gluonique +")
    print(f"  topologique + cinématique).")
    print("=" * 96)


if __name__ == "__main__":
    main()
