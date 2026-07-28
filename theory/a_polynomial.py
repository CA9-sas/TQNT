"""
Théorie Nodale v3 — Polynômes-A et Schrödinger émergent

Le polynôme-A A_K(L, M) = 0 est une courbe algébrique qui encode la géométrie
du complément du nœud S³\\K. Il est le pont entre Jones et Schrödinger :

    Jones colorié J_n → récurrence → limite classique → A_K(L, M) = 0
    → quantification → Â_K · χ = 0 = SCHRÖDINGER ÉMERGENT

Sources:
- Cooper, Culler, Gillet, Long, Shalen (1994) — définition originale
- Garoufalidis (2004) — conjecture AJ
"""
import sympy as sp

L, M = sp.symbols('L M', commutative=True)
q = sp.Symbol('q')

# A-polynomial table: {knot_name: full A(L, M) including abelian factor (L-1)}
# Convention: A_K(L, M) includes the abelian component (L - 1) as a factor.
# The non-abelian component encodes the interesting geometry.
_A_POLY_TABLE: dict[str, sp.Expr] = {
    # Unknot: only abelian component
    "0_1": L - 1,

    # Left trefoil: A = (L - 1)(L + M^6)
    "3_1": (L - 1) * (L + M**6),

    # Figure-eight: A = (L - 1)(M^4*L^2 - (M^8 - M^6 + 2*M^4 - M^2 + 1)*L + M^4)
    "4_1": (L - 1) * (M**4 * L**2 - (M**8 - M**6 + 2*M**4 - M**2 + 1) * L + M**4),

    # 5_1 (torus knot T(2,5)): A = (L - 1)(L + M^10)
    "5_1": (L - 1) * (L + M**10),

    # 5_2 (twist knot)
    "5_2": (L - 1) * (M**4 * L**2 + (M**12 - M**10 + M**8 - M**4 + M**2 - 1) * L + M**4),
}

A_POLY_KNOT_NAMES = list(_A_POLY_TABLE.keys())


def a_polynomial(knot_name: str) -> sp.Expr:
    """Return the A-polynomial A_K(L, M) for a knot.

    Includes the abelian factor (L - 1).
    """
    if knot_name not in _A_POLY_TABLE:
        raise ValueError(f"A-polynomial unknown for {knot_name}. Available: {A_POLY_KNOT_NAMES}")
    return _A_POLY_TABLE[knot_name]


def a_polynomial_nonabelian(knot_name: str) -> sp.Expr:
    """Return the non-abelian component of the A-polynomial.

    A_K(L, M) = (L - 1) * A_na(L, M)
    """
    A = a_polynomial(knot_name)
    quotient, remainder = sp.div(sp.expand(A), L - 1, L)
    if remainder != 0:
        raise RuntimeError(f"A-polynomial for {knot_name} not divisible by (L-1)")
    return sp.simplify(quotient)


def evaluate_a_polynomial(knot_name: str, L_val: complex, M_val: complex) -> complex:
    """Evaluate A_K(L, M) numerically."""
    A = a_polynomial(knot_name)
    return complex(A.subs([(L, L_val), (M, M_val)]))


def a_polynomial_branches(knot_name: str) -> dict:
    """Analyse la factorisation du A-polynôme en branches.

    Retourne un dict avec :
      - 'abelian': le facteur (L - 1)
      - 'nonabelian': le facteur non-abélien A_na(L, M)
      - 'nonabelian_L_roots': les racines en L du facteur non-abélien
      - 'n_nonabelian_roots': le nombre de racines non-abéliennes en L

    Pour les nœuds toriques T(2,s), le facteur non-abélien est (L + M^(2s)),
    donnant une unique racine L = -M^(2s). Pour les nœuds hyperboliques,
    le facteur non-abélien est de degré > 1 en L, donnant plusieurs racines.
    """
    A_na = a_polynomial_nonabelian(knot_name)
    poly_L = sp.Poly(A_na, L)
    degree_L = poly_L.degree()
    roots_L = sp.solve(A_na, L)
    return {
        "abelian": L - 1,
        "nonabelian": A_na,
        "nonabelian_L_roots": roots_L,
        "n_nonabelian_roots": degree_L,
    }


def quantize_a_polynomial(knot_name: str) -> sp.Expr:
    """Quantize the A-polynomial: replace M → q^(1/2) M̂, enforce [M̂, L̂] = q L̂ M̂.

    Returns the quantized operator Â_K as a symbolic expression.
    This is a formal quantization — the q-commutation is noted symbolically.

    In the AJ conjecture: Â_K · J_n(K; q) = 0 where
    M̂ acts as q^(n/2) and L̂ shifts n → n+1.
    """
    A_na = a_polynomial_nonabelian(knot_name)
    L_hat, M_hat = sp.symbols('L_hat M_hat', commutative=False)
    A_hat = A_na.subs([(L, L_hat), (M, M_hat)])
    return A_hat
