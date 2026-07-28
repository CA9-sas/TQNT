r"""
Q affinés (physique des produits dominants) + sous-classification du canal faible.

Le Q pertinent dépend du TYPE de désintégration faible :
  - β 3-corps leptonique (n, μ, τ) : Q = énergie libérée ; espace des phases Q^5 (Sargent).
  - 2-corps leptonique (π±, K± → μν) : Q = m − m_μ ; phase space ∝ Q (pas Q^5).
  - semi-leptonique quark lourd (D, Λc, Ωc ; B, Λb) : Q ≈ masse du quark (m_c, m_b).
  - non-leptonique (Λ, Σ, Ξ, Ω, K0_S) : produits noués, quasi Q-indépendant.

But : montrer honnêtement que même avec de bons Q, le canal faible garde une
structure (2 vs 3 corps, suppression CKM charm/bottom) qu'un simple Q^-ν ne
capture pas — c'est de la phénoménologie faible réelle, à origine topologique
dans la TQNT (état final ondes libres vs noué ; CKM = changement de génération).

Usage :  python3 -m simulation.experiments.lifetime_Q_refined
"""
from __future__ import annotations

import numpy as np

# (nom, Q affiné MeV, τ_exp s, type)
M_C, M_B = 1270.0, 4180.0
DATA = [
    ("n",       0.78,    878.4,     "beta3"),    # n→peν, 3-corps
    ("mu-",     105.66,  2.197e-6,  "beta3"),    # μ→eνν, 3-corps
    ("tau-",    1776.9,  2.903e-13, "beta3"),    # τ→ντ+X, ~3-corps
    ("pi+-",    33.91,   2.603e-8,  "lep2"),     # π→μν, 2-corps
    ("K+-",     388.0,   1.238e-8,  "lep2"),     # K→μν, 2-corps
    ("D0",      M_C,     4.10e-13,  "charm"),
    ("D+",      M_C,     1.033e-12, "charm"),
    ("Lambdac", M_C,     2.03e-13,  "charm"),
    ("Omegac",  M_C,     2.68e-13,  "charm"),
    ("B",       M_B,     1.519e-12, "bottom"),
    ("Lambdab", M_B,     1.471e-12, "bottom"),
    ("Lambda",  37.84,   2.617e-10, "nonlep"),
    ("Sigma+",  113.0,   8.018e-11, "nonlep"),
    ("Sigma-",  118.31,  1.479e-10, "nonlep"),
    ("Xi0",     64.20,   2.90e-10,  "nonlep"),
    ("Xi-",     66.46,   1.639e-10, "nonlep"),
    ("Omega-",  63.0,    8.21e-11,  "nonlep"),
    ("K0_S",    218.5,   8.954e-11, "nonlep"),
]


def fit_pow(items):
    """τ ∝ Q^-ν ; renvoie ν, RMS, corr."""
    lQ = np.log10([q for _, q, _, _ in items])
    lt = np.log10([t for _, _, t, _ in items])
    A = np.column_stack([np.ones(len(items)), -lQ])
    c, *_ = np.linalg.lstsq(A, lt, rcond=None)
    rms = float(np.sqrt(np.mean((lt - A @ c) ** 2)))
    cr = float(np.corrcoef(lQ, lt)[0, 1]) if len(items) > 2 else float("nan")
    return c[1], rms, cr


def main():
    print("=" * 88)
    print("  Q AFFINÉS + SOUS-STRUCTURE DU CANAL FAIBLE")
    print("=" * 88)

    groups = {
        "beta3  (3-corps leptonique, Sargent)": "beta3",
        "lep2   (2-corps leptonique π,K→μν)":   "lep2",
        "charm  (semi-lept. c→s)":              "charm",
        "bottom (semi-lept. b→c, CKM)":         "bottom",
        "nonlep (hadronique, hypérons)":        "nonlep",
    }
    for lab, key in groups.items():
        items = [d for d in DATA if d[3] == key]
        print(f"\n  --- {lab} ({len(items)} part.) ---")
        for nm, q, t, _ in sorted(items, key=lambda z: z[1]):
            print(f"      {nm:>8} Q={q:>7.1f}  τ={t:.2e}")
        if len(items) >= 2:
            nu, rms, cr = fit_pow(items)
            extra = f", corr={cr:+.2f}" if not np.isnan(cr) else ""
            print(f"      -> τ∝Q^-ν : ν={nu:.2f}, RMS={rms:.3f} déc{extra}")

    print("\n  Lecture :")
    print("  • non-leptonique : quasi Q-indépendant (reconnexion topologique).")
    print("  • beta3 leptonique : suit Sargent (Q^5) — n, μ, τ sur ~16 ordres.")
    print("  • 2-corps (π,K) : loi différente (phase space ∝ Q, pas Q^5).")
    print("  • charm vs bottom : bottom Q PLUS grand mais τ PLUS long => CKM (b→c")
    print("    supprimé) ; Q seul ne suffit pas, il faut la structure de génération.")

    # Sargent strict sur beta3 : exposant attendu 5 ?
    b3 = [d for d in DATA if d[3] == "beta3"]
    nu, rms, cr = fit_pow(b3)
    print(f"\n  Test Sargent strict (beta3 n,μ,τ) : ν={nu:.2f} (attendu ~5), "
          f"RMS={rms:.3f} déc, corr={cr:+.3f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
