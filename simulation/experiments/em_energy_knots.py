r"""
Remodèlement du concept de masse TQNT — énergie EM intégrée sur 4/3 π R³
pour chaque nœud individuel.

Intuition physique (réinitialisation conceptuelle) :
   masse(K) = énergie électromagnétique du champ généré par la boucle K
            = (ε₀/2) ∫_{ball(R)} |E(r)|² dV

où :
   - Le nœud K est traité comme une boucle filaire de charge λ (ou
     équivalent en courant via dualité électrique-magnétique).
   - Le rayon R définit la sphère d'intégration englobant K.
   - L'énergie EM AUGMENTE quand :
       * la boucle est SERRÉE (rayon de courbure petit → champ fort
         à courte distance)
       * deux brins sont PROCHES (interaction forte entre fragments)
       * la boucle approche sa rupture topologique
   - L'énergie EM DIMINUE quand la boucle est large/distendue.

Cette définition unifie :
   - Énergie de Möbius (mathématique, invariante par Möbius) :
        E_M(K) = ∮∮ [1/|γ(s)-γ(t)|² - 1/d_arc²] ds dt
   - Énergie de champ EM dans un volume (physique directe).

On calcule pour les 19 nœuds catalogue plusieurs quantités :
   1. L      : longueur d'arc totale
   2. R_max  : rayon de la boule englobante (max distance au centroïde)
   3. d_min  : distance min entre brins non-adjacents (étanchéité)
   4. κ_max  : courbure maximale locale (point le plus serré)
   5. κ_mean : courbure moyenne
   6. E_Möbius : énergie de Möbius régularisée (sans unités)
   7. U_ball : énergie EM ∫|E|²/2 sur la boule de rayon R = 1.5·R_max
   8. U_self : self-énergie coulombienne ∮∮ 1/|γ-γ| · λ² ds ds'
               (avec cutoff au rayon de tube a = d_min/4)

Toutes les quantités sont calculées sur les courbes catalogue
(theory.knot_catalog), donc dépendent partiellement de la paramétrisation
Fourier — mais la dépendance est différente de celle de U_EM_V4
(∮(κ²+τ²)ds) car les nouvelles intégrales sont non-locales.

Sortie : tableau console + LaTeX dans docs/em_energy_table.tex.
"""
from __future__ import annotations
import numpy as np

from theory.knot_catalog import knot_curve
from theory.abraham import REFERENCE_N50K

KNOTS = list(REFERENCE_N50K.keys())


# ────────────────────────────────────────────────────────────────────────
# Intégrales géométriques
# ────────────────────────────────────────────────────────────────────────
def segment_lengths(pts):
    """ds_i = |γ_{i+1} - γ_i|."""
    return np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)


def arc_length(pts):
    return float(segment_lengths(pts).sum())


def midpoints(pts):
    return 0.5 * (pts + np.roll(pts, -1, axis=0))


def curvature_arr(pts):
    """κ(s) ≈ |γ' × γ''| / |γ'|³ via différences finies."""
    N = len(pts)
    dt = 2 * np.pi / N
    p_next = np.roll(pts, -1, axis=0)
    p_prev = np.roll(pts, 1, axis=0)
    dr = (p_next - p_prev) / (2 * dt)
    ddr = (p_next - 2 * pts + p_prev) / dt ** 2
    speed = np.linalg.norm(dr, axis=1)
    cross = np.cross(dr, ddr)
    cn = np.linalg.norm(cross, axis=1)
    return cn / (speed ** 3 + 1e-30), speed


def d_min_non_adjacent(pts, skip=5):
    """Distance min entre points non-arc-adjacents."""
    N = len(pts)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    idx = np.arange(N)
    arc = np.abs(idx[:, None] - idx[None, :])
    arc = np.minimum(arc, N - arc)
    far = arc > skip
    return float(np.where(far, dist, np.inf).min())


# ────────────────────────────────────────────────────────────────────────
# Énergies EM
# ────────────────────────────────────────────────────────────────────────
def mobius_energy(pts):
    """E_Möbius(K) = ∮∮ [1/|γ(s)-γ(t)|² - 1/d_arc²] ds_s ds_t

    Régularisée à i=j et invariante par similarité/inversion.
    """
    N = len(pts)
    ds = segment_lengths(pts)
    L = ds.sum()
    diff = pts[:, None, :] - pts[None, :, :]
    d_norm = np.linalg.norm(diff, axis=-1)
    # Distance d'arc (cyclique)
    s = np.concatenate([[0.0], np.cumsum(ds[:-1])])
    arc_diff = np.abs(s[:, None] - s[None, :])
    arc_dist = np.minimum(arc_diff, L - arc_diff)
    # Régularisation : remplacer 0 par 1 pour éviter division (le masque l'efface)
    d_safe = np.where(d_norm > 1e-12, d_norm, 1.0)
    arc_safe = np.where(arc_dist > 1e-12, arc_dist, 1.0)
    integrand = 1.0 / d_safe ** 2 - 1.0 / arc_safe ** 2
    # Masque : exclure diagonale
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    weights = np.outer(ds, ds)
    return float(np.sum(integrand * weights * mask))


def coulomb_self_energy(pts, a):
    """U_self = ∮∮ 1/|γ(s)-γ(t)| ds_s ds_t   avec cutoff |γ-γ| > a.

    a est le rayon de tube (= épaisseur fictive du fil).
    Modélise l'énergie de Coulomb d'un fil chargé uniformément.
    """
    N = len(pts)
    ds = segment_lengths(pts)
    diff = pts[:, None, :] - pts[None, :, :]
    d_norm = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(d_norm, np.inf)
    # Cutoff : remplacer les distances < a par a
    d_eff = np.where(d_norm < a, a, d_norm)
    integrand = 1.0 / d_eff
    weights = np.outer(ds, ds)
    return float(np.sum(integrand * weights))


def em_field_energy_in_ball(pts, R_ball=None, n_grid=24, charge_density=1.0):
    """U_EM = (1/8π) ∫_{|r-c|<R} |E(r)|² dV

    Le nœud est traité comme un fil de charge λ = charge_density.
    On échantillonne une grille 3D de n_grid³ points dans la boule
    de rayon R_ball centrée sur le centroïde, et on intègre |E|².

    Cutoff : si r est très près d'un point du nœud (< L/N), on
    skippe (singularité du fil).
    """
    N = len(pts)
    ds = segment_lengths(pts)
    L = float(ds.sum())
    mids = midpoints(pts)
    centroid = pts.mean(axis=0)

    # Rayon de la boule
    R_max = float(np.linalg.norm(pts - centroid, axis=1).max())
    if R_ball is None:
        R_ball = 1.5 * R_max

    # Grille 3D
    xs = np.linspace(-R_ball, R_ball, n_grid)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    grid_local = np.stack([X, Y, Z], axis=-1)  # (n_g, n_g, n_g, 3)
    grid = grid_local + centroid
    r_from_c = np.linalg.norm(grid_local, axis=-1)
    inside_ball = r_from_c < R_ball

    cutoff_dist = L / N  # ~ taille d'un segment
    dV = (2 * R_ball / n_grid) ** 3

    # E(r) = Σ_n λ ds_n · (r - mid_n) / |r - mid_n|³
    # Vectorisation : pour limiter la mémoire, on traite par batches.
    # Mais pour n_grid=24, on a 24³ = 13824 points × N segments ≈ 2.7M paires.
    # Acceptable en mémoire.
    diff = grid[..., None, :] - mids[None, None, None, :, :]  # (n_g³, N, 3)
    dist3 = np.linalg.norm(diff, axis=-1)
    # Cutoff : on annule les contributions au voisinage du fil
    too_close = dist3 < cutoff_dist
    safe_dist3 = np.where(dist3 > cutoff_dist, dist3 ** 3, 1.0)
    contrib = (charge_density * ds[None, None, None, :, None]
               * diff / safe_dist3[..., None])
    contrib = np.where(too_close[..., None], 0.0, contrib)
    field = contrib.sum(axis=-2)

    E2 = np.sum(field ** 2, axis=-1)
    energy_density = E2 / 2.0  # ε₀/(8π) facteurs omis (unités naturelles)
    U = float((energy_density * inside_ball).sum() * dV)
    return U, R_ball, R_max


# ────────────────────────────────────────────────────────────────────────
# Tableau principal
# ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 110)
    print("  ÉNERGIES EM TOPOLOGIQUES — 19 NŒUDS DU CATALOGUE")
    print("=" * 110)
    print(f"\n  Hypothèse : masse(K) ∝ énergie EM du champ généré par la boucle K,")
    print(f"  intégrée sur la sphère englobante 4/3 π R³.")
    print(f"\n  4 quantités calculées :")
    print(f"    L        : longueur d'arc (paramétrisation catalogue)")
    print(f"    d_min    : distance min entre brins non-adjacents")
    print(f"    E_Möbius : ∮∮ [1/|γ-γ|² - 1/d_arc²] ds ds' (invariante)")
    print(f"    U_ball   : ∫|E|²/2 dV sur sphère 1.5·R_max (grille 24³)")
    print(f"    U_self   : ∮∮ 1/|γ-γ| ds ds' avec cutoff a = d_min/2 (Coulomb)")

    rows = []
    print(f"\n  Calcul (N=200 points par courbe, normalisation L = 100)…")
    print(f"  (Renormalisation par L = 100 pour comparaison équitable :")
    print(f"   l'énergie de Möbius EST invariante d'échelle, mais U_ball et")
    print(f"   U_self dépendent de l'échelle ; on les met sur même pied.)")
    for K in KNOTS:
        pts = knot_curve(K, 201)[:-1]
        # Normalisation par longueur d'arc : L → 100
        L_raw = arc_length(pts)
        scale = 100.0 / L_raw
        pts_norm = pts * scale
        # Quantités sur la version normalisée
        L = arc_length(pts_norm)
        kap, _ = curvature_arr(pts_norm)
        dmin = d_min_non_adjacent(pts_norm)
        # Cutoff réaliste = max(d_min/3, L/200)
        a_cutoff = max(dmin / 3, L / 200)
        E_M = mobius_energy(pts_norm)
        U_self = coulomb_self_energy(pts_norm, a=a_cutoff)
        U_ball, R_ball, R_max = em_field_energy_in_ball(pts_norm, n_grid=20)
        rows.append({
            "K": K, "L": L, "R_max": R_max, "d_min": dmin,
            "k_max": float(kap.max()), "k_mean": float(kap.mean()),
            "E_M": E_M, "U_ball": U_ball, "U_self": U_self,
            "L_raw": L_raw,
        })
        flag = "★" if dmin < 0.05 else " "
        print(f"    {K}: L={L:.1f}, R_max={R_max:.2f}, d_min={dmin:.4f}{flag}, "
              f"E_M={E_M:.2f}, U_ball={U_ball:.1f}, U_self={U_self:.1f}")

    # ─── Tableau formaté ──────────────────────────────────────────────
    print(f"\n  TABLEAU FINAL (toutes courbes normalisées à d_min = 1) :")
    print(f"  {'K':>5} {'L':>8} {'R_max':>7} {'κ_max':>8} {'κ_moy':>8} "
          f"{'E_Möbius':>10} {'U_ball':>10} {'U_self':>10}")
    print("  " + "-" * 78)
    for r in rows:
        print(f"  {r['K']:>5} {r['L']:>8.2f} {r['R_max']:>7.2f} "
              f"{r['k_max']:>8.2f} {r['k_mean']:>8.2f} {r['E_M']:>10.2f} "
              f"{r['U_ball']:>10.2f} {r['U_self']:>10.2f}")

    # ─── Stats et observations ────────────────────────────────────────
    arr_EM = np.array([r["E_M"] for r in rows])
    arr_U = np.array([r["U_ball"] for r in rows])
    arr_self = np.array([r["U_self"] for r in rows])
    print(f"\n  STATS sur 19 nœuds :")
    print(f"    E_Möbius  : min = {arr_EM.min():.2f}, max = {arr_EM.max():.2f}, "
          f"médian = {np.median(arr_EM):.2f}")
    print(f"    U_ball    : min = {arr_U.min():.2f}, max = {arr_U.max():.2f}, "
          f"médian = {np.median(arr_U):.2f}")
    print(f"    U_self    : min = {arr_self.min():.2f}, max = {arr_self.max():.2f}, "
          f"médian = {np.median(arr_self):.2f}")

    # Corrélations entre énergies et avec cr
    cr_arr = np.array([int(r["K"].split("_")[0]) for r in rows])
    for name, arr in [("E_Möbius", arr_EM), ("U_ball", arr_U), ("U_self", arr_self)]:
        rho_cr = np.corrcoef(arr, cr_arr)[0, 1]
        print(f"    Corrélation Pearson({name}, cr) = {rho_cr:+.3f}")
    r_EM_U = np.corrcoef(arr_EM, arr_U)[0, 1]
    r_EM_self = np.corrcoef(arr_EM, arr_self)[0, 1]
    print(f"    Corrélation Pearson(E_Möbius, U_ball) = {r_EM_U:+.3f}")
    print(f"    Corrélation Pearson(E_Möbius, U_self) = {r_EM_self:+.3f}")

    # Lien avec rope-length publiée
    PUBLISHED_ROPE = {
        "3_1": 16.37, "4_1": 20.99, "5_1": 23.55, "5_2": 26.13,
        "6_1": 28.31, "6_2": 28.13, "6_3": 27.21,
        "7_1": 30.69, "7_2": 32.96, "7_3": 33.61, "7_4": 34.97,
        "7_5": 35.16, "7_6": 32.66, "7_7": 31.93,
        "8_1": 37.49, "8_2": 37.32, "8_3": 37.30,
        "9_1": 38.62, "10_1": 42.33,
    }
    rope_arr = np.array([PUBLISHED_ROPE[r["K"]] for r in rows])
    for name, arr in [("E_Möbius", arr_EM), ("U_ball", arr_U), ("U_self", arr_self)]:
        rho = np.corrcoef(arr, rope_arr)[0, 1]
        print(f"    Corrélation Pearson({name}, rope-length pub.) = {rho:+.3f}")

    # ─── Export LaTeX ─────────────────────────────────────────────────
    tex_path = "docs/em_energy_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"% Tableau des énergies EM topologiques sur 19 nœuds" + "\n")
        f.write(r"% Généré par simulation/experiments/em_energy_knots.py" + "\n")
        f.write(r"\begin{center}" + "\n")
        f.write(r"\renewcommand{\arraystretch}{1.15}" + "\n")
        f.write(r"\begin{tabular}{lrrrrrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"$K$ & $L$ & $R_{\max}$ & $\kappa_{\max}$ & "
                r"$\kappa_{\mathrm{moy}}$ & $E_{\mathrm{M\"ob}}$ & "
                r"$U_{\mathrm{ball}}$ & $U_{\mathrm{self}}$ \\" + "\n")
        f.write(r"\midrule" + "\n")
        for r in rows:
            knot_tex = r["K"].replace("_", r"_")
            f.write(f"  ${knot_tex}$ & {r['L']:.2f} & {r['R_max']:.2f} & "
                    f"{r['k_max']:.2f} & {r['k_mean']:.2f} & "
                    f"{r['E_M']:.2f} & {r['U_ball']:.2f} & "
                    f"{r['U_self']:.2f} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{center}" + "\n")
    print(f"\n  ✓ Tableau LaTeX exporté dans : {tex_path}")
    print("=" * 110)
    return rows


if __name__ == "__main__":
    main()
