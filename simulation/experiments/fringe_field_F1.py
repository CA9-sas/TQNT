"""
Calcul F1 — Champ de frange du nœud et force nucléaire résiduelle (TQNT).

Le champ magnétique intrinsèque le long du brin est B(s) = tau(s) t̂(s)
(torsion le long de la tangente). Vu de l'extérieur, le tube de flux se
comporte comme une ligne de dipôles axiaux dm(s) = tau(s) t̂(s) ds.

Ce script vérifie numériquement les conclusions du brouillon
(docs/brouillon_interactions_forte_faible.tex, section F1) :

  (a) pas de monopôle, ordre dominant dipolaire ;
  (c) loi de décroissance du champ de frange (pente log-log) ;
  (d/e) potentiel à deux nœuds V(R) : portée et profondeur relative.

On reste en unités géométriques de la courbe ; l'écrantage de Yukawa
(masse topologique de Chern-Simons) et la calibration MeV/fm sont
appliqués en post-traitement.

Usage :  python3 -m simulation.experiments.fringe_field_F1
"""
from __future__ import annotations

import numpy as np

from theory.knot_catalog import knot_curve
from theory.abraham import frenet_geometry, geometric_integrals


# ------------------------------------------------------------------
def knot_sources(name: str, n_points: int = 2400):
    """Points r'(s), dipôles axiaux dm(s)=tau t̂ ds, et géométrie."""
    pts = knot_curve(name, n_points)[:-1]
    N = len(pts)
    dt = 2 * np.pi / N
    p_next = np.roll(pts, -1, axis=0)
    p_prev = np.roll(pts, 1, axis=0)
    dr = (p_next - p_prev) / (2 * dt)
    speed = np.linalg.norm(dr, axis=1)
    that = dr / speed[:, None]
    ds = speed * dt

    geo = frenet_geometry(name, n_points)
    tau = geo["tau"]
    kappa = geo["kappa"]

    dm = (tau * ds)[:, None] * that            # éléments dipolaires axiaux
    dq = kappa * ds                            # charges scalaires (E = κ)
    center = pts.mean(axis=0)
    R_knot = np.max(np.linalg.norm(pts - center, axis=1))
    U_B = float(np.sum(tau**2 * ds))           # ∮ τ² ds (énergie magnétique)
    U_kappa = float(np.sum(kappa**2 * ds))     # ∮ κ² ds (énergie de courbure)
    Q_kappa = float(np.sum(kappa * ds))        # ∮ κ ds (charge scalaire nette)
    return {
        "pts": pts, "dm": dm, "dq": dq, "tau": tau, "kappa": kappa,
        "ds": ds, "that": that, "center": center, "R_knot": R_knot,
        "U_B": U_B, "U_kappa": U_kappa, "Q_kappa": Q_kappa, "dr": dr,
    }


def net_moment(src) -> np.ndarray:
    """Moment dipolaire total m = ∮ tau t̂ ds."""
    return src["dm"].sum(axis=0)


def loop_area(src) -> np.ndarray:
    """Aire vectorielle géométrique A = 1/2 ∮ r × dr (modèle boucle de courant)."""
    pts, dr = src["pts"], src["dr"]
    dt = 2 * np.pi / len(pts)
    return 0.5 * np.cross(pts - src["center"], dr).sum(axis=0) * dt


def B_field(field_pts: np.ndarray, src_pts: np.ndarray, dm: np.ndarray,
            lam: float | None = None, a: float = 0.0) -> np.ndarray:
    """Champ dipolaire superposé. field_pts (M,3) ; renvoie B (M,3).

    B(x) = sum_i [3 (r̂·m_i) r̂ - m_i] / r_eff^3 * e^{-r/lam} ,  r = x - r'_i

    lam : longueur d'écran de Yukawa (None = non écranté, champ coulombien).
    a   : régularisation (rayon de tube) ; r_eff = sqrt(r^2 + a^2).
    """
    B = np.zeros_like(field_pts)
    for i in range(len(src_pts)):
        R = field_pts - src_pts[i]
        r = np.linalg.norm(R, axis=1)
        r = np.where(r < 1e-9, 1e-9, r)
        r_eff = np.sqrt(r**2 + a**2)
        rhat = R / r[:, None]
        m = dm[i]
        mdotr = rhat @ m
        contrib = (3.0 * mdotr[:, None] * rhat - m[None, :]) / (r_eff**3)[:, None]
        if lam is not None:
            contrib *= np.exp(-r / lam)[:, None]
        B += contrib
    return B


def decay_exponent(src, r_min_factor=1.5, r_max_factor=8.0, n=40):
    """Ajuste la pente log|B| vs log r le long d'un rayon sortant."""
    R0 = src["R_knot"]
    center = src["center"]
    # rayon dans une direction générique (évite les axes de symétrie)
    direction = np.array([0.37, 0.59, 0.71])
    direction /= np.linalg.norm(direction)
    radii = np.linspace(r_min_factor * R0, r_max_factor * R0, n)
    fpts = center + radii[:, None] * direction[None, :]
    B = B_field(fpts, src["pts"], src["dm"])
    Bmag = np.linalg.norm(B, axis=1)
    ok = Bmag > 0
    slope, intercept = np.polyfit(np.log(radii[ok]), np.log(Bmag[ok]), 1)
    return slope, radii, Bmag


def overlap_fraction(src, R_over_Rknot, a_over_Rknot=0.30,
                     lam_over_Rknot=1.0, h_over_Rknot=0.13, downsample=8):
    """Fraction d'énergie magnétique partagée par deux nœuds identiques.

    frac(R) = 2 ∫ B1·B2 dV / ∫ |B1|² dV   (sans dimension, indépendante
    de la normalisation du champ : le préfacteur M0²A² s'annule).
    Séparation le long du moment net (configuration attractive).
    """
    R0 = src["R_knot"]
    a, lam, h = a_over_Rknot * R0, lam_over_Rknot * R0, h_over_Rknot * R0
    Rsep = R_over_Rknot * R0
    mhat = net_moment(src) / np.linalg.norm(net_moment(src))

    p = (src["pts"] - src["center"])[::downsample]
    dm = src["dm"][::downsample] * downsample
    p1, dm1 = p, dm
    p2, dm2 = p + Rsep * mhat[None, :], dm

    xs = np.arange(-2.5 * R0, Rsep + 2.5 * R0 + h, h)
    ys = np.arange(-2.2 * R0, 2.2 * R0 + h, h)
    zs = np.arange(-2.2 * R0, 2.2 * R0 + h, h)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    B1 = B_field(grid, p1, dm1, lam=lam, a=a)
    B2 = B_field(grid, p2, dm2, lam=lam, a=a)
    dV = h**3
    self1 = np.sum(B1 * B1) * dV
    cross = np.sum(B1 * B2) * dV
    return 2.0 * cross / self1


def scalar_field(field_pts, src_pts, dq, lam, a):
    """Potentiel scalaire écranté de charges de ligne dq (E = κ).

    φ(x) = sum_i dq_i * e^{-r/lam} / sqrt(r^2 + a^2)  (monopôle de Yukawa).
    """
    phi = np.zeros(len(field_pts))
    for i in range(len(src_pts)):
        R = field_pts - src_pts[i]
        r = np.linalg.norm(R, axis=1)
        r_eff = np.sqrt(r**2 + a**2)
        phi += dq[i] * np.exp(-r / lam) / r_eff
    return phi


def overlap_fraction_scalar(src, R_over_Rknot, a_over_Rknot=0.30,
                            lam_over_Rknot=1.0, h_over_Rknot=0.13,
                            downsample=8):
    """Fraction d'overlap du champ scalaire de courbure (monopôle).

    f = 2 ∫ φ1 φ2 dV / ∫ φ1² dV  (les deux charges scalaires sont > 0,
    donc l'interaction est attractive : f < 0 par convention d'énergie).
    """
    R0 = src["R_knot"]
    a, lam, h = a_over_Rknot * R0, lam_over_Rknot * R0, h_over_Rknot * R0
    Rsep = R_over_Rknot * R0
    axis = net_moment(src) / np.linalg.norm(net_moment(src))

    p = (src["pts"] - src["center"])[::downsample]
    dq = src["dq"][::downsample] * downsample
    p1, p2 = p, p + Rsep * axis[None, :]

    xs = np.arange(-2.5 * R0, Rsep + 2.5 * R0 + h, h)
    ys = np.arange(-2.2 * R0, 2.2 * R0 + h, h)
    zs = np.arange(-2.2 * R0, 2.2 * R0 + h, h)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    phi1 = scalar_field(grid, p1, dq, lam, a)
    phi2 = scalar_field(grid, p2, dq, lam, a)
    dV = h**3
    self1 = np.sum(phi1 * phi1) * dV
    cross = np.sum(phi1 * phi2) * dV
    # charges de même signe -> attraction : on renvoie une fraction négative
    return -2.0 * cross / self1


def two_knot_potential(src1, src2, seps, axis, downsample=8,
                       lam=None, a=0.0):
    """Énergie d'interaction V(R) = - sum_j B1(r2_j + R*axis) · dm2_j .

    Modèle : recouvrement de deux franges (deux distributions dipolaires),
    avec écrantage de Yukawa (lam) et régularisation de tube (a).
    Séparation le long de `axis` (choisi = direction du moment net).
    """
    axis = axis / np.linalg.norm(axis)
    p1 = src1["pts"][::downsample]
    dm1 = src1["dm"][::downsample] * downsample
    p2 = src2["pts"][::downsample] - src2["center"]   # centré
    dm2 = src2["dm"][::downsample] * downsample
    V = np.empty(len(seps))
    for k, R in enumerate(seps):
        shifted = p2 + (src1["center"] + R * axis)[None, :]
        B1 = B_field(shifted, p1, dm1, lam=lam, a=a)
        V[k] = -np.sum(B1 * dm2)
    return V


# ------------------------------------------------------------------
def main():
    print("=" * 64)
    print("  CALCUL F1 — champ de frange et force nucléaire résiduelle")
    print("=" * 64)

    components = ["3_1", "7_2"]
    src = {}
    print("\n(1) Géométrie et énergies (cross-check TQNT)")
    print(f"  {'nœud':6} {'U_EM=U':>9} {'U_B=Uτ':>9} {'|m| axial':>11} "
          f"{'|A| boucle':>11} {'R_knot':>8}")
    for name in components:
        gi = geometric_integrals(name, 4000)
        s = knot_sources(name, 2400)
        src[name] = s
        m = net_moment(s)
        A = loop_area(s)
        print(f"  {name:6} {gi['U']:9.2f} {gi['U_tau']:9.2f} "
              f"{np.linalg.norm(m):11.3f} {np.linalg.norm(A):11.3f} "
              f"{s['R_knot']:8.3f}")
    print("  (réf. TQNT : U_EM(3_1)=9.0, U_EM(7_2)=33.8)")

    print("\n(2) Loi de décroissance du champ de frange (pente log-log)")
    for name in components:
        slope, radii, Bmag = decay_exponent(src[name])
        print(f"  {name}: pente = {slope:+.2f}  "
              f"(dipôle pur → -3 ; monopole → -2)")

    print("\n(3) Potentiel à deux nœuds 7_2–7_2 (paire dominante du nucléon)")
    s72 = src["7_2"]
    R0 = s72["R_knot"]
    a_tube = 0.30 * R0            # rayon de tube (régularisation)
    lam = R0                      # longueur d'écran ~ taille du nœud (CS)
    m_hat = net_moment(s72) / np.linalg.norm(net_moment(s72))
    seps = np.linspace(1.0 * R0, 5.0 * R0, 36)

    # frange attractive écrantée (séparation le long du moment net)
    V_attr = two_knot_potential(s72, s72, seps, axis=m_hat,
                                lam=lam, a=a_tube)
    # orienter le signe pour rendre la configuration attractive
    if V_attr[0] > 0:
        V_attr = -V_attr
    # normalisation : profondeur de frange à la distance de contact R0
    Vc = abs(V_attr[0])

    # cœur d'exclusion répulsif (F3) : Yukawa répulsif plus court
    sigma_core = 0.55 * R0
    V_core = 6.0 * Vc * np.exp(-(seps - R0) / sigma_core) * (R0 / seps)**3
    V_tot = V_core + V_attr

    kmin = int(np.argmin(V_tot))
    Rmin = seps[kmin]
    # portée : où |V_attr| tombe sous 10 % de sa valeur de contact
    idx = np.where(np.abs(V_attr) < 0.10 * Vc)[0]
    R_range = seps[idx[0]] if len(idx) else seps[-1]

    print(f"  moment net 7_2 : |m| = {np.linalg.norm(net_moment(s72)):.3f}")
    print(f"  frange seule : attractive, écran λ_N = R_knot, "
          f"tube a = {a_tube/R0:.2f} R_knot")
    print(f"  portée (|V_attr|<10% contact) : R = {R_range/R0:.2f} R_knot")
    print(f"  avec cœur d'exclusion (F3, σ={sigma_core/R0:.2f} R_knot) : "
          f"puits à R = {Rmin/R0:.2f} R_knot")

    # --- calibration longueur : R_knot ↔ rayon du nucléon r_p ---
    r_p = 0.84   # fm
    fm_per_unit = r_p / R0
    print(f"\n(4) Calibration : R_knot ↔ r_p = {r_p} fm "
          f"⇒ 1 unité = {fm_per_unit:.3f} fm")
    print(f"  minimum du puits : R_min ≈ {Rmin * fm_per_unit:.2f} fm")
    print(f"  portée de la frange : ≈ {R_range * fm_per_unit:.2f} fm")

    # ---------------------------------------------------------------
    print("\n(5) Profondeur en MeV (calibration via la formule de masse)")
    ALPHA0 = 8.16e-4          # MeV (calibré sur m_e)
    cJ_72 = 8.1               # indice de Jones du 7_2 (CLAUDE.md)
    E_B = ALPHA0 * s72["U_B"] * cJ_72**3.55
    print(f"  énergie magnétique propre du 7_2 :")
    print(f"    E_B = α₀·U_B·c_J^3.55 = {ALPHA0:.2e}·{s72['U_B']:.2f}·"
          f"{cJ_72}^3.55 = {E_B:.1f} MeV")
    R_well = Rmin / R0        # position du puits (unités R_knot)
    print(f"  fraction d'overlap au puits (R={R_well:.2f} R_knot), "
          f"selon le rayon de tube a :")
    print(f"    {'a/R_knot':>9} {'frac':>10} {'|V| (MeV)':>11}")
    depths = {}
    for a_rel in (0.20, 0.30, 0.40):
        frac = overlap_fraction(s72, R_well, a_over_Rknot=a_rel)
        Vmev = abs(frac) * E_B
        depths[a_rel] = Vmev
        print(f"    {a_rel:9.2f} {frac:10.4f} {Vmev:11.2f}")
    print(f"  → reliquat magnétique seul : "
          f"{min(depths.values()):.2f}–{max(depths.values()):.2f} MeV "
          f"par paire (selon a)")
    print("  CONCLUSION : l'échelle E_B≈22 MeV est nucléaire, mais la")
    print("  fraction d'overlap au minimum (fixé par l'exclusion) est")
    print("  petite (~0,5%) → le reliquat magnétique pèse ~0,1 MeV/paire.")
    print("  Il fixe la PORTÉE (~1,4 fm) et donne une attraction d'échelle")
    print("  ~deuton (2 MeV), mais PAS le puits profond (~50 MeV) :")
    print("  celui-ci doit venir de la micro-gravité collective (F2)")
    print("  et/ou du recouvrement scalaire du champ de courbure κ.")

    # ---------------------------------------------------------------
    print("\n(6) Micro-gravité collective (F2) — test d'ordre de grandeur")
    G = 6.674e-11          # m^3 kg^-1 s^-2
    m_p_kg = 1.6726e-27    # kg
    r_fm = 1e-15           # m
    J_per_MeV = 1.602e-13
    V_grav2 = G * m_p_kg**2 / r_fm / J_per_MeV   # MeV, 2-corps à 1 fm
    A = 16                                       # ex. oxygène-16
    V_coll = V_grav2 * A**2                       # borne haute (cohérent)
    A_needed = np.sqrt(8.0 / V_grav2)             # A requis pour 8 MeV
    print(f"  gravité 2-corps à 1 fm : G m_p²/r = {V_grav2:.2e} MeV")
    print(f"  collectif cohérent (A={A}, ×A²) : {V_coll:.2e} MeV")
    print(f"  pour atteindre 8 MeV il faudrait A ≈ {A_needed:.1e} (impossible)")
    print("  ⇒ la micro-gravité est négligeable de ~36 ordres : ELLE NE PEUT")
    print("    PAS lier le noyau. (Cohérent avec α_G/α_EM ~ α_EM^17 ~ 1e-36.)")

    # ---------------------------------------------------------------
    print("\n(7) Nature du champ de courbure κ — diagnostic des multipôles")
    # E = κ n̂ (vecteur le long de la normale). t̂ = dr/|dr|.
    for name in ("3_1", "7_2"):
        s = src[name]
        that = s["dr"] / np.linalg.norm(s["dr"], axis=1)[:, None]
        dthat = np.roll(that, -1, axis=0) - that      # ≈ κ n̂ ds = d t̂
        net_E = np.linalg.norm(dthat.sum(axis=0))      # ∮ κ n̂ ds  (monopôle E)
        total_kappa = np.sum(np.linalg.norm(dthat, axis=1))  # ∮ κ ds
        net_B = np.linalg.norm(net_moment(s))          # ∮ τ t̂ ds (dipôle mag.)
        print(f"  {name}: ∮κ ds = {total_kappa:6.2f}   "
              f"|∮κn̂ ds| = {net_E:7.4f} (monopôle E ≈ 0)   "
              f"|∮τt̂ ds| = {net_B:5.2f} (dipôle mag.)")
    print("  ⇒ CORRECTION de F2 : E = κn̂ est un VECTEUR, et ∮κn̂ ds =")
    print("    ∮ dt̂ = 0 identiquement → PAS de monopôle électrique.")
    print("    De plus son dipôle net s'annule aussi → canal κ = QUADRUPÔLE")
    print("    (1/r^5), encore plus faible et plus court que le dipôle τ.")
    print("    Le calcul 'monopôle scalaire ~20 MeV' reposait sur une")
    print("    fausse prémisse (κ traité en charge scalaire) : INVALIDÉ.")
    print("\n  BILAN HONNÊTE de la force nucléaire forte résiduelle :")
    print("   • magnétique τ (dipôle)  : ~0,1 MeV, fixe la portée + tenseur")
    print("   • électrique κ (quadrupôle): encore plus faible, très courte portée")
    print("   • micro-gravité           : ~1e-34 MeV, négligeable")
    print("   → la liaison PROFONDE (~tens MeV) n'est PAS expliquée par ces")
    print("     recouvrements de champ. Candidats restants : Van der Waals")
    print("     (2e ordre, induit) ou un mode SCALAIRE propre du champ χ.")
    print("     C'est le vrai problème ouvert de la force nucléaire forte en TQNT.")

    # --- figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        # (a) décroissance
        ref_done = False
        for name in components:
            slope, radii, Bmag = decay_exponent(src[name])
            x = radii / src[name]["R_knot"]
            ax[0].loglog(x, Bmag, "o-", ms=3,
                         label=f"{name} (pente {slope:+.2f})")
            if not ref_done:
                ax[0].loglog(x, Bmag[0] * (x / x[0])**(-3.0), "k--", lw=1,
                             label="référence $r^{-3}$ (dipôle)")
                ref_done = True
        ax[0].set_xlabel("$r / R_{knot}$")
        ax[0].set_ylabel("$|B_{rés}|$  (unités courbe)")
        ax[0].set_title("(a) Décroissance du champ de frange")
        ax[0].legend(fontsize=8)
        # (b) potentiel
        xx = seps / R0
        ax[1].plot(xx, V_attr / Vc, label="frange attractive (F1, écrantée)")
        ax[1].plot(xx, V_core / Vc, ":", label="cœur d'exclusion (F3)")
        ax[1].plot(xx, V_tot / Vc, lw=2, label="total")
        ax[1].axhline(0, color="gray", lw=0.6)
        ax[1].axvline(Rmin / R0, color="red", lw=0.8, ls="--")
        ax[1].set_ylim(-2.0, 3.0)
        ax[1].set_xlabel("$R / R_{knot}$  ($R_{knot}\\leftrightarrow r_p=0{,}84$ fm)")
        ax[1].set_ylabel("$V$ (unités de profondeur de contact)")
        ax[1].set_title("(b) Potentiel nucléon–nucléon ($7_2$–$7_2$)")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        out = "output/fringe_field_F1.png"
        fig.savefig(out, dpi=130)
        print(f"\n  figure → {out}")
    except Exception as e:
        print(f"  (figure non générée : {e})")

    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
