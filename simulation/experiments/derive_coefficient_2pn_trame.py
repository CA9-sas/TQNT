"""Coefficient 2PN exact de la métrique canonique de la trame vs RG.

Objet : la V6 (eq:BH-metric) fixe la forme champ-fort de la métrique
émergente en coordonnées isotropes du fond plat :

    g_00 = -c^2 e^{-h},   g_ij = (1+h) delta_ij,   h = 2GM/(c^2 r)

(h = champ de déplétion ; ∇²h = -κ_g|Ψ|² est LINÉAIRE, donc
h = 2GM/(c²r) exactement à l'extérieur — toute la non-linéarité vit
dans les formes fermées e^{-h} et 1+h).

RG (Schwarzschild isotrope) : A_S = ((1-U/2)/(1+U/2))², B_S = (1+U/2)⁴.

Développements (U = GM/c²r) :
    TQNT : A = 1-2U+2U²-(4/3)U³...,  B = 1+2U             (EXACT)
    RG   : A = 1-2U+2U²-(3/2)U³...,  B = 1+2U+(3/2)U²+(1/2)U³...

=> beta_PPN = gamma_PPN = 1 identiques (1PN) ; l'écart des métriques
   est delta_alpha3 = +1/6 (g_00, ordre U³) et delta_b2 = -3/2
   (secteur SPATIAL, ordre U² — la source dominante).

DÉRIVATION (manuscrite, vérifiée ici) — avance du péricentre par
orbite, invariants (E, L), w = 1/r, u = GM/(c²p), p = L²c²/(GM),
e² = 1 + (E²-1)L²c²/(GM)² (étiquettes invariantes newtoniennes) :

    L²(dw/dphi)² = P(w) = B[(E²)/A - 1] - L²w²
    P tronqué 2PN = eps + c1 w - c2h w² + c3 w³   (eps = E²-1)
      c1 = m[2(1+eps) + 2 gamma eps]
      c2h = L² - m²[(a2'+4 gamma)(1+eps) + b2 eps],  a2' = 4-2 alpha2
      c3 = m³[(a3'+2 gamma a2'+2 b2)(1+eps) + b3 eps],
                                              a3' = 8-8 alpha2-alpha3
    Factorisation cubique EXACTE (racines w-, w+, w3 = 1/eta lointaine,
    méthode de Weinberg) puis développement (w/w3 ~ u²) :

    Delta phi / 2pi = 1 + (a2'+4g)u/2
        + u² [ 3/8 (a2'+4g)² + 3/2 (a3'+2g a2'+2b2)
               - (1-e²)(a2'+4g+b2)/2 ]                (g = gamma)

    Contrôle : RG -> 6 pi u + pi u² (45 + 15/2 e²) ;
               TQNT -> 6 pi u + pi u² (37 + 6 e²).
    Pour deux théories à beta=gamma=1 identiques (alpha2=1) :

    delta(Delta phi) = pi u² [ -3 delta_alpha3 + (5 + e²) delta_b2 ]

Spécialisation TQNT - RG (delta_alpha3 = 1/6, delta_b2 = -3/2) :

    delta(Delta phi) = -pi (8 + 3/2 e²) u²    par orbite
    (part delta_b2 spatiale : -15/2 - 3e²/2 ; part delta_alpha3 : -1/2
     — le secteur spatial domine 15:1).

NB méthode : l'astuce Delta phi = 4L d/d(eps) Int sqrt(P) est FAUSSE
ici (dP/d(eps) = B/A != 1) — première version manuscrite corrigée par
l'arbitre numérique ; la factorisation cubique est exacte.

La trame précesse MOINS que la RG à 2PN. NB : la ligne V6 « la
différence n'apparaît qu'au 2e ordre PPN (~U³~1e-23) » situait le bon
ordre PN mais sous-estimait la magnitude : l'écart relatif sur la
précession est ((8+3e²/2)/6) u, pas u².

Ce script VÉRIFIE numériquement (mpmath) sur métriques EXACTES :
  [1] contrôle 1PN commun 6 pi u ;
  [2] différence TQNT-RG aux mêmes (E,L) -> -pi(8+3e²/2)u² (2 valeurs
      de e, séquence u -> 0, extrapolation de Richardson) ;
  [3] métriques-sondes (Schwarzschild + delta_b2 u² seul, puis
      + delta_alpha3 u³ seul) -> chaque pièce de la formule générale ;
puis APPLIQUE : S301 (Sgr A*), contrôle Mercure, pulsar double
(indicatif — EOM à deux corps non dérivée, cible nommée).
"""

import mpmath as mp

mp.mp.dps = 40


# ================================================================
# Métriques exactes (coordonnées isotropes), w = 1/r, m = GM/c²
# ================================================================
def metric_tqnt(w, m):
    """Canonique V6 : A = e^{-2mw}, B = 1 + 2mw (exactes)."""
    u = m*w
    return mp.e**(-2*u), 1 + 2*u


def metric_schw_iso(w, m):
    """Schwarzschild isotrope exacte."""
    x = m*w/2
    return ((1 - x)/(1 + x))**2, (1 + x)**4


def make_probe_b2(db2):
    """Schwarzschild + db2 u² dans B seulement (sonde exacte)."""
    def metric(w, m):
        A, B = metric_schw_iso(w, m)
        return A, B + db2*(m*w)**2
    return metric


def make_probe_a3(da3):
    """Schwarzschild + da3 u³ dans A seulement (sonde exacte)."""
    def metric(w, m):
        A, B = metric_schw_iso(w, m)
        return A + da3*(m*w)**3, B
    return metric


# ================================================================
# Intégrale d'orbite : Delta phi (E, L) — invariant de jauge
# ================================================================
def P_of_w(w, m, E2, L2, metric):
    A, B = metric(w, m)
    return B*(E2/A - 1) - L2*w**2


def solve_EL(wm, wp, m, metric):
    """(E², L²) tels que P(w-) = P(w+) = 0 (système linéaire)."""
    Am, Bm = metric(wm, m)
    Ap, Bp = metric(wp, m)
    M11, M12, R1 = Bm/Am, -wm**2, Bm
    M21, M22, R2 = Bp/Ap, -wp**2, Bp
    det = M11*M22 - M12*M21
    return (R1*M22 - M12*R2)/det, (M11*R2 - R1*M21)/det


def turning_points(m, E2, L2, metric, wm0, wp0):
    f = lambda w: P_of_w(w, m, E2, L2, metric)
    return mp.findroot(f, wm0), mp.findroot(f, wp0)


def delta_phi(m, E2, L2, metric, wm, wp):
    """Delta phi = 2L ∫ dw/√P, substitution w = c - d cos(theta)."""
    cmid = (wp + wm)/2
    d = (wp - wm)/2
    L = mp.sqrt(L2)

    def integrand(theta):
        st2 = mp.sin(theta)**2
        if st2 == 0:
            return mp.mpf(0)
        w = cmid - d*mp.cos(theta)
        R = P_of_w(w, m, E2, L2, metric)/(d**2*st2)
        # aux bornes, l'annulation P ~ 0 peut rendre R negatif d'un
        # epsilon numerique : |R| est exact a la precision de travail
        return 2*L/mp.sqrt(abs(R))

    return mp.re(mp.quad(integrand, [0, mp.pi/2, mp.pi]))


def orbit_invariants(m, E2, L2):
    """Étiquettes invariantes newtoniennes u = m/p, e²."""
    pinv = L2/m
    u = m/pinv
    e2 = 1 + (E2 - 1)*L2/m**2
    return u, e2


def compare(metric_test, metric_ref, e, pvals, coeff_th, label):
    """delta(Delta phi)/(pi u²) pour u -> 0, vs prédiction coeff_th(e²)."""
    m = mp.mpf(1)
    print(f"\n  {label}   (e = {float(e):.4f})")
    print(f"  {'u = m/p':>11s} {'delta/(pi u^2)':>17s} {'attendu':>11s}")
    rows = []
    for pval in pvals:
        wm, wp = (1 - e)/pval, (1 + e)/pval
        E2, L2 = solve_EL(wm, wp, m, metric_test)
        d1 = delta_phi(m, E2, L2, metric_test, wm, wp)
        wmR, wpR = turning_points(m, E2, L2, metric_ref, wm, wp)
        d0 = delta_phi(m, E2, L2, metric_ref, wmR, wpR)
        u, e2 = orbit_invariants(m, E2, L2)
        Cnum = (d1 - d0)/(mp.pi*u**2)
        Cth = coeff_th(e2)
        rows.append((u, Cnum, Cth))
        print(f"  {float(u):11.3e} {float(Cnum):17.6f} {float(Cth):11.4f}")
    (u1, C1, _), (u2, C2, Cth) = rows[-2], rows[-1]
    Cx = C2 + (C2 - C1)*u2/(u1 - u2)      # Richardson lineaire en u
    err = abs(Cx - Cth)/abs(Cth)
    print(f"  extrapolé u->0 : {float(Cx):.6f}  attendu {float(Cth):.6f}"
          f"  écart {float(err):.1e}")
    assert err < 5e-3, f"VERIF ECHOUEE ({label})"
    return Cx


print("=" * 70)
print("[1] Contrôle 1PN : les deux métriques donnent 6 pi u")
print("=" * 70)
m1 = mp.mpf(1)
for met, name in ((metric_tqnt, "TQNT"), (metric_schw_iso, "RG")):
    e, pval = mp.mpf('0.3'), mp.mpf(20000)
    wm, wp = (1 - e)/pval, (1 + e)/pval
    E2, L2 = solve_EL(wm, wp, m1, met)
    dp = delta_phi(m1, E2, L2, met, wm, wp)
    u, _ = orbit_invariants(m1, E2, L2)
    ratio = (dp - 2*mp.pi)/(6*mp.pi*u)
    print(f"  {name:5s}: (Delta phi - 2pi)/(6 pi u) = {float(ratio):.6f}")
    assert abs(ratio - 1) < 2e-3

print("\n" + "=" * 70)
print("[1b] Contrôle 2PN absolu : (Dphi - 2pi - 6pi u)/(pi u²)")
print("     prédiction : RG 45 + 7,5 e² ; TQNT 37 + 6 e²")
print("=" * 70)
for met, name, k2 in ((metric_schw_iso, "RG",
                       lambda e2: 45 + mp.mpf(15)/2*e2),
                      (metric_tqnt, "TQNT",
                       lambda e2: 37 + 6*e2)):
    e = mp.mpf('0.5')
    rows = []
    for pval in (mp.mpf(3000), mp.mpf(10000)):
        wm, wp = (1 - e)/pval, (1 + e)/pval
        E2, L2 = solve_EL(wm, wp, m1, met)
        dp = delta_phi(m1, E2, L2, met, wm, wp)
        u, e2 = orbit_invariants(m1, E2, L2)
        rows.append((u, (dp - 2*mp.pi - 6*mp.pi*u)/(mp.pi*u**2), k2(e2)))
    (u1, K1, _), (u2, K2, Kth) = rows
    Kx = K2 + (K2 - K1)*u2/(u1 - u2)
    print(f"  {name:5s}: extrapolé {float(Kx):9.4f}  attendu"
          f" {float(Kth):9.4f}  écart {float(abs(Kx-Kth)/Kth):.1e}")
    assert abs(Kx - Kth)/Kth < 5e-3

print("\n" + "=" * 70)
print("[2] Différence TQNT - RG aux mêmes (E, L)")
print("    prédiction : -(8 + 1,5 e²)")
print("=" * 70)
PV = (mp.mpf(300), mp.mpf(1000), mp.mpf(3000), mp.mpf(10000))
for e in (mp.mpf('0.3'), mp.mpf('0.9824')):
    compare(metric_tqnt, metric_schw_iso, e, PV,
            lambda e2: -(8 + mp.mpf(3)/2*e2), "TQNT vs RG")

print("\n" + "=" * 70)
print("[3] Métriques-sondes : chaque pièce de la formule générale")
print("    delta(Dphi) = pi u² [ -3 d_alpha3 + (5 + e²) d_b2 ]")
print("=" * 70)
compare(make_probe_b2(mp.mpf(-1.5)), metric_schw_iso, mp.mpf('0.5'), PV,
        lambda e2: mp.mpf(-1.5)*(5 + e2), "sonde d_b2 = -3/2")
compare(make_probe_a3(mp.mpf(1)/6), metric_schw_iso, mp.mpf('0.5'), PV,
        lambda e2: -3*(mp.mpf(1)/6), "sonde d_alpha3 = +1/6")

# ================================================================
# Applications
# ================================================================
print("\n" + "=" * 70)
print("[4] Applications")
print("=" * 70)
G = 6.6710e-11          # G_prédit (convention de calcul du cadre)
c = 2.99792458e8
Msun = 1.989e30
AU = 1.496e11

# ---- S301 (Nature 2026 : P = 8,7 an, v_péri = 25 000 km/s) ----
M_sgr = 4.297e6*Msun
GM = G*M_sgr
T = 8.7*365.25*86400.0
a_orb = float((GM*(T/(2*mp.pi))**2)**(mp.mpf(1)/3))
v_peri = 2.5e7
r_p = 2/(v_peri**2/GM + 1/a_orb)         # vis-viva
e_s = 1 - r_p/a_orb
p_s = a_orb*(1 - e_s**2)
u_s = GM/c**2/p_s
dphi_1pn = 6*mp.pi*u_s
dphi_diff = -(8 + 1.5*e_s**2)*mp.pi*u_s**2
R0 = 8.277e3*3.0857e16
apo_ang = a_orb*(1 + e_s)/R0
uas = 1e6*3600*180/mp.pi
print(f"\nS301 : a = {a_orb/AU:.0f} UA, e = {e_s:.4f}, "
      f"r_p = {r_p/AU:.1f} UA = {r_p/(2*GM/c**2):.0f} r_s, "
      f"u = {u_s:.3e}")
print(f"  précession 1PN        = {float(dphi_1pn)*206265:.0f}\"/orbite"
      f" ({float(dphi_1pn*180/mp.pi):.2f} deg)")
print(f"  écart TQNT-RG (2PN)   = {float(dphi_diff)*206265:+.1f}\"/orbite"
      f"  ({float(abs(dphi_diff)/dphi_1pn*100):.2f} % du 1PN)")
print(f"  dérive apocentre      = "
      f"{float(abs(dphi_diff)*apo_ang*uas):.1f} muas/orbite = "
      f"{float(abs(dphi_diff)*apo_ang*uas)/8.7:.1f} muas/an")
dM_equiv = float(abs(dphi_diff)/(2*mp.pi))*M_sgr/Msun
print(f"  masse étendue mimant l'écart ~ {dM_equiv:.0f} Msun "
      f"(bornes actuelles ~1e3 Msun)")

# ---- contrôle Mercure ----
GMsol = G*1.989e30
p_m = 5.791e10*(1 - 0.2056**2)
u_m = GMsol/c**2/p_m
frac_m = (8 + 1.5*0.2056**2)/6*u_m
print(f"\nMercure : écart relatif sur la précession = {frac_m:.2e}"
      f" ({frac_m*42.98:.1e} \"/siècle) — inobservable.")
print("  NB V6 : « ~U³ ~ 1e-23 » -> à corriger en ((8+3e²/2)/6) u"
      " ~ 4e-8 ;")
print("  conclusion solaire (invisible) inchangée.")

# ---- pulsar double J0737-3039A/B (INDICATIF, test-masse) ----
M_tot = 2.587*Msun
Pb = 2.4535*3600.0
e_p = 0.0878
a_p = float((G*M_tot*(Pb/(2*mp.pi))**2)**(mp.mpf(1)/3))
u_p = G*M_tot/c**2/(a_p*(1 - e_p**2))
frac_p = (8 + 1.5*e_p**2)/6*u_p
print(f"\nPulsar double J0737-3039 (test-masse, INDICATIF) :")
print(f"  u = {u_p:.2e} ; écart relatif sur omega_dot ~ {frac_p:.1e}")
print("  précision mesurée 1,3e-6 (Kramer+ 2021) ; terme 2PN RG ~ qq"
      " 1e-5 du total")
print("  => écart TQNT ~ l'ordre du terme 2PN résolu : confrontation")
print("  PRIORITAIRE — exige d'abord l'EOM à deux corps de la trame.")
