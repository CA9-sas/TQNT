"""Dette 2 : stabilité relativiste d'une enveloppe d'hydrogène sur un coeur
sombre transparent de Classe F, durée de la phase visible, rôle de la
rotation et de la dynamo.

MODÈLE. Enveloppe P = K rho^(4/3) en équilibre hydrostatique dans le
potentiel gaz + coeur uniforme transparent (M_F, R_F). L'entropie K est le
paramètre de la séquence de contraction : à chaque K on ajuste rho_c pour
que M_gaz = M_env, on lit R_env, T_c, <beta>_P (P_gaz/P résolu localement),
et la compacité pondérée par la pression <2GM(r)/rc^2>_P.

CRITÈRE (Chandrasekhar post-newtonien, calibré sur n=3 pur) :
    beta_eff/6  >  C * <2GM(r)/(r c^2)>_P ,   C calibré (~0,87)
    beta_eff = <beta>_P + 1,17 (T/|W|)   [rotation, Baumgarte-Shapiro 1999 :
    uniforme au bris T/|W| = 0,009 -> R_crit = 640 R_S]
CONTRACTION : dE/dt = -L_Edd, E = -(3 beta/4) G M_tot M_env / R.
DYNAMO : B d'équipartition avec la rotation ; MRI en t ~ 1/(alpha Omega).
"""
from __future__ import annotations
import math
import numpy as np
from scipy.integrate import solve_ivp

G, c, kB, mH, sigT, aR, mu0 = 6.6710e-11, 2.998e8, 1.381e-23, 1.673e-27, 6.652e-29, 7.566e-16, 4e-7 * math.pi
Msun, Rsun, AU, yr, GAUSS = 1.989e30, 6.96e8, 1.496e11, 3.156e7, 1e-4
MU, K_ROT = 0.6, 1.17


def L_edd(M):
    return 4 * math.pi * G * M * mH * c / sigT


def beta_local(rho, P):
    """P_gaz/P : résout rho k T/(mu mH) + a T^4/3 = P par Newton (T_rad comme départ)."""
    T = (3 * P / aR) ** 0.25
    A = rho * kB / (MU * mH)
    for _ in range(8):
        f = A * T + aR * T ** 4 / 3 - P
        T -= f / (A + 4 * aR * T ** 3 / 3)
    return A * T / P, T


def integre(K, M_F, R_F, rho_c):
    """Intègre en x = ln r. État : [P, M_gaz, I_P, S_P]. Retourne R, M_gaz, <comp>_P, beta_c.
    beta est pris uniforme et égal à sa valeur centrale (modèle standard d'Eddington)."""
    r0 = 1e-4 * (R_F if R_F > 0 else Rsun)
    P0 = K * rho_c ** (4 / 3)
    b_c, _ = beta_local(rho_c, P0)

    def rhs(x, y):
        r = math.exp(x)
        P, Mg = y[0], y[1]
        if P <= 0:
            return [0, 0, 0, 0, 0]
        rho = (P / K) ** 0.75
        M_in = Mg + (M_F * min(1.0, (r / R_F) ** 3) if R_F > 0 else M_F)
        dP = -G * M_in * rho / r          # dP/dx = r dP/dr
        dM = 4 * math.pi * r ** 3 * rho
        dV = 4 * math.pi * r ** 3
        comp = 2 * G * M_in / (r * c ** 2)
        return [dP, dM, comp * P * dV, P * dV]

    def surface(x, y):
        return y[0] - 1e-9 * P0
    surface.terminal = True

    sol = solve_ivp(rhs, [math.log(r0), math.log(3e5 * AU)], [P0, 0.0, 0.0, 0.0],
                    events=surface, rtol=1e-6, atol=1e-30, max_step=0.25)
    R = math.exp(sol.t[-1])
    P, Mg, I, S = sol.y[:, -1]
    return R, Mg, I / S, b_c


def structure(rho_c, M_env, M_F, R_F):
    """À rho_c FIXÉ, ajuste l'entropie K pour que M_gaz = M_env.
    C'est la bonne paramétrisation d'une séquence de contraction à masse fixée :
    pour n=3 pur, K ne dépend alors pas de rho_c (dégénérescence d'Eddington) et
    R ∝ rho_c^(-1/6) ; le coeur lève la dégénérescence."""
    K0 = K_of_beta(7e-3)
    lo, hi = 1e-5 * K0, 1e5 * K0
    for _ in range(36):
        K = math.sqrt(lo * hi)
        R, Mg, I, b = integre(K, M_F, R_F, rho_c)
        if Mg < M_env:
            lo = K          # pas assez de masse : plus de pression
        else:
            hi = K
    _, T_c = beta_local(rho_c, K * rho_c ** (4 / 3))
    ok = abs(Mg / M_env - 1) < 0.05
    return dict(R=R, I=I, beta=b, T_c=T_c, rho_c=rho_c, K=K, Mg=Mg, ok=ok)


def K_of_beta(beta):
    return (3 * (1 - beta) / aR) ** (1 / 3) * (kB / (beta * MU * mH)) ** (4 / 3)


RHOS = [10 ** (x / 2) for x in range(-16, 7)]   # 1e-8 ... 1e3 kg/m3, pas 1/2 décade


def sequence(M_env, M_F, R_F, C, tw=0.0):
    Mt = M_env + M_F
    etats = [e for e in (structure(rc, M_env, M_F, R_F) for rc in RHOS) if e["ok"]]
    for s in etats:
        s["beta_eff"] = s["beta"] + K_ROT * tw
        s["marge"] = (s["beta_eff"] / 6) / (C * s["I"])
    etats.sort(key=lambda s: -s["R"])
    L, t, prev, crit = L_edd(Mt), 0.0, None, None
    for s in etats:
        if prev is not None:
            E1 = -(3 * prev["beta"] / 4) * G * Mt * M_env / prev["R"]
            E2 = -(3 * s["beta"] / 4) * G * Mt * M_env / s["R"]
            t += max(0.0, E1 - E2) / L
            if crit is None and s["marge"] < 1 <= prev["marge"]:
                crit = s
        s["t"] = t
        prev = s
    return etats, crit


def calibre_C(M=1e6 * Msun):
    """C tel que le polytrope pur devienne instable à R = 6,75 R_S/beta."""
    s = structure(1e-4, M, 0.0, 0.0)
    Rs = 2 * G * M / c ** 2
    R_crit = 6.75 * Rs / s["beta"]
    I_crit = s["I"] * s["R"] / R_crit          # I ∝ 1/R à structure fixée
    return (s["beta"] / 6) / I_crit, s


def dynamo(M_tot, R, tw=0.009, alpha=0.01):
    W = G * M_tot ** 2 / R
    V = 4 * math.pi / 3 * R ** 3
    B = math.sqrt(2 * mu0 * tw * W / V)
    Om = math.sqrt(G * M_tot / R ** 3)
    return B, Om, 1 / (alpha * Om)


def zeeman_halpha_nm(B_T):
    return 4.67e-13 * 6563.0 ** 2 * (B_T / GAUSS) * 0.1


if __name__ == "__main__":
    import time
    t0 = time.time()
    M_env, M_F = 1e6 * Msun, 1e5 * Msun
    Mt = M_env + M_F

    C, s0 = calibre_C()
    print(f"=== 0. CALIBRATION n=3 PUR : rho_c/rho_moy = {s0['rho_c']/(1e6*Msun/(4*math.pi/3*s0['R']**3)):.1f} (exact 54,2)"
          f" ; C = {C:.3f}  [{time.time()-t0:.0f} s]")
    et, cr = sequence(1e6 * Msun, 0.0, 0.0, C)
    print(f"  séquence pure : instabilité à R = {cr['R']/AU:.1f} UA (Chandrasekhar 18,7), t = {cr['t']/yr:.0f} ans, T_c = {cr['T_c']:.1e} K"
          if cr else "  séquence pure : pas d'instabilité trouvée dans la plage")

    print("\n=== DIAGNOSTIC : séquence complète, coeur 10 UA, sans rotation ===")
    et, _ = sequence(M_env, M_F, 10 * AU, C)
    print(f"  {'rho_c':>8s} {'R (UA)':>9s} {'beta':>8s} {'<comp>_P':>10s} {'marge':>7s} {'T_c':>8s} {'t (ans)':>9s}")
    for s in et:
        print(f"  {s['rho_c']:8.1e} {s['R']/AU:9.2f} {s['beta']:8.2e} {s['I']:10.2e} {s['marge']:7.2f} {s['T_c']:8.1e} {s['t']/yr:9.0f}")

    print("\n=== 1. COEUR 1e5 Msun, ENVELOPPE 1e6, SANS ROTATION ===")
    print(f"  {'R_F':>8s} {'R_crit':>10s} {'T_c(crit)':>10s} {'t_visible':>11s}  fusion avant l'instabilité ?")
    for RF_AU in (1, 3, 10, 30, 100):
        et, cr = sequence(M_env, M_F, RF_AU * AU, C)
        fus = next((s for s in et if s["T_c"] > 1e8), None)
        if cr:
            print(f"  {RF_AU:6d} UA {cr['R']/AU:8.1f} UA {cr['T_c']:10.1e} {cr['t']/yr:9.0f} ans  "
                  f"{'OUI' if fus and fus['t'] <= cr['t'] else 'non'}")
        else:
            last = et[-1]
            print(f"  {RF_AU:6d} UA {'stable':>10s} {last['T_c']:10.1e} {last['t']/yr:9.0f} ans  "
                  f"{'OUI' if fus else 'non'} (jusqu'à R = {last['R']/AU:.1f} UA)")
    print(f"  [{time.time()-t0:.0f} s]")

    print("\n=== 2. AVEC ROTATION T/|W| ===")
    print(f"  {'R_F':>8s} {'T/|W|':>6s} {'R_crit':>10s} {'T_c(crit)':>10s} {'t_visible':>11s}")
    for RF_AU in (10,):
        for tw in (0.0, 0.009, 0.05):
            et, cr = sequence(M_env, M_F, RF_AU * AU, C, tw=tw)
            if cr:
                print(f"  {RF_AU:6d} UA {tw:6.3f} {cr['R']/AU:8.1f} UA {cr['T_c']:10.1e} {cr['t']/yr:9.0f} ans")
            else:
                last = et[-1]
                print(f"  {RF_AU:6d} UA {tw:6.3f} {'stable':>10s} {last['T_c']:10.1e} {last['t']/yr:9.0f} ans (R>={last['R']/AU:.1f} UA)")
    print(f"  [{time.time()-t0:.0f} s]")

    print("\n=== 3. LA DYNAMO ===")
    for R_AU in (20, 50, 200):
        B, Om, t_mri = dynamo(Mt, R_AU * AU)
        print(f"  R={R_AU:4d} UA : B_équip = {B/GAUSS:.1e} G, P_rot = {2*math.pi/Om/86400:5.0f} j, "
              f"t_MRI(alpha=0,01) = {t_mri/yr:5.1f} ans, Zeeman H-alpha = {zeeman_halpha_nm(B)*1e3:.0f} pm (thermique 1e4 K : 47 pm)")


def enveloppes_legeres(C, M_F=1e5 * Msun, R_F=10 * AU):
    """Enveloppes de 1e5 et 3e5 Msun : la fusion s'allume-t-elle avant l'instabilité ?"""
    out = []
    for Me in (1e5, 3e5):
        for tw in (0.0, 0.009):
            et, cr = sequence(Me * Msun, M_F, R_F, C, tw=tw)
            fus = next((s for s in et if s["T_c"] > 1e8), None)
            out.append((Me, tw, L_edd(Me * Msun + M_F), cr, fus, et[-1]))
    return out


def duree_nucleaire(M_env, M_F, frac_H=0.1):
    return 0.007 * frac_H * M_env * c ** 2 / L_edd(M_env + M_F)


def table_coeur(C, M_F_sun, M_envs=(1e5, 3e5, 5e5, 1e6), R_Fs_AU=(10, 30), tws=(0.0, 0.009)):
    """Table (M_env, R_F, T/|W|) -> L_Edd, fusion ou non, durée. Utilisé pour M_F = 3e5."""
    M_F = M_F_sun * Msun
    rows = []
    for Me in M_envs:
        M_env = Me * Msun
        for RF in R_Fs_AU:
            for tw in tws:
                et, cr = sequence(M_env, M_F, RF * AU, C, tw=tw)
                fus = next((s for s in et if s["T_c"] > 1e8), None)
                fusion = bool(fus) and (cr is None or fus["t"] <= cr["t"])
                rows.append(dict(M_env=Me, R_F=RF, tw=tw, L=L_edd(M_env + M_F),
                                 fusion=fusion, crit=cr,
                                 t_nuc=duree_nucleaire(M_env, M_F) if fusion else None))
    return rows
