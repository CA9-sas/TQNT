#!/usr/bin/env python3
"""
Monte Carlo — Thermodynamique émergente TQNT (chapitre Thermo V3)
  Gravité P5 (Poisson FFT) + Lennard-Jones (KDTree vectorisé)
  Scan en T*, comparaison baryonique vs matière noire (C₆=0)
"""
import numpy as np
from scipy.spatial import cKDTree
import time, os

# === Paramètres TQNT (chapitre Thermodynamique) ===
N       = 3_000
SIGMA   = 0.76
EPSILON = 6.67
KAPPA_G = 0.1
R_CUT   = 2.5 * SIGMA
L       = (N / 0.08) ** (1/3)   # ~33.5

# MC
DELTA_R = SIGMA
N_EQUIL = 1500
N_PROD  = 2500
N_GRID  = 48
GS      = L / N_GRID

# Shift LJ
_s6 = (SIGMA**2 / R_CUT**2)**3
E_SHIFT = 4*EPSILON*(_s6**2 - _s6)

TEMPERATURES = [0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', '..', 'output')
os.makedirs(OUT, exist_ok=True)

# === LJ vectorisé ===
def total_lj_energy(pos, tree):
    """Énergie LJ totale (paires dans le cutoff, vectorisé)."""
    pairs = tree.query_pairs(R_CUT, output_type='ndarray')
    if len(pairs) == 0:
        return 0.0
    dr = pos[pairs[:,0]] - pos[pairs[:,1]]
    dr -= L * np.round(dr / L)
    r2 = np.sum(dr**2, axis=1)
    mask = r2 > 1e-10
    r2 = r2[mask]
    sr2 = SIGMA**2 / r2
    sr6 = sr2**3
    return np.sum(4*EPSILON*(sr6**2 - sr6) - E_SHIFT)

def lj_energy_of(pos, idx, tree):
    """Énergie LJ d'une particule (vectorisé sur ses voisins)."""
    nb = tree.query_ball_point(pos[idx], R_CUT)
    if len(nb) <= 1:
        return 0.0
    nb = np.array([j for j in nb if j != idx])
    if len(nb) == 0:
        return 0.0
    dr = pos[idx] - pos[nb]
    dr -= L * np.round(dr / L)
    r2 = np.sum(dr**2, axis=1)
    mask = (r2 > 1e-10) & (r2 < R_CUT**2)
    r2 = r2[mask]
    if len(r2) == 0:
        return 0.0
    sr2 = SIGMA**2 / r2
    sr6 = sr2**3
    return np.sum(4*EPSILON*(sr6**2 - sr6) - E_SHIFT)

# === Gravité P5 : Poisson FFT ===
def deposit_cic(pos):
    rho = np.zeros((N_GRID,N_GRID,N_GRID))
    gp = (pos % L) / GS
    gi = gp.astype(int) % N_GRID
    gf = gp - np.floor(gp)
    for dx in (0,1):
        wx = (1-gf[:,0]) if dx==0 else gf[:,0]
        for dy in (0,1):
            wy = (1-gf[:,1]) if dy==0 else gf[:,1]
            for dz in (0,1):
                wz = (1-gf[:,2]) if dz==0 else gf[:,2]
                np.add.at(rho, ((gi[:,0]+dx)%N_GRID,
                                (gi[:,1]+dy)%N_GRID,
                                (gi[:,2]+dz)%N_GRID), wx*wy*wz)
    return rho / GS**3

def poisson_fft(rho):
    rk = np.fft.rfftn(rho)
    kx = np.fft.fftfreq(N_GRID, d=GS)*2*np.pi
    ky = np.fft.fftfreq(N_GRID, d=GS)*2*np.pi
    kz = np.fft.rfftfreq(N_GRID, d=GS)*2*np.pi
    k2 = kx[:,None,None]**2 + ky[None,:,None]**2 + kz[None,None,:]**2
    k2[0,0,0] = 1.0
    pk = KAPPA_G * rk / k2
    pk[0,0,0] = 0.0
    return np.fft.irfftn(pk, s=[N_GRID]*3)

def interp_phi(phi, r):
    gp = (r % L) / GS
    gi = gp.astype(int) % N_GRID
    gf = gp - np.floor(gp)
    v = 0.0
    for dx in (0,1):
        wx = (1-gf[0]) if dx==0 else gf[0]
        for dy in (0,1):
            wy = (1-gf[1]) if dy==0 else gf[1]
            for dz in (0,1):
                wz = (1-gf[2]) if dz==0 else gf[2]
                v += wx*wy*wz*phi[(gi[0]+dx)%N_GRID,
                                  (gi[1]+dy)%N_GRID,
                                  (gi[2]+dz)%N_GRID]
    return v

# === Mesures ===
def compute_gr(pos, nb=150, rmax=4.0):
    tree = cKDTree(pos, boxsize=L)
    pairs = tree.query_pairs(rmax, output_type='ndarray')
    if len(pairs)==0:
        return np.linspace(0,rmax,nb), np.zeros(nb)
    dr = pos[pairs[:,0]] - pos[pairs[:,1]]
    dr -= L*np.round(dr/L)
    d = np.sqrt(np.sum(dr**2,axis=1))
    h, e = np.histogram(d, bins=nb, range=(0,rmax))
    rm = 0.5*(e[:-1]+e[1:])
    sv = 4*np.pi*rm**2*(rmax/nb)
    rho0 = N/L**3
    return rm, h / (N*rho0*sv/2)

def radial_rho(pos, nb=60):
    c = np.mean(pos,axis=0)
    dr = pos - c; dr -= L*np.round(dr/L)
    r = np.sqrt(np.sum(dr**2,axis=1))
    rmax = L/2
    h, e = np.histogram(r, bins=nb, range=(0,rmax))
    rm = 0.5*(e[:-1]+e[1:])
    sv = 4*np.pi*rm**2*(rmax/nb)
    return rm, h/np.maximum(sv,1e-30)

# === MC pour une température ===
def run_T(Tstar, use_lj=True, tag=""):
    kBT = Tstar * EPSILON
    # Init réseau cubique
    ns = int(np.ceil(N**(1/3)))
    sp = L/ns
    pos = np.array([[(ix+.5)*sp,(iy+.5)*sp,(iz+.5)*sp]
                    for ix in range(ns) for iy in range(ns)
                    for iz in range(ns)][:N])
    pos += .1*sp*(np.random.rand(N,3)-.5)
    pos %= L

    rho = deposit_cic(pos)
    phi = poisson_fft(rho)
    tree = cKDTree(pos, boxsize=L)

    na = nt = 0
    energies = []
    ntot = N_EQUIL + N_PROD

    for step in range(ntot):
        # Batch de moves
        nm = min(N, 200)
        idxs = np.random.randint(0, N, nm)
        sa = 0
        for idx in idxs:
            old = pos[idx].copy()
            eg0 = interp_phi(phi, old)
            el0 = lj_energy_of(pos, idx, tree) if use_lj else 0.0

            pos[idx] = (old + DELTA_R*(np.random.rand(3)-.5)) % L
            eg1 = interp_phi(phi, pos[idx])
            el1 = lj_energy_of(pos, idx, tree) if use_lj else 0.0

            de = (el1-el0) + (eg1-eg0)
            if de < 0 or np.random.rand() < np.exp(-de/kBT):
                sa += 1
            else:
                pos[idx] = old
        na += sa; nt += nm

        if step % 40 == 0:
            tree = cKDTree(pos, boxsize=L)
            rho = deposit_cic(pos)
            phi = poisson_fft(rho)

        if step >= N_EQUIL and step % 30 == 0:
            eg = np.sum(phi*rho)*GS**3/2
            if use_lj:
                el = total_lj_energy(pos, tree)
                energies.append((el + eg)/N)
            else:
                energies.append(eg/N)

    acc = na/max(nt,1)
    em = np.mean(energies) if energies else 0
    es = np.std(energies) if energies else 0
    rg, gr = compute_gr(pos)
    rr, rp = radial_rho(pos)
    print(f"  {tag} T*={Tstar:5.1f}  E/N={em:8.3f}±{es:.3f}  acc={acc:.3f}")
    return dict(T=Tstar, E_mean=em, E_std=es, acc=acc,
                r_gr=rg, gr=gr, r_rho=rr, rho=rp, pos=pos.copy(),
                energies=energies)

# === Run complet ===
def main():
    print("="*60)
    print(f"TQNT MC : {N} nœuds, σ={SIGMA}, ε={EPSILON}, κ_g={KAPPA_G}")
    print(f"L={L:.1f}, Δr={DELTA_R}, grille={N_GRID}³")
    print(f"Températures : {TEMPERATURES}")
    print("="*60)
    t0 = time.time()

    print("\n--- BARYONIQUE (LJ + gravité P5) ---")
    rb = [run_T(T, True, "BAR") for T in TEMPERATURES]

    print("\n--- MATIÈRE NOIRE (gravité seule, C₆=0) ---")
    rd = [run_T(T, False, "DM ") for T in [0.5, 2.0, 10.0]]

    print(f"\nTotal : {time.time()-t0:.0f}s")
    plot(rb, rd)
    summary(rb, rd)

def plot(rb, rd):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(17,14))
    fig.suptitle(f'Thermodynamique émergente TQNT — {N} nœuds, '
                 f'σ={SIGMA}, ε={EPSILON}, κ_g={KAPPA_G}',
                 fontsize=13, fontweight='bold')

    cm = plt.cm.coolwarm
    ts = [r['T'] for r in rb]
    tn = [(np.log10(t)-np.log10(min(ts)))/(np.log10(max(ts))-np.log10(min(ts))) for t in ts]

    # g(r) baryonique
    ax = axes[0,0]
    for i,r in enumerate(rb):
        ax.plot(r['r_gr']/SIGMA, r['gr'], color=cm(tn[i]), lw=1, label=f"T*={r['T']}")
    ax.axhline(1, color='gray', ls=':', lw=.5)
    ax.set_xlabel('r / σ'); ax.set_ylabel('g(r)')
    ax.set_title('g(r) baryonique'); ax.set_xlim(.8,5); ax.legend(fontsize=6,ncol=2)

    # g(r) matière noire
    ax = axes[0,1]
    for r in rd:
        ax.plot(r['r_gr']/SIGMA, r['gr'], lw=1.2, label=f"T*={r['T']}")
    ax.axhline(1, color='gray', ls=':', lw=.5)
    ax.set_xlabel('r / σ'); ax.set_ylabel('g(r)')
    ax.set_title('g(r) matière noire (C₆=0)'); ax.set_xlim(.8,5); ax.legend(fontsize=8)

    # E(T) + C_V
    ax = axes[0,2]
    T = np.array([r['T'] for r in rb])
    E = np.array([r['E_mean'] for r in rb])
    dE = np.array([r['E_std'] for r in rb])
    ax.errorbar(T, E, yerr=dE, fmt='o-', color='blue', ms=4, label='E/N')
    ax.set_xscale('log'); ax.set_xlabel('T*'); ax.set_ylabel('E/N')
    ax.set_title('Énergie + C_V'); ax.legend(loc='lower right')
    ax2 = ax.twinx()
    if len(T)>2:
        cv = np.gradient(E,T)
        ax2.plot(T, cv, 's--', color='red', ms=3)
        ax2.set_ylabel('C_V/k_B', color='red')

    # ρ(r) baryonique
    ax = axes[1,0]; rho0 = N/L**3
    for r in rb:
        if r['T'] in (0.5, 2.0, 30.0):
            ax.plot(r['r_rho'], r['rho']/rho0, lw=1.2, label=f"T*={r['T']}")
    ax.axhline(1, color='gray', ls=':', lw=.5)
    ax.set_xlabel('r'); ax.set_ylabel('ρ/ρ_moy'); ax.set_title('Profil radial baryon'); ax.legend(fontsize=8)

    # ρ(r) DM
    ax = axes[1,1]
    for r in rd:
        ax.plot(r['r_rho'], r['rho']/rho0, lw=1.2, label=f"T*={r['T']}")
    ax.axhline(1, color='gray', ls=':', lw=.5)
    ax.set_xlabel('r'); ax.set_ylabel('ρ/ρ_moy'); ax.set_title('Profil radial DM'); ax.legend(fontsize=8)

    # Comparaison T*=2
    ax = axes[1,2]
    for r in rb:
        if r['T']==2.0: ax.plot(r['r_gr']/SIGMA, r['gr'], 'b-', lw=1.5, label='Baryon')
    for r in rd:
        if r['T']==2.0: ax.plot(r['r_gr']/SIGMA, r['gr'], 'r--', lw=1.5, label='DM')
    ax.axhline(1, color='gray', ls=':', lw=.5)
    ax.set_xlabel('r/σ'); ax.set_ylabel('g(r)'); ax.set_title('Baryon vs DM (T*=2)')
    ax.set_xlim(.8,5); ax.legend()

    # Snapshots
    for k, Ts in enumerate([0.5, 2.0, 30.0]):
        ax = axes[2,k]
        for r in rb:
            if r['T']==Ts:
                p = r['pos']
                n = min(2000,N)
                idx = np.random.choice(N,n,replace=False)
                ax.scatter(p[idx,0], p[idx,1], s=.4, alpha=.5, c='blue')
        ax.set_xlim(0,L); ax.set_ylim(0,L); ax.set_aspect('equal')
        phase = {0.5:'AMAS',2.0:'LIQUIDE',30.0:'GAZ'}.get(Ts,'')
        ax.set_title(f'T*={Ts} — {phase}')

    plt.tight_layout()
    p = os.path.join(OUT, 'mc_knots_gravity_lj.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Figure : {p}")
    plt.close()

def summary(rb, rd):
    print("\n=== Baryonique ===")
    print(f"{'T*':>6} {'E/N':>10} {'±':>8} {'acc':>6} {'Phase':>8}")
    for r in rb:
        T=r['T']; ph='amas' if T<=.5 else ('liquide' if T<=8 else 'gaz')
        print(f"{T:6.1f} {r['E_mean']:10.3f} {r['E_std']:8.3f} {r['acc']:6.3f} {ph:>8}")
    print("\n=== Matière noire (C₆=0) ===")
    for r in rd:
        print(f"  T*={r['T']:5.1f}  E/N={r['E_mean']:8.3f}  acc={r['acc']:.3f}")

if __name__=='__main__':
    main()
