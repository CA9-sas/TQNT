"""Koide r=√2 via Z₃ tight-binding of the trefoil χ field.
Spec: docs/superpowers/specs/2026-07-01-koide-r-sqrt2-tight-binding-design.md
Plan: docs/superpowers/plans/2026-07-01-koide-r-sqrt2-tight-binding.md

Model: the 3 charged leptons are the 3 Z₃ tight-binding modes of the trefoil,
  √m_k = ε + 2t cos(2πk/3 + φ),  r = 2t/ε.
ε (on-site) and t (inter-lobe hopping at the crossings) are computed ONLY from
the trefoil geometry and the χ mode — NEVER from the lepton masses. So r is a
prediction. r = √2  ⟺  t/ε = 1/√2.
"""
import numpy as np

NT = 3000  # curve samples


def trefoil(t):
    """Standard 3-fold-symmetric trefoil immersion."""
    x = np.sin(t) + 2*np.sin(2*t)
    y = np.cos(t) - 2*np.cos(2*t)
    z = -np.sin(3*t)
    return np.stack([x, y, z], axis=-1)


def _sample():
    t = np.linspace(0, 2*np.pi, NT, endpoint=False)
    return t, trefoil(t)


def _arclength(pts):
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    return seg, np.concatenate([[0], np.cumsum(seg)])


def crossings(pts, sep_frac=0.15):
    """Find the 3 self-crossings via local minima of the self-approach distance
    (arclength separation > sep_frac of the loop). The trefoil has 6 such minima
    (each of 3 crossings seen from both strands); dedupe to 3 unordered pairs.
    Returns [(i, j, gap), ...] (3 entries; all gaps equal by Z₃)."""
    N = len(pts)
    _, arc = _arclength(pts)
    L = arc[-1]
    mind = np.empty(N)
    minj = np.empty(N, dtype=int)
    for i in range(N):
        ds = np.abs(arc[:N] - arc[i])
        ds = np.minimum(ds, L - ds)
        d = np.linalg.norm(pts - pts[i], axis=1).copy()
        d[ds <= sep_frac*L] = np.inf
        minj[i] = int(np.argmin(d))
        mind[i] = d[minj[i]]
    w = max(1, int(N*0.03))
    locs = [i for i in range(N)
            if mind[i] == min(mind[(i+k) % N] for k in range(-w, w+1))]
    locs = sorted(set(locs))
    groups = []
    for i in locs:
        if groups and (i - groups[-1][-1]) <= w:
            groups[-1].append(i)
        else:
            groups.append([i])
    centers = [g[len(g)//2] for g in groups]
    seen, out = set(), []
    for c in centers:
        j = int(minj[c])
        key = tuple(sorted((c, j)))
        if key not in seen:
            seen.add(key)
            out.append((c, j, float(mind[c])))
    return out


def frenet_UEM(t, pts):
    """(U_kappa, U_tau) = ∮κ²ds, ∮τ²ds via finite differences."""
    dt = t[1]-t[0]
    d1 = np.gradient(pts, dt, axis=0)
    d2 = np.gradient(d1, dt, axis=0)
    d3 = np.gradient(d2, dt, axis=0)
    speed = np.linalg.norm(d1, axis=1)
    cr = np.cross(d1, d2)
    ncr = np.linalg.norm(cr, axis=1)
    kappa = ncr / speed**3
    tau = np.einsum('ij,ij->i', cr, d3) / (ncr**2 + 1e-30)
    ds = speed*dt
    return float(np.sum(kappa**2*ds)), float(np.sum(tau**2*ds))


def writhe(pts):
    """Gauss double-integral writhe: W = (1/4π) Σ_ij (dr_i × dr_j)·r_ij / |r_ij|³."""
    N = len(pts)
    dr = np.roll(pts, -1, axis=0) - pts
    mid = pts + dr/2
    W = 0.0
    for i in range(N):
        rij = mid[i] - mid
        rij[i] = np.array([1e9, 1e9, 1e9])
        crn = np.cross(dr[i], dr)
        W += np.sum(np.einsum('kj,kj->k', crn, rij) / np.linalg.norm(rij, axis=1)**3)
    return W/(4*np.pi)


def chi_width(mode='geom'):
    """χ transverse mode width ξ (same length units as the curve)."""
    _, pts = _sample()
    seg, _ = _arclength(pts)
    L = float(np.sum(seg))
    if mode == 'geom':
        return 0.2 * L/(2*np.pi)
    elif mode == 'fem':
        import sys
        sys.path.insert(0, '/home/euk9/projets/physique_nouvelle')
        from simulation.solver import solve_knot
        res = solve_knot("3_1", n_coarse=24, epsilon=0.2, n_eigenvalues=1)
        P = res["points"]
        rho = np.abs(res["eigenvectors"][:, 0])**2
        ctr = np.average(P, axis=0, weights=rho)
        r = np.linalg.norm(P-ctr, axis=1)
        order = np.argsort(r)
        rho_s = rho[order]
        r_s = r[order]
        peak = rho_s[:max(5, len(rho_s)//50)].mean()
        idx = int(np.argmax(rho_s < peak/np.e))
        return float(r_s[idx]) if idx > 0 else 0.2*L/(2*np.pi)
    raise ValueError(mode)


def compute_r(overlap='gauss', xi_mode='geom'):
    t, pts = _sample()
    cr = crossings(pts)
    d = float(np.mean([g for _, _, g in cr])) if cr else float('nan')
    xi = chi_width(xi_mode)
    if overlap == 'gauss':
        toe = np.exp(-d**2/(4*xi**2))
    elif overlap == 'exp':
        toe = np.exp(-d/xi)
    else:
        raise ValueError(overlap)
    r = 2*toe
    Wr = writhe(pts)
    delta_pred = (2*np.pi*Wr) % (2*np.pi/3)
    return dict(r=r, t_over_eps=toe, d=d, xi=xi, delta_pred=delta_pred, writhe=Wr)


def robustness():
    print("\n--- ROBUSTNESS (r must be stable to be predictive) ---")
    base = chi_width('geom')
    _, pts = _sample()
    cr = crossings(pts)
    d = float(np.mean([g for _, _, g in cr]))
    rows = []
    for ov in ('gauss', 'exp'):
        for fx, lab in ((0.8, '0.8xi'), (1.0, 'xi  '), (1.2, '1.2xi')):
            xi = base*fx
            toe = np.exp(-d**2/(4*xi**2)) if ov == 'gauss' else np.exp(-d/xi)
            rows.append((ov, lab, 2*toe))
    for ov, lab, r in rows:
        print(f"  [{ov:5}] {lab}: r={r:.4f}")
    rr = [r for *_, r in rows]
    print(f"  r range: {min(rr):.3f} - {max(rr):.3f}  (spread x{max(rr)/min(rr):.2f})")
    if max(rr)/min(rr) > 2:
        print("  -> ILL-DEFINED: r varies >x2 with choices; model not predictive.")


if __name__ == "__main__":
    t, pts = _sample()
    seg, arc = _arclength(pts)
    L = arc[-1]

    # --- Task 1 ---
    cr = crossings(pts)
    gaps = sorted(g for _, _, g in cr)
    print(f"[T1] crossings found: {len(cr)}; gaps = {[f'{g:.4f}' for g in gaps]}; L={L:.3f}")
    assert len(cr) == 3, f"expected 3 crossings, got {len(cr)}"
    assert (max(gaps)-min(gaps))/np.mean(gaps) < 0.05, "Z3: gaps not equal within 5%"
    print("[T1] OK: 3 Z3-equal crossings")

    # --- Task 2 ---
    Uk, Ut = frenet_UEM(t, pts)
    print(f"[T2] U_kappa={Uk:.3f}, U_tau={Ut:.3f}, U_EM={Uk+Ut:.3f}, U_k/U_t={Uk/Ut:.2f}")
    Wr = writhe(pts)
    print(f"[T2] writhe = {Wr:.3f}")

    # --- Task 3 ---
    xi_g = chi_width('geom')
    print(f"[T3] xi (geom) = {xi_g:.4f}")
    assert 0 < xi_g < L/2, "xi out of range"

    # --- Task 4 ---
    for ov in ('gauss', 'exp'):
        out = compute_r(ov, 'geom')
        print(f"[T4] [{ov:5}] d={out['d']:.4f} xi={out['xi']:.4f} t/e={out['t_over_eps']:.4f} "
              f"r={out['r']:.4f}  (sqrt2={np.sqrt(2):.4f}); delta_pred={out['delta_pred']:.4f} (2/9={2/9:.4f})")

    # --- Task 5 ---
    robustness()
