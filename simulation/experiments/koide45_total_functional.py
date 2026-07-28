import sys
sys.path.insert(0, '/home/euk9/projets/physique_nouvelle')
import numpy as np
from scipy.spatial import cKDTree
from theory.schrodinger import generate_mesh, assemble_system, solve_eigenproblem
from theory.knot_catalog import knot_curve
from theory.postulates import KAPPA

curve = knot_curve('3_1', 500)
tree = cKDTree(curve)
pts, tets, boundary = generate_mesh(curve, n_coarse=26, epsilon=0.2, refine=True)
H0, M0 = assemble_system(pts, tets, boundary, '3_1', tree, energy=0.0, epsilon=0.2)
evals, evf = solve_eigenproblem(H0, M0, n_eigenvalues=4)
free = ~boundary
print('evals :', np.round(evals, 4), flush=True)

# modes M-normalises (convention solveur)
def scatter(v):
    f = np.zeros(len(pts)); f[free] = v; return f
psi0 = scatter(evf[:, 0]); za = scatter(evf[:, 1]); zb = scatter(evf[:, 2])
# amplitude de la retroaction pour psi0 (convention exacte du solveur)
rho_max = (psi0**2).max()
print('kappa = %.2f ; max kappa|psi0|^2 = %.3f (taille de la retroaction)' % (KAPPA, KAPPA*rho_max), flush=True)

def rayleigh_dressed(phi):
    omega_sq = 1.0 + KAPPA * phi**2
    H, M = assemble_system(pts, tets, boundary, '3_1', tree, energy=0.0,
                           epsilon=0.2, omega_sq=omega_sq, metric_mode='conformal')
    x = phi[free]
    return float(x @ (H @ x)) / float(x @ (M @ x))

# famille melange (orientation generique du doublet ; moyenne d'orbite Z3 : 3 phases)
import itertools
for th_deg in (0, 15, 30, 45, 60):
    th = np.radians(th_deg)
    Es = []
    for k in range(3):
        ph = 2*np.pi*k/3
        z = np.cos(ph)*za + np.sin(ph)*zb
        z = z / np.sqrt(float(z[free] @ (M0 @ z[free])))
        phi = np.cos(th)*psi0 + np.sin(th)*z
        Es.append(rayleigh_dressed(phi))
    print('theta = %2d deg : E_habille (moyenne Z3) = %.6f' % (th_deg, np.mean(Es)), flush=True)
print('FINI', flush=True)
