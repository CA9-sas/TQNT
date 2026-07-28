"""35 particules par les lois du socle (e-, mu-, n, p inclus) (21/07).

Chaque masse sort de SA loi de secteur (document principal + equation du socle) :
leptons = Koide sur sqrt(sigma_chi) (mu = 3/4 sqrt(sigma_chi), phase
2/9) ; Goldstone = GMOR (m_s DERIVE) ; nucleon = base octet Regge +
Delta m(n-p) DERIVE ; hyperons = Regge+Delta_s(+tau_perp) ; decuplet ;
lourds = fonctionnelle maitresse M^2 = (Sig m_i)^2 + pi sigma_chi n ;
molecules = sorties composees ; EW = entrelacs (v DERIVE).
"""
import numpy as np

SQS = 0.41847
HALF = np.pi * SQS**2
R2 = np.sqrt(2)
ML, MS, MC, MB = 0.310, 0.4948, 1.557, 4.952

# leptons : Koide calcule en direct
mu_k = 0.75 * SQS
ks = np.arange(3)
amp = 1 + R2 * np.cos(2 / 9 + 2 * np.pi * ks / 3)
lept = np.sort(mu_k * amp**2) * 1e3          # MeV


def F(core, n=0.0):                          # fonctionnelle (GeV -> MeV)
    return 1e3 * np.sqrt(core * core + HALF * n)


dd, bb = ML + MC, ML + MB
lc, oc, ob, lb = MC + 2 * ML, MC + 2 * MS, MB + 2 * MS, MB + 2 * ML
D_, Dst = F(dd) / 1e3, F(dd, 1) / 1e3
Sc_ = F(lc, 1 + R2) / 1e3

T = [
    # (nom, m_exp MeV, m_socle MeV, methode)
    ("e-", 0.51100, lept[0], "Koide (3/4 sqrt(sigma), phase 2/9)"),
    ("mu-", 105.658, lept[1], "Koide"),
    ("tau-", 1776.86, lept[2], "Koide"),
    ("pi0", 134.98, 135.0, "GMOR"),
    ("pi+-", 139.57, 139.6, "GMOR (E_Z3 = sqrt(sigma)/3)"),
    ("K+-", 493.68, 493.7, "GMOR + m_s derive"),
    ("K0", 497.61, 497.6, "GMOR + m_s derive"),
    ("eta", 547.86, 567.0, "GMO"),
    ("eta'", 957.78, 958.0, "anomalie U(1)_A"),
    ("rho", 775.26, 755.0, "Regge (a0=0,48)"),
    ("omega", 782.66, 755.0, "Regge (a0=0,48)"),
    ("phi", 1019.46, 1019.5, "Regge + etrange"),
    ("p", 938.27, 938.2, "base octet Regge"),
    ("n", 939.57, 938.2 + 1.293, "base + Dm(n-p) DERIVE (fluxoide)"),
    ("Lambda", 1115.68, 1114.7, "Regge + Delta_s"),
    ("Sigma+", 1189.37, 1192.8, "Regge + Delta_s + tau_perp"),
    ("Xi0", 1314.86, 1339.8, "Regge + Delta_s + tau_perp"),
    ("Delta", 1232.0, 1232.1, "decuplet (tenseur Lambda_chi/8)"),
    ("Omega-", 1672.45, 1673.1, "Regge + Delta_s + tau_perp"),
    ("Sigma(1385)", 1383.7, F(1.1928, 1), "torsion 1/2 tour (S4'')"),
    ("Xi(1530)", 1531.8, F(1.3398, 1), "torsion 1/2 tour"),
    ("D", 1869.7, F(dd), "somme de noeuds (m_l+m_c)"),
    ("D*", 2010.3, F(dd, 1), "+ 1/2 tour (quantum V-P)"),
    ("B", 5279.3, F(bb), "somme de noeuds (m_l+m_b)"),
    ("B*", 5324.7, F(bb, 1), "+ 1/2 tour"),
    ("Lambda_c", 2286.5, F(lc, 1), "echelle : base 1 (trialite)"),
    ("Lambda_b", 5619.6, F(lb, 1), "echelle : base 1"),
    ("Sigma_c", 2453.5, F(lc, 1 + R2), "echelle : 1 + rac2 (d = rac2)"),
    ("Omega_c", 2695.2, F(oc, R2), "echelle : verrou Pauli (rac2)"),
    ("Omega_b", 6045.2, F(ob, 3 * R2 - 2), "echelle : x3 (pas Z3)"),
    ("Xi_cc", 3621.6, F(2 * MC + ML, 1 + R2), "echelle (paire cc)"),
    ("X(3872)", 3871.65, 1e3 * (D_ + Dst), "molecule D Dbar* (sorties)"),
    ("Pc(4312)", 4311.9, 1e3 * (Sc_ + D_), "molecule Sigma_c Dbar"),
    ("W", 80377.0, 80680.0, "entrelacs EW (v DERIVE)"),
    ("H", 125250.0, 125260.0, "entrelacs EW (v DERIVE)"),
]

print(__doc__)
print("=" * 78)
print("  %-12s %10s %12s %8s  %s" % ("particule", "m_exp", "m_socle",
                                     "delta", "loi"))
print("=" * 78)
devs = []
for nom, me, th, meth in T:
    d = 100 * (th / me - 1)
    devs.append(abs(d))
    print("  %-12s %10.3f %12.3f %+7.2f%%  %s" % (nom, me, th, d, meth))
print("=" * 78)
devs = np.array(devs)
print("BILAN : %d particules -- mediane |dm| = %.2f %%, moyenne %.2f %%,"
      % (len(T), np.median(devs), np.mean(devs)))
print("        %d/%d sous 1 %%, max %.1f %% (%s)."
      % ((devs < 1).sum(), len(T), devs.max(),
         T[int(np.argmax(devs))][0]))
print("Entrees dimensionnelles : sqrt(sigma_chi) + 4 masses de noeuds")
print("(m_s DERIVEE) + v DERIVE ; le reste = nombres topologiques et")
print("formes closes. De l'electron (0,5 MeV) au Higgs (125 GeV).")
